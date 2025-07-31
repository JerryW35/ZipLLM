use crate::types::*;
use crate::traits::*;
use crate::bitx::bitx_compress;
use anyhow::Result;
use std::collections::{HashMap, BTreeMap};
use rayon::prelude::*;
use crate::bitx::bitx_bytes::zstd_compress_data;
use log::{info, warn, debug};

#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub total_tensors: usize,
    pub paired_tensors: usize,
    pub solo_tensors: usize,
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f64,
}

#[derive(Debug)]
pub struct CompressionResult {
    pub compression_stats: CompressionStats,
}

pub struct CompressionEngine {
    storage: std::sync::Arc<crate::storage::FileSystemStorage>,
}

impl CompressionEngine {
    pub fn new(storage: std::sync::Arc<crate::storage::FileSystemStorage>) -> Self {
        Self { 
            storage,
        }
    }

    pub fn compress_models(
        &self,
        base_model_id: &str,
        finetune_model_id: &str,
    ) -> Result<CompressionResult> {
        use std::time::Instant;
        let start_time = Instant::now();
        info!("Starting compression: base={base_model_id}, finetune={finetune_model_id}");
        
        // Step 1: Get tensor information for both models
        let step1_start = Instant::now();
        let base_tensors: HashMap<String, TensorInfo> = self.get_model_tensor_info_with_details(base_model_id)?;
        let finetune_tensors: HashMap<String, TensorInfo> = self.get_model_tensor_info_with_details(finetune_model_id)?;
        let step1_time = step1_start.elapsed().as_secs_f64();
        info!("Base model tensors: {}", base_tensors.len());
        info!("Finetune model tensors: {}", finetune_tensors.len());
        info!("Step 1 (Get tensor info) time: {:.4}s", step1_time);
        
        // Step 2: Sort tensors by name
        let step2_start = Instant::now();
        let base_sorted = base_tensors.into_iter().collect::<BTreeMap<String, TensorInfo>>();
        let finetune_sorted = finetune_tensors.into_iter().collect::<BTreeMap<String, TensorInfo>>();
        let step2_time = step2_start.elapsed().as_secs_f64();
        info!("Step 2 (Sort tensors) time: {:.4}s", step2_time);
        
        // Step 3: Form tensor pairs and identify solo tensors
        let step3_start = Instant::now();
        let (tensor_pairs, solo_tensors) = self.form_tensor_pairs(&base_sorted, &finetune_sorted)?;
        let step3_time = step3_start.elapsed().as_secs_f64();
        info!("Tensor pairs: {}", tensor_pairs.len());
        info!("Solo tensors: {}", solo_tensors.len());
        info!("Step 3 (Form pairs) time: {:.4}s", step3_time);
        
        // Step 4: Compress tensor pairs using BitX
        let step4_start = Instant::now();
        let (bitx_results, bitx_total_original, bitx_total_compressed) = 
            self.compress_tensor_pairs_optimized(&tensor_pairs)?;
        let step4_time = step4_start.elapsed().as_secs_f64();
        info!("Step 4 (BitX compression) time: {:.4}s", step4_time);
        
        // Step 5: Compress solo tensors using Zstd if having solo tensors
        let step5_start = Instant::now();
        let (zstd_results, zstd_total_original, zstd_total_compressed) = 
            self.compress_solo_tensors(&solo_tensors)?;
        let step5_time = step5_start.elapsed().as_secs_f64();
        info!("Step 5 (Zstd compression) time: {:.4}s", step5_time);
        
        // Step 6: Store compression results
        let step6_start = Instant::now();
        self.store_compression_results(&bitx_results, &zstd_results)?;
        let step6_time = step6_start.elapsed().as_secs_f64();
        info!("Step 6 (Store results) time: {:.4}s", step6_time);
        
        let total_original = bitx_total_original + zstd_total_original;
        let total_compressed = bitx_total_compressed + zstd_total_compressed;
        
        let stats = CompressionStats {
            total_tensors: tensor_pairs.len() * 2 + solo_tensors.len(),
            paired_tensors: tensor_pairs.len() * 2,
            solo_tensors: solo_tensors.len(),
            original_size: total_original,
            compressed_size: total_compressed,
            compression_ratio: if total_original > 0 { 
                total_compressed as f64 / total_original as f64 
            } else { 
                1.0 
            },
        };

        let total_time = start_time.elapsed().as_secs_f64();
        info!("Total compression time: {:.4}s", total_time);

        Ok(CompressionResult {
            compression_stats: stats,
        })
    }

    pub fn compress_base_model(&self, model_id: &str) -> Result<CompressionResult> {
        info!("Starting base model compression for: {model_id}");
        
        // Step 1: Get all tensor information for the model
        let model_tensors = self.get_model_tensor_info_with_details(model_id)?;
        
        info!("Base model tensors: {}", model_tensors.len());
        
        // Step 2: Convert to vector for processing
        let all_tensors: Vec<TensorInfo> = model_tensors.into_values().collect();
        
        // Step 3: Compress all tensors using Zstd
        let (zstd_results, total_original, total_compressed) = 
            self.compress_solo_tensors(&all_tensors)?;
        
        // Step 4: Store compression results
        self.store_compression_results(&[], &zstd_results)?;
        
        let stats = CompressionStats {
            total_tensors: all_tensors.len(),
            paired_tensors: 0,
            solo_tensors: all_tensors.len(),
            original_size: total_original,
            compressed_size: total_compressed,
            compression_ratio: if total_original > 0 { 
                total_compressed as f64 / total_original as f64 
            } else { 
                1.0 
            },
        };

        Ok(CompressionResult {
            compression_stats: stats,
        })
    }
    
    fn get_model_tensor_info_with_details(&self, model_id: &str) -> Result<HashMap<String, TensorInfo>> {
        let tensor_infos = self.storage.get_model_tensor_info(model_id)
            .map_err(|e| anyhow::anyhow!("Failed to get model tensor info: {}", e))?;
        
        let mut tensor_map = HashMap::new();
        for tensor_info in tensor_infos {
            tensor_map.insert(tensor_info.name.clone(), tensor_info);
        }
        
        Ok(tensor_map)
    }
    
    fn form_tensor_pairs(
        &self,
        base_tensors: &BTreeMap<String, TensorInfo>,
        finetune_tensors: &BTreeMap<String, TensorInfo>,
    ) -> Result<(Vec<TensorPair>, Vec<TensorInfo>)> {
        let mut tensor_pairs = Vec::new();
        let mut solo_tensors = Vec::new();
        
        let base_names: Vec<_> = base_tensors.keys().cloned().collect();
        let finetune_names: Vec<_> = finetune_tensors.keys().cloned().collect();
        
        debug!("Base tensor names count: {}", base_names.len());
        debug!("Finetune tensor names count: {}", finetune_names.len());
        
        for (tensor_name, finetune_tensor) in finetune_tensors {
            if let Some(base_tensor) = base_tensors.get(tensor_name) {
                // Check if tensors are compatible for pairing
                if self.tensors_compatible(base_tensor, finetune_tensor) {
                    tensor_pairs.push(TensorPair {
                        base_tensor: base_tensor.clone(),
                        finetune_tensor: finetune_tensor.clone(),
                    });
                } else {
                    warn!("Incompatible tensors for pairing: {tensor_name} (shape/dtype mismatch)");
                    solo_tensors.push(finetune_tensor.clone());
                }
            } else {
                // Tensor only exists in finetune model
                solo_tensors.push(finetune_tensor.clone());
            }
        }
        
        Ok((tensor_pairs, solo_tensors))
    }
    
    fn tensors_compatible(&self, base: &TensorInfo, finetune: &TensorInfo) -> bool {
        base.shape == finetune.shape && base.dtype == finetune.dtype
    }
    
    
    fn compress_solo_tensors(&self, solo_tensors: &[TensorInfo]) -> Result<(Vec<CompressedTensorMetadata>, u64, u64)> {
        info!("Compressing {} solo tensors using Zstd", solo_tensors.len());
        
        let results: Vec<_> = solo_tensors
            .par_iter()
            .map(|tensor| self.compress_single_solo(tensor))
            .collect::<Result<Vec<_>>>()?;
        
        let total_original: u64 = results.iter().map(|r| r.original_size).sum();
        let total_compressed: u64 = results.iter().map(|r| r.compressed_size).sum();
        
        info!("Zstd compression completed: {:.2}% of original size", 
                (total_compressed as f64 / total_original as f64) * 100.0);
        
        Ok((results, total_original, total_compressed))
    }
    
    fn compress_single_solo(&self, tensor: &TensorInfo) -> Result<CompressedTensorMetadata> {
        // Load tensor data
        let tensor_data = self.storage.load_tensor(&tensor.hash)
            .map_err(|e| anyhow::anyhow!("Failed to load tensor: {}", e))?;
        
        debug!("Compressing solo tensor: {} ({} bytes)", tensor.name, tensor_data.len());
        
        // Perform Zstd compression in parallel
        let compressed_data = zstd_compress_data(&tensor_data, 3);
        
        let original_size = tensor_data.len() as u64;
        let compressed_size = compressed_data.len() as u64;
        
        // Store compressed data
        self.storage.store_compressed_tensor(&tensor.hash, &compressed_data)
            .map_err(|e| anyhow::anyhow!("Failed to store compressed tensor: {}", e))?;
        
        Ok(CompressedTensorMetadata {
            name: tensor.name.clone(),
            original_hash: tensor.hash.clone(),
            compressed_hash: tensor.hash.clone(),
            compression_type: CompressionType::ZstdSolo,
            original_size,
            compressed_size,
            base_tensor_hash: None,
        })
    }
    
    fn store_compression_results(
        &self,
        bitx_results: &[CompressedTensorMetadata],
        zstd_results: &[CompressedTensorMetadata],
    ) -> Result<()> {
        info!("Storing compression metadata...");
        
        for metadata in bitx_results.iter().chain(zstd_results.iter()) {
            self.storage.store_compressed_metadata(metadata)
                .map_err(|e| anyhow::anyhow!("Failed to store compressed metadata: {}", e))?;
        }
        
        info!("Compression metadata stored successfully");
        Ok(())
    }

    /// Optimized tensor pair compression with batch loading
    fn compress_tensor_pairs_optimized(&self, tensor_pairs: &[TensorPair]) -> Result<(Vec<CompressedTensorMetadata>, u64, u64)> {
        info!("Compressing {} tensor pairs using optimized BitX", tensor_pairs.len());
        
        // Step 1: Batch load all tensor data
        let load_start = std::time::Instant::now();
        let tensor_data = self.batch_load_tensor_pairs(tensor_pairs)?;
        let load_duration = load_start.elapsed();
        info!("****** Batch load completed in {:.2?}", load_duration);
        
        // Step 2: Parallel compression and storage
        let compress_start = std::time::Instant::now();
        let results: Result<Vec<(CompressedTensorMetadata, Vec<u8>)>> = tensor_pairs
            .par_iter()
            .zip(tensor_data.par_iter())
            .map(|(pair, (base_data, finetune_data))| {
                // Compress the pair
                let (compressed_exp, compressed_sm) = bitx_compress(base_data, finetune_data);
                
                // Combine compressed data with pre-allocation
                let total_size = 8 + compressed_exp.len() + compressed_sm.len();
                let mut combined_compressed = Vec::with_capacity(total_size);
                combined_compressed.extend_from_slice(&(compressed_exp.len() as u64).to_le_bytes());
                combined_compressed.extend_from_slice(&compressed_exp);
                combined_compressed.extend_from_slice(&compressed_sm);
                
                let original_size = finetune_data.len() as u64;
                let compressed_size = combined_compressed.len() as u64;
                
                let metadata = CompressedTensorMetadata {
                    name: pair.finetune_tensor.name.clone(),
                    original_hash: pair.finetune_tensor.hash.clone(),
                    compressed_hash: pair.finetune_tensor.hash.clone(),
                    compression_type: CompressionType::BitX,
                    original_size,
                    compressed_size,
                    base_tensor_hash: Some(pair.base_tensor.hash.clone()),
                };
                
                Ok((metadata, combined_compressed))
            })
            .collect();
        
        let compression_results = results?;
        let compress_duration = compress_start.elapsed();
        info!("****** Parallel compression completed in {:.2?}", compress_duration);
        
        // Step 3: Parallel storage
        let store_start = std::time::Instant::now();
        compression_results.par_iter().try_for_each(|(metadata, compressed_data)| {
            self.storage.store_compressed_tensor(&metadata.compressed_hash, compressed_data)
                .map_err(|e| anyhow::anyhow!("Failed to store compressed tensor: {}", e))
        })?;
        let store_duration = store_start.elapsed();
        info!("****** Parallel storage completed in {:.2?}", store_duration);
        
        let metadata_results: Vec<CompressedTensorMetadata> = compression_results
            .into_iter()
            .map(|(metadata, _)| metadata)
            .collect();
        
        let total_original: u64 = metadata_results.iter().map(|r| r.original_size).sum();
        let total_compressed: u64 = metadata_results.iter().map(|r| r.compressed_size).sum();
        
        info!("Optimized BitX compression completed: {:.2}% of original size", 
              (total_compressed as f64 / total_original as f64) * 100.0);
        info!("****** Total time: {:.2?}", load_duration + compress_duration + store_duration);
        
        Ok((metadata_results, total_original, total_compressed))
    }

    /// Batch load all tensor pairs to reduce IO operations  
    fn batch_load_tensor_pairs(&self, tensor_pairs: &[TensorPair]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        info!("Batch loading {} tensor pairs", tensor_pairs.len());
        
        // Collect all unique hashes to avoid duplicate loads
        let mut unique_hashes: std::collections::HashSet<String> = std::collections::HashSet::new();
        for pair in tensor_pairs {
            unique_hashes.insert(pair.base_tensor.hash.clone());
            unique_hashes.insert(pair.finetune_tensor.hash.clone());
        }
        
        let unique_hashes: Vec<String> = unique_hashes.into_iter().collect();
        info!("Loading {} unique tensors (deduplicating from {} pairs)", unique_hashes.len(), tensor_pairs.len());
        
        // Parallel load all unique tensors
        let loaded_pairs: Result<Vec<(String, Vec<u8>)>> = unique_hashes
            .par_iter()
            .map(|hash| {
                let data = self.storage.load_tensor(hash)
                    .map_err(|e| anyhow::anyhow!("Failed to load tensor {}: {}", hash, e))?;
                Ok((hash.clone(), data))
            })
            .collect();
        
        let hash_to_data: std::collections::HashMap<String, Vec<u8>> = loaded_pairs?
            .into_iter()
            .collect();
        
        // Build result vector maintaining pair order without cloning large data
        let mut result = Vec::with_capacity(tensor_pairs.len());
        for pair in tensor_pairs {
            let base_data = hash_to_data.get(&pair.base_tensor.hash)
                .ok_or_else(|| anyhow::anyhow!("Missing base tensor data"))?
                .clone();
            let finetune_data = hash_to_data.get(&pair.finetune_tensor.hash)
                .ok_or_else(|| anyhow::anyhow!("Missing finetune tensor data"))?
                .clone();
            result.push((base_data, finetune_data));
        }
        
        Ok(result)
    }



}
