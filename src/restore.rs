use crate::traits::*;
use crate::types::*;
use crate::storage::FileSystemStorage;
use crate::bitx::bitx_bytes::{bitx_decompress, zstd_decompress_data};
use std::fs;
use std::path::Path;
use log::{info, warn, debug};
use anyhow::{Result, anyhow};
use std::sync::Arc;
use rayon::prelude::*;
use std::time::Instant;
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::collections::{HashMap, HashSet};

pub struct RestoreEngine {
    storage: Arc<FileSystemStorage>,
}

impl RestoreEngine {
    pub fn new(storage: Arc<FileSystemStorage>) -> Self {
        Self { storage }
    }

    /// Helper function to format throughput information
    fn format_throughput(bytes: usize, duration: std::time::Duration) -> String {
        let seconds = duration.as_secs_f64();
        if seconds > 0.0 {
            let mb_per_sec = (bytes as f64) / (1024.0 * 1024.0) / seconds;
            let gb_per_sec = mb_per_sec / 1024.0;
            if gb_per_sec >= 1.0 {
                format!("{gb_per_sec:.2} GB/s ({seconds:.3}s for {} MB)", bytes as f64 / (1024.0 * 1024.0))
            } else {
                format!("{mb_per_sec:.2} MB/s ({seconds:.3}s for {} MB)", bytes as f64 / (1024.0 * 1024.0))
            }
        } else {
            format!("instant ({} MB)", bytes as f64 / (1024.0 * 1024.0))
        }
    }

    /// Restore all safetensors files for a given model using optimized batch processing
    pub fn restore_model(&self, model_id: &str, output_dir: &str) -> Result<()> {
        let total_start = Instant::now();
        info!("🚀 Starting optimized batch restore for model: {model_id}");

        // Check if model exists
        if !self.storage.exists_model(model_id) {
            return Err(anyhow!("Model '{}' not found in storage", model_id));
        }

        // Load model metadata
        let model_metadata = self.storage.load_model_metadata(model_id)
            .map_err(|e| anyhow!("Failed to load model metadata for '{}': {}", model_id, e))?;

        if !model_metadata.is_processed {
            warn!("Model '{model_id}' is not fully processed, restoration may be incomplete");
        }

        info!("Found {} files to restore for model '{}'", model_metadata.files.len(), model_id);

        // Create output directory
        let output_path = Path::new(output_dir);
        fs::create_dir_all(output_path)
            .map_err(|e| anyhow!("Failed to create output directory '{}': {}", output_dir, e))?;

        // Restore each safetensors file using optimized batch processing
        for (filename, file_hash) in &model_metadata.files {
            info!("🔄 Restoring file: {filename}");
            
            self.restore_safetensors_file_optimized(file_hash, output_path, filename)?;

            info!("✅ Restored: {filename}");
        }

        let total_duration = total_start.elapsed();
        info!("🎉 Optimized batch restore completed in {:.2}s!", total_duration.as_secs_f64());
        Ok(())
    }

    /// Optimized restore with batch tensor loading and parallel decompression
    fn restore_safetensors_file_optimized(&self, file_hash: &str, output_path: &Path, filename: &str) -> Result<()> {
        // Step 1: Load metadata
        let metadata_start = Instant::now();
        let file_metadata = self.storage.load_file_metadata(file_hash)
            .map_err(|e| anyhow!("Failed to load file metadata for hash '{}': {}", file_hash, e))?;

        let original_header = self.storage.load_safetensors_header(file_hash)
            .map_err(|e| anyhow!("Failed to load safetensors header for file '{}': {}", filename, e))?;

        let (tensor_info, total_size) = self.parse_header_with_size_info(&original_header)?;
        let metadata_duration = metadata_start.elapsed();
        
        debug!("🔍 Metadata loading: {:.3}s - Parsed {} tensors, total size: {} MB", 
               metadata_duration.as_secs_f64(), tensor_info.len(), total_size as f64 / (1024.0 * 1024.0));

        // Step 2: Batch load all tensors and compressed metadata
        info!("📦 Batch loading {} tensors...", tensor_info.len());
        let batch_load_start = Instant::now();
        
        let (tensor_data_map, ft_compressed_size) = self.batch_load_all_tensors_with_size(&file_metadata.tensor_hashes)?;
        let batch_load_duration = batch_load_start.elapsed();
        
        let total_loaded_bytes: usize = tensor_data_map.values().map(|data| data.len()).sum();
        info!("✅ Batch loading: {}", Self::format_throughput(total_loaded_bytes, batch_load_duration));

        // Step 3: Parallel decompression
        info!("🔧 Parallel decompression of {} tensors...", tensor_info.len());
        let decomp_start = Instant::now();
        
        let processed_tensors: Result<Vec<_>> = tensor_info
            .par_iter()
            .map(|(tensor_name, offset)| {
                let tensor_data = tensor_data_map.get(tensor_name)
                    .ok_or_else(|| anyhow!("Missing tensor data for: {}", tensor_name))?;
                
                Ok((tensor_name.clone(), *offset, tensor_data.clone()))
            })
            .collect();

        let processed_tensors = processed_tensors?;
        let decomp_duration = decomp_start.elapsed();
        
        // Correct throughput calculation: use only finetune compressed data size
        let ft_decomp_throughput = if decomp_duration.as_secs_f64() > 0.0 && ft_compressed_size > 0 {
            ft_compressed_size as f64 / (1024.0 * 1024.0 * 1024.0) / decomp_duration.as_secs_f64()
        } else {
            0.0
        };
        
        let total_output_bytes: usize = processed_tensors.iter().map(|(_, _, data)| data.len()).sum();
        info!("✅ Decompression completed:");
        info!("   FT compressed data: {:.1} MB", ft_compressed_size as f64 / (1024.0 * 1024.0));
        info!("   Total output: {:.1} MB", total_output_bytes as f64 / (1024.0 * 1024.0));
        info!("   FT decomp throughput: {:.2} GB/s", ft_decomp_throughput);

        // Step 4: Optimized mmap writing
        info!("🚀 Writing {} tensors using parallel mmap...", processed_tensors.len());
        let write_start = Instant::now();
        
        self.write_tensors_mmap(&processed_tensors, &original_header, output_path, filename, total_size)?;
        
        let write_duration = write_start.elapsed();
        info!("✅ Writing: {}", Self::format_throughput(total_size as usize, write_duration));

        // Summary stats
        let total_file_duration = metadata_duration + batch_load_duration + decomp_duration + write_duration;
        info!("**📊 File restore summary for '{}':**", filename);
        info!("   Metadata:      {:.3}s", metadata_duration.as_secs_f64());
        info!("   Batch Load:    {:.3}s ({:.2} GB/s)", 
              batch_load_duration.as_secs_f64(),
              total_loaded_bytes as f64 / (1024.0 * 1024.0 * 1024.0) / batch_load_duration.as_secs_f64());
        info!("   Decompression: {:.3}s ({:.2} GB/s FT throughput)", 
              decomp_duration.as_secs_f64(), ft_decomp_throughput);
        info!("   Writing:       {:.3}s ({:.2} GB/s)", 
              write_duration.as_secs_f64(),
              total_size as f64 / (1024.0 * 1024.0 * 1024.0) / write_duration.as_secs_f64());
        info!("   Total:         {:.3}s ({:.2} GB/s overall)", 
              total_file_duration.as_secs_f64(),
              total_size as f64 / (1024.0 * 1024.0 * 1024.0) / total_file_duration.as_secs_f64());

        Ok(())
    }

    /// Batch load all tensors and return only finetune compressed data size
    fn batch_load_all_tensors_with_size(&self, tensor_hashes: &HashMap<String, String>) -> Result<(HashMap<String, Vec<u8>>, usize)> {
        // Step 1: Collect compressed metadata and calculate FT compressed size
        let metadata_collection_start = Instant::now();
        
        let tensor_metadata: Result<HashMap<String, Option<CompressedTensorMetadata>>> = tensor_hashes
            .par_iter()
            .map(|(tensor_name, tensor_hash)| {
                let metadata = if self.storage.exists_compressed_metadata(tensor_hash) {
                    Some(self.storage.load_compressed_metadata(tensor_hash)
                        .map_err(|e| anyhow!("Failed to load compressed metadata for '{}': {}", tensor_name, e))?)
                } else {
                    None
                };
                Ok((tensor_name.clone(), metadata))
            })
            .collect();
        
        let tensor_metadata = tensor_metadata?;
        let metadata_collection_duration = metadata_collection_start.elapsed();
        debug!("Collected metadata in {:.3}s", metadata_collection_duration.as_secs_f64());

        // Step 2: Calculate finetune compressed data size and identify unique hashes
        let mut unique_tensor_hashes = HashSet::new();
        let mut unique_compressed_hashes = HashSet::new();
        let mut ft_compressed_size = 0usize;  // Only count finetune compressed data
        let mut bitx_count = 0;
        let mut zstd_count = 0;
        let mut dedup_count = 0;
        let mut none_count = 0;

        for (tensor_name, metadata_opt) in &tensor_metadata {
            if let Some(metadata) = metadata_opt {
                match &metadata.compression_type {
                    CompressionType::BitX => {
                        bitx_count += 1;
                        if let Some(base_hash) = &metadata.base_tensor_hash {
                            unique_tensor_hashes.insert(base_hash.clone()); // Base tensor needed
                        }
                        unique_compressed_hashes.insert(metadata.compressed_hash.clone());
                        // Only count the compressed finetune data size (not base tensor)
                        ft_compressed_size += metadata.compressed_size as usize;
                    },
                    CompressionType::ZstdSolo => {
                        zstd_count += 1;
                        unique_compressed_hashes.insert(metadata.compressed_hash.clone());
                        ft_compressed_size += metadata.compressed_size as usize;
                    },
                    CompressionType::Deduplicated { original_hash } => {
                        dedup_count += 1;
                        unique_tensor_hashes.insert(original_hash.clone());
                        // For deduplicated, no actual decompression work, so don't count
                    },
                    CompressionType::None => {
                        none_count += 1;
                        unique_tensor_hashes.insert(metadata.original_hash.clone());
                        // No compression, no decompression work
                    }
                }
            } else {
                none_count += 1;
                unique_tensor_hashes.insert(tensor_hashes[tensor_name].clone());
            }
        }

        info!("📈 Compression stats: BitX:{}, Zstd:{}, Dedup:{}, None:{}", 
              bitx_count, zstd_count, dedup_count, none_count);
        info!("🔄 FT compressed data size: {:.1} MB", ft_compressed_size as f64 / (1024.0 * 1024.0));
        info!("🔄 Loading {} unique tensors + {} compressed files", 
              unique_tensor_hashes.len(), unique_compressed_hashes.len());

        // Step 3: Parallel batch loading of all required data
        let batch_load_start = Instant::now();
        
        let (tensor_data_map, compressed_data_map): (
            Result<HashMap<String, Vec<u8>>>,
            Result<HashMap<String, Vec<u8>>>
        ) = rayon::join(
            || {
                // Load all unique tensor data (base tensors)
                let tensor_pairs: Result<Vec<(String, Vec<u8>)>> = unique_tensor_hashes
                    .par_iter()
                    .map(|hash| {
                        let data = self.storage.load_tensor(hash)
                            .map_err(|e| anyhow!("Failed to load tensor {}: {}", hash, e))?;
                        Ok((hash.clone(), data))
                    })
                    .collect();
                
                Ok(tensor_pairs?.into_iter().collect())
            },
            || {
                // Load all compressed data (finetune tensors)
                let compressed_pairs: Result<Vec<(String, Vec<u8>)>> = unique_compressed_hashes
                    .par_iter()
                    .map(|hash| {
                        let data = self.storage.load_compressed_tensor(hash)
                            .map_err(|e| anyhow!("Failed to load compressed tensor {}: {}", hash, e))?;
                        Ok((hash.clone(), data))
                    })
                    .collect();
                
                Ok(compressed_pairs?.into_iter().collect())
            }
        );

        let tensor_data_map = tensor_data_map?;
        let compressed_data_map = compressed_data_map?;
        let batch_load_duration = batch_load_start.elapsed();

        debug!("Batch load completed in {:.3}s", batch_load_duration.as_secs_f64());

        // Step 4: Parallel decompression with pre-loaded data
        let decompress_start = Instant::now();
        
        let final_tensor_data: Result<HashMap<String, Vec<u8>>> = tensor_metadata
            .par_iter()
            .map(|(tensor_name, metadata_opt)| {
                let tensor_hash = &tensor_hashes[tensor_name];
                
                let decompressed_data = if let Some(metadata) = metadata_opt {
                    self.decompress_tensor_with_preloaded_data(metadata, &tensor_data_map, &compressed_data_map)?
                } else {
                    // No compression, load directly
                    tensor_data_map.get(tensor_hash)
                        .ok_or_else(|| anyhow!("Missing tensor data for: {}", tensor_name))?
                        .clone()
                };
                
                Ok((tensor_name.clone(), decompressed_data))
            })
            .collect();

        let final_tensor_data = final_tensor_data?;
        let decompress_duration = decompress_start.elapsed();
        
        debug!("Decompression completed in {:.3}s", decompress_duration.as_secs_f64());

        Ok((final_tensor_data, ft_compressed_size))
    }

    /// Decompress tensor using pre-loaded data maps
    fn decompress_tensor_with_preloaded_data(
        &self, 
        metadata: &CompressedTensorMetadata,
        tensor_data_map: &HashMap<String, Vec<u8>>,
        compressed_data_map: &HashMap<String, Vec<u8>>
    ) -> Result<Vec<u8>> {
        match &metadata.compression_type {
            CompressionType::BitX => {
                let base_tensor_hash = metadata.base_tensor_hash.as_ref()
                    .ok_or_else(|| anyhow!("BitX compressed tensor missing base tensor hash"))?;
                
                let base_data = tensor_data_map.get(base_tensor_hash)
                    .ok_or_else(|| anyhow!("Missing pre-loaded base tensor data"))?;
                
                let compressed_data = compressed_data_map.get(&metadata.compressed_hash)
                    .ok_or_else(|| anyhow!("Missing pre-loaded compressed data"))?;
                
                // Parse compressed data
                if compressed_data.len() < 8 {
                    return Err(anyhow!("Compressed data too short for BitX format"));
                }

                let exp_len = u64::from_le_bytes(compressed_data[0..8].try_into()?) as usize;
                if compressed_data.len() < 8 + exp_len {
                    return Err(anyhow!("Compressed data truncated"));
                }

                let compressed_exp = &compressed_data[8..8 + exp_len];
                let compressed_sm = &compressed_data[8 + exp_len..];

                Ok(bitx_decompress(base_data, compressed_exp, compressed_sm))
            },

            CompressionType::ZstdSolo => {
                let compressed_data = compressed_data_map.get(&metadata.compressed_hash)
                    .ok_or_else(|| anyhow!("Missing pre-loaded compressed data"))?;
                
                Ok(zstd_decompress_data(compressed_data))
            },

            CompressionType::Deduplicated { original_hash } => {
                tensor_data_map.get(original_hash)
                    .ok_or_else(|| anyhow!("Missing pre-loaded deduplicated tensor data"))
                    .map(|data| data.clone())
            },

            CompressionType::None => {
                tensor_data_map.get(&metadata.original_hash)
                    .ok_or_else(|| anyhow!("Missing pre-loaded uncompressed tensor data"))
                    .map(|data| data.clone())
            }
        }
    }

    /// Optimized mmap writing
    fn write_tensors_mmap(
        &self,
        processed_tensors: &[(String, usize, Vec<u8>)],
        original_header: &[u8],
        output_path: &Path,
        filename: &str,
        total_size: u64
    ) -> Result<()> {
        let output_file_path = output_path.join(filename);
        
        // Remove existing file first
        if output_file_path.exists() {
            fs::remove_file(&output_file_path)
                .map_err(|e| anyhow!("Failed to remove existing file '{}': {}", output_file_path.display(), e))?;
        }
        
        let file = OpenOptions::new()
            .read(true)
            .create(true)
            .write(true)
            .open(&output_file_path)
            .map_err(|e| anyhow!("Failed to create output file: {}", e))?;
        
        file.set_len(total_size)
            .map_err(|e| anyhow!("Failed to set file size: {}", e))?;

        // Memory map the output file
        let mut mmap = unsafe {
            MmapMut::map_mut(&file)
                .map_err(|e| anyhow!("Failed to memory map output file: {}", e))?
        };

        // Copy header first
        let header_size = original_header.len();
        mmap[0..header_size].copy_from_slice(original_header);

        // Get mmap pointer for parallel access
        let mmap_ptr_addr = mmap.as_mut_ptr() as usize;
        let mmap_len = mmap.len();
        
        // Parallel mmap write operations
        let write_results: Result<Vec<()>> = processed_tensors.par_iter()
            .map(|(tensor_name, offset, tensor_data)| {
                let write_offset = header_size + offset;
                let end_offset = write_offset + tensor_data.len();
                
                if end_offset > mmap_len {
                    return Err(anyhow!("Tensor '{}' data exceeds file bounds", tensor_name));
                }
                
                // Direct memory copy using unsafe pointer (zero-copy)
                unsafe {
                    let dst_ptr = (mmap_ptr_addr + write_offset) as *mut u8;
                    std::ptr::copy_nonoverlapping(tensor_data.as_ptr(), dst_ptr, tensor_data.len());
                }
                
                Ok(())
            })
            .collect();

        write_results?;

        // Flush to disk
        mmap.flush()
            .map_err(|e| anyhow!("Failed to flush memory map: {}", e))?;

        Ok(())
    }

    /// Parse header and extract tensor info with size calculation
    fn parse_header_with_size_info(&self, header_data: &[u8]) -> Result<(Vec<(String, usize)>, u64)> {
        if header_data.len() < 8 {
            return Err(anyhow!("Header data too short"));
        }

        let header_len = u64::from_le_bytes(header_data[0..8].try_into()?) as usize;
        if header_data.len() < 8 + header_len {
            return Err(anyhow!("Header data truncated"));
        }

        let header_str = std::str::from_utf8(&header_data[8..8 + header_len])
            .map_err(|e| anyhow!("Invalid UTF-8 in header: {}", e))?;

        let raw_json: serde_json::Value = serde_json::from_str(header_str)
            .map_err(|e| anyhow!("Invalid JSON in header: {}", e))?;

        let mut tensor_with_info = Vec::new();
        let mut max_end_offset = 0u64;

        if let serde_json::Value::Object(obj) = raw_json {
            for (key, value) in obj {
                if key == "__metadata__" {
                    continue; // Skip metadata field
                }
                
                let meta: SafeTensorMetadata = serde_json::from_value(value)
                    .map_err(|e| anyhow!("Invalid tensor metadata for '{}': {}", key, e))?;
                
                let start_offset = meta.data_offsets[0] as usize;
                let end_offset = meta.data_offsets[1] as u64;
                
                tensor_with_info.push((key, start_offset));
                max_end_offset = max_end_offset.max(end_offset);
            }
        }

        // No need to sort - we'll write directly to offsets in parallel
        
        // Total size = header + max tensor end offset
        let total_size = header_data.len() as u64 + max_end_offset;
        
        Ok((tensor_with_info, total_size))
    }

    /// Restore tensor (legacy method for compatibility)
    fn restore_tensor(&self, tensor_hash: &str) -> Result<Vec<u8>> {
        // Check if tensor has compressed metadata
        if self.storage.exists_compressed_metadata(tensor_hash) {
            let compressed_metadata = self.storage.load_compressed_metadata(tensor_hash)
                .map_err(|e| anyhow!("Failed to load compressed metadata: {}", e))?;

            return self.decompress_tensor(&compressed_metadata);
        }

        // If no compressed metadata, load original tensor directly
        self.storage.load_tensor(&tensor_hash.to_string())
            .map_err(|e| anyhow!("Failed to load original tensor: {}", e))
    }

    /// Decompress tensor (legacy method for compatibility)
    fn decompress_tensor(&self, compressed_metadata: &CompressedTensorMetadata) -> Result<Vec<u8>> {

        match &compressed_metadata.compression_type {
            CompressionType::BitX => {
                // Get required hashes
                let base_tensor_hash = compressed_metadata.base_tensor_hash.as_ref()
                    .ok_or_else(|| anyhow!("BitX compressed tensor missing base tensor hash"))?;
                let compressed_hash = &compressed_metadata.compressed_hash;

                // Parallel loading of base tensor and compressed data
                let (base_data, compressed_data): (Result<Vec<u8>>, Result<Vec<u8>>) = rayon::join(
                    || self.storage.load_tensor(base_tensor_hash)
                        .map_err(|e| anyhow!("Failed to load base tensor: {}", e)),
                    || self.storage.load_compressed_tensor(compressed_hash)
                        .map_err(|e| anyhow!("Failed to load compressed tensor: {}", e))
                );
                
                let base_data = base_data?;
                let compressed_data = compressed_data?;
                
                // Parse and decompress
                if compressed_data.len() < 8 {
                    return Err(anyhow!("Compressed data too short for BitX format"));
                }

                let exp_len = u64::from_le_bytes(compressed_data[0..8].try_into()?) as usize;
                if compressed_data.len() < 8 + exp_len {
                    return Err(anyhow!("Compressed data truncated"));
                }

                let compressed_exp = &compressed_data[8..8 + exp_len];
                let compressed_sm = &compressed_data[8 + exp_len..];

                // BitX decompression
                let decompressed_data = bitx_decompress(&base_data, compressed_exp, compressed_sm);

                Ok(decompressed_data)
            },

            CompressionType::ZstdSolo => {
                // Load compressed data
                let compressed_data = self.storage.load_compressed_tensor(&compressed_metadata.compressed_hash)
                    .map_err(|e| anyhow!("Failed to load compressed tensor: {}", e))?;
                
                // Zstd decompression
                let decompressed_data = zstd_decompress_data(&compressed_data);
                
                Ok(decompressed_data)
            },

            CompressionType::Deduplicated { original_hash } => {
                // Load deduplicated tensor
                self.storage.load_tensor(original_hash)
                    .map_err(|e| anyhow!("Failed to load deduplicated tensor: {}", e))
            },

            CompressionType::None => {
                // Load uncompressed tensor
                self.storage.load_tensor(&compressed_metadata.original_hash)
                    .map_err(|e| anyhow!("Failed to load uncompressed tensor: {}", e))
            }
        }
    }

    /// Legacy restore method for individual files (kept for backward compatibility)
    pub fn restore_safetensors_file(&self, file_hash: &str, output_path: &Path, filename: &str) -> Result<()> {
        // Load file metadata
        let file_metadata = self.storage.load_file_metadata(file_hash)
            .map_err(|e| anyhow!("Failed to load file metadata for hash '{}': {}", file_hash, e))?;

        // Load original safetensors header
        let original_header = self.storage.load_safetensors_header(file_hash)
            .map_err(|e| anyhow!("Failed to load safetensors header for file '{}': {}", filename, e))?;

        // Parse header to get tensor metadata and calculate total size
        let (tensor_info, total_size) = self.parse_header_with_size_info(&original_header)?;

        debug!("🔍 Parsed {} tensors, total size: {} bytes", tensor_info.len(), total_size);

        // Step 1: Process all tensors in parallel
        info!("📦 Processing {} tensors in parallel...", tensor_info.len());
        let process_start = Instant::now();
        
        let processed_tensors: Result<Vec<_>> = tensor_info
            .par_iter()
            .map(|(tensor_name, offset)| {
                if let Some(tensor_hash) = file_metadata.tensor_hashes.get(tensor_name) {
                    let tensor_data = self.restore_tensor(tensor_hash)
                        .map_err(|e| anyhow!("Failed to restore tensor '{}': {}", tensor_name, e))?;
                    
                    Ok((tensor_name.clone(), *offset, tensor_data))
                } else {
                    Err(anyhow!("Missing tensor hash for: {}", tensor_name))
                }
            })
            .collect();

        let processed_tensors = processed_tensors?;
        let process_duration = process_start.elapsed();

        let total_tensor_bytes: usize = processed_tensors.iter().map(|(_, _, data)| data.len()).sum();
        info!("✅ Processing: {}", Self::format_throughput(total_tensor_bytes, process_duration));
        
        // Step 2: Create output file and setup mmap
        let output_file_path = output_path.join(filename);
        
        // Remove existing file first
        if output_file_path.exists() {
            fs::remove_file(&output_file_path)
                .map_err(|e| anyhow!("Failed to remove existing file '{}': {}", output_file_path.display(), e))?;
        }
        
        let file = OpenOptions::new()
            .read(true)
            .create(true)
            .write(true)
            .open(&output_file_path)
            .map_err(|e| anyhow!("Failed to create output file: {}", e))?;
        
        file.set_len(total_size)
            .map_err(|e| anyhow!("Failed to set file size: {}", e))?;

        // Memory map the output file
        let mut mmap = unsafe {
            MmapMut::map_mut(&file)
                .map_err(|e| anyhow!("Failed to memory map output file: {}", e))?
        };

        // Copy header first
        let header_size = original_header.len();
        mmap[0..header_size].copy_from_slice(&original_header);

        // Step 3: Parallel mmap writing
        info!("🚀 Writing {} tensors using parallel mmap...", processed_tensors.len());
        let write_start = Instant::now();
        
        // Get mmap pointer for parallel access
        let mmap_ptr_addr = mmap.as_mut_ptr() as usize;
        let mmap_len = mmap.len();
        
        // Parallel mmap write operations
        let write_results: Result<Vec<()>> = processed_tensors.par_iter()
            .map(|(tensor_name, offset, tensor_data)| {
                // Calculate write position in mmap
                let write_offset = header_size + offset;
                let end_offset = write_offset + tensor_data.len();
                
                if end_offset > mmap_len {
                    return Err(anyhow!("Tensor '{}' data exceeds file bounds", tensor_name));
                }
                
                // Direct memory copy using unsafe pointer (zero-copy)
                unsafe {
                    let dst_ptr = (mmap_ptr_addr + write_offset) as *mut u8;
                    std::ptr::copy_nonoverlapping(tensor_data.as_ptr(), dst_ptr, tensor_data.len());
                }
                
                Ok(())
            })
            .collect();

        write_results?;
        let write_duration = write_start.elapsed();

        info!("✅ Writing: {}", Self::format_throughput(total_size as usize, write_duration));

        // Flush to disk
        mmap.flush()
            .map_err(|e| anyhow!("Failed to flush memory map: {}", e))?;

        info!("Reconstructed safetensors file: {total_size} bytes");
        Ok(())
    }

}

