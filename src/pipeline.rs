use crate::traits::*;
use crate::types::*;
use crate::deduplication::*;
use crate::storage::*;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use anyhow::Result;
use crate::config::CONFIG;
use crate::tensor_loader::*;
use std::collections::HashMap;
use std::fs;
use std::fs::read_dir;
use std::fs::read;
use crate::tensor_loader::{extract_safetensors_header, parse_safetensors_from_bytes};
use serde_json;
use log::{info, error,debug};

pub struct ZipLLMPipeline {
    storage: Arc<FileSystemStorage>,
    hasher: Arc<dyn Hasher>,
    dedup_index: Arc<Mutex<DeduplicationIndex>>,
}

impl ZipLLMPipeline {
    pub fn new(
        storage: Arc<FileSystemStorage>,
        hasher: Arc<dyn Hasher>,
    ) -> Result<Self> {
        let dedup_index = Arc::new(Mutex::new(DeduplicationIndex::new(Arc::clone(&storage))));
    
        Ok(Self {
            storage,
            hasher,
            dedup_index,
        })
    }


    pub fn process_models_sequential(&self, models: &[String]){
        
        info!("Processing {} models sequentially", models.len());
        
        for (i, model_id) in models.iter().enumerate() {
            info!("Processing model {}/{}: {}", i + 1, models.len(), model_id);
            
            let mut context = ProcessingContext {
                model_id: model_id.clone(),
                base_model_id: self.find_base_model_id(model_id),
            };
            
            self.process_single_model(&mut context);
            
            info!("Completed model {}/{}: {}", i + 1, models.len(), model_id);
        }
        
        
    }

    pub fn process_single_model(&self, context: &mut ProcessingContext){
        
        info!("[{}] Starting processing", context.model_id);
        
        // Check if model is already fully processed
        if let Ok(existing_metadata) = self.storage.load_model_metadata(&context.model_id) {
            if existing_metadata.is_processed {
                info!("[{}] Model already fully processed, skipping", context.model_id);
                return;
            } else {
                info!("[{}] Model metadata exists but not fully processed, continuing", context.model_id);
            }
        } else {
            info!("[{}] No existing model metadata found, starting fresh processing", context.model_id);
        }
        
        // Stage 1: File Deduplication
        let (tensors, processed_files, tensor_to_file) = self.file_dedup_stage(context).unwrap();
        
        // Stage 2: Tensor Deduplication
        let deduplicated_tensors = self.deduplication_stage(tensors, context, &processed_files, &tensor_to_file).unwrap();
        
        // Stage 3: Storage
        self.storage_stage(&deduplicated_tensors, context);

        // Stage 4: Compression 
        self.compression_stage(&deduplicated_tensors, context, context.base_model_id.as_deref());
        
        // Stage 5: Mark files as fully processed
        self.mark_files_processed(processed_files);
        
        // Stage 6: Mark model as fully processed
        self.mark_model_processed(&context.model_id);
        
        info!("[{}] Processing completed", context.model_id);
    }

    fn mark_model_processed(&self, model_id: &str) {
        match self.storage.load_model_metadata(model_id) {
            Ok(mut metadata) => {
                if !metadata.is_processed {
                    metadata.is_processed = true;
                    match self.storage.store_model_metadata(&metadata) {
                        Ok(_) => {
                            info!("[{model_id}] Model marked as fully processed");
                        }
                        Err(e) => {
                            error!("[{model_id}] Failed to mark model as processed: {e}");
                        }
                    }
                } else {
                    info!("[{model_id}] Model already marked as processed");
                }
            }
            Err(e) => {
                error!("[{model_id}] Failed to load model metadata to mark as processed: {e}");
            }
        }
    }

    pub fn find_base_model_id(&self, model_id: &str) -> Option<String> {
        // Load base_ft.json file using config
        let base_ft_path = &CONFIG.base_ft_path;
        let query_model_id = model_id.replacen("_", "/", 1);
        match fs::read_to_string(&base_ft_path) {
            Ok(json_content) => {
                match serde_json::from_str::<HashMap<String, Vec<String>>>(&json_content) {
                    Ok(base_ft_map) => {
                        // Check if current model_id is a base model (key in the map)
                        if base_ft_map.contains_key(&query_model_id.to_string()) {
                            info!("[{model_id}] This is a base model");
                            return None;
                        }
                        
                        // Check if current model_id is in any finetune list
                        for (base_model, finetune_models) in &base_ft_map {
                            if finetune_models.contains(&query_model_id.to_string()) {
                                let base_model_underscore = base_model.replace("/", "_");
                                info!("[{model_id}] Found base model: {base_model_underscore}");
                                return Some(base_model_underscore);
                            }
                        }
                        
                        info!("[{model_id}] No base model found, treating as standalone model");
                        None
                    }
                    Err(e) => {
                        error!("[{model_id}] Failed to parse base_ft.json: {e}");
                        None
                    }
                }
            }
            Err(e) => {
                error!("[{model_id}] Failed to read base_ft.json from {base_ft_path}: {e}");
                None
            }
        }
    }

    fn file_dedup_stage(&self, context: &ProcessingContext) -> Result<(Vec<Tensor>, Vec<FileHash>, HashMap<String, FileHash>)> {
    
        info!("[{}] File deduplication stage", context.model_id);
        let model_dir = format!("{}/{}", CONFIG.model_dir, context.model_id);
        let mut tensors: Vec<Tensor> = Vec::new();
        let mut all_files: Vec<FileHash> = Vec::new();
        let mut tensor_to_file: HashMap<String, FileHash> = HashMap::new(); // tensor name -> file hash
        let mut filename_to_hash: HashMap<String, String> = HashMap::new(); // filename -> file hash

        let dir = read_dir(&model_dir).map_err(|_| SafeTensorError::FileNotFound(model_dir.clone()))?;
        for entry in dir {
            let entry = entry?;
            let path = entry.path();

            if let Some(ext) = path.extension() {
                if ext == "safetensors" {
                    let data: Vec<u8> = read(&path).map_err(SafeTensorError::Io)?;
                    let file_name = path.file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let file_hash = XxHasher.hash_bytes(&data);
                    all_files.push(file_hash.clone());
                    filename_to_hash.insert(file_name.clone(), file_hash.clone());
                    
                    let should_parse_tensors = if self.storage.exists_file(&file_hash.clone()) {
                        // File exists, check if tensors have been processed
                        match self.storage.load_file_metadata(&file_hash) {
                            Ok(existing_metadata) => {
                                if existing_metadata.is_processed {
                                    info!("[{}] File already exists and tensors processed: {}", context.model_id, file_name);
                                    false
                                } else {
                                    info!("[{}] File exists but tensors not processed: {}", context.model_id, file_name);
                                    true
                                }
                            }
                            Err(_) => {
                                info!("[{}] Failed to load file metadata, will reprocess: {}", context.model_id, file_name);
                                true
                            }
                        }
                    } else {
                        info!("[{}] New file: {}", context.model_id, file_name);
                        true
                    };
                    
                    if should_parse_tensors {
                        // Parse tensors from the file and store header separately
                        if let Some(header) = extract_safetensors_header(&data) {
                            if let Err(e) = self.storage.store_safetensors_header(&file_hash, &header) {
                                error!("[{}] Failed to store safetensors header for {}: {}", 
                                    context.model_id, file_name, e);
                            }
                        }
                        
                        let file_metadata = if self.storage.exists_file(&file_hash) {
                            match self.storage.load_file_metadata(&file_hash) {
                                Ok(existing_metadata) => {
                                    info!("[{}] Loading existing file metadata for: {}", context.model_id, file_name);
                                    existing_metadata
                                }
                                Err(_) => {
                                    info!("[{}] Failed to load existing metadata, creating new for: {}", context.model_id, file_name);
                                    FileMetadata {
                                        filename: file_name.clone(),
                                        file_hash: file_hash.clone(),
                                        size: data.len() as u64,
                                        is_processed: false,
                                        tensor_hashes: HashMap::new(),
                                    }
                                }
                            }
                        } else {
                            info!("[{}] Creating new file metadata for: {}", context.model_id, file_name);
                            FileMetadata {
                                filename: file_name.clone(),
                                file_hash: file_hash.clone(),
                                size: data.len() as u64,
                                is_processed: false,
                                tensor_hashes: HashMap::new(),
                            }
                        };
                        
                        match parse_safetensors_from_bytes(&data) {
                            Ok(mut file_tensors) => {
                                for tensor in &file_tensors {
                                    tensor_to_file.insert(tensor.name.clone(), file_hash.clone());
                                }
                                tensors.append(&mut file_tensors);
                            }
                            Err(e) => {
                                error!("[{}] Failed to parse tensors from {}: {}", 
                                    context.model_id, file_name, e);
                            }
                        }
                        
                        self.storage.store_file_metadata(&file_metadata).unwrap();
                    }
                }
            }
        }
        
        // Create and store ModelMetadata after processing all files
        let model_metadata = ModelMetadata {
            model_id: context.model_id.clone(),
            base_model_id: context.base_model_id.clone(),
            files: filename_to_hash,
            is_processed: false, 
        };
        
        match self.storage.store_model_metadata(&model_metadata) {
            Ok(_) => {
                info!("[{}] Model metadata stored successfully", context.model_id);
            }
            Err(e) => {
                error!("[{}] Failed to store model metadata: {}", context.model_id, e);
            }
        }
        
        Ok((tensors, all_files, tensor_to_file))
    }
    

    fn deduplication_stage(&self, mut tensors: Vec<Tensor>, context: &ProcessingContext, file_hashes: &[FileHash], tensor_to_file: &HashMap<String, FileHash>) -> Result<Vec<(Tensor, bool)>> {
        info!(
            "[{}] Deduplicating {} tensors",
            context.model_id,
            tensors.len()
        );
        
        // Step 1: compute hashes
        let tensor_hashes: Vec<(usize, TensorHash)> = tensors
            .par_iter()
            .enumerate()
            .map(|(i, tensor)| (i, self.hasher.hash_tensor(tensor)))
            .collect();
    
        for (i, hash) in &tensor_hashes {
            tensors[*i].hash = Some(Arc::new(hash.clone()));
        }

        // Step 2: Collect tensor name to hash mapping for each file
        let mut file_tensor_maps: HashMap<FileHash, HashMap<String, TensorHash>> = HashMap::new();
      
        for file_hash in file_hashes {
            file_tensor_maps.insert(file_hash.clone(), HashMap::new());
        }
        
        for (i, hash) in &tensor_hashes {
            let tensor_name = &tensors[*i].name;
            if let Some(file_hash) = tensor_to_file.get(tensor_name) {
                if let Some(tensor_map) = file_tensor_maps.get_mut(file_hash) {
                    tensor_map.insert(tensor_name.clone(), hash.clone());
                }
            }
        }

        // Step 3: update dedup index
        let mut results = Vec::with_capacity(tensors.len());
        let mut dedup_count = 0;
    
        {
            let mut dedup_index = self.dedup_index.lock().unwrap();
    
            for (i, hash) in tensor_hashes {
                let tensor = tensors[i].clone();
                let is_duplicate = dedup_index.add_tensor(&tensor, &hash);
    
                if is_duplicate {
                    dedup_count += 1;
                    debug!(
                        "[{}] Duplicate tensor found: {} (hash: {})",
                        context.model_id,
                        tensor.name,
                        hash
                    );
                }
    
                results.push((tensor, is_duplicate));
            }
        }
        
        // Step 4: Update file metadata with tensor hashes
        for file_hash in file_hashes {
            if let Ok(mut file_metadata) = self.storage.load_file_metadata(file_hash) {
                if let Some(tensor_map) = file_tensor_maps.get(file_hash) {
                    for (tensor_name, tensor_hash) in tensor_map {
                        file_metadata.tensor_hashes.insert(tensor_name.clone(), tensor_hash.clone());
                    }
                }
                
                if let Err(e) = self.storage.store_file_metadata(&file_metadata) {
                    error!("[{}] Failed to update file metadata with tensor hashes for {}: {}", 
                        context.model_id, file_hash, e);
                } else {
                    debug!("[{}] Updated file metadata with {} tensor hashes for file: {}", 
                        context.model_id, file_metadata.tensor_hashes.len(), file_hash);
                }
            }
        }
    
        info!(
            "[{}] Deduplication completed: {} duplicates found out of {} tensors",
            context.model_id,
            dedup_count,
            results.len()
        );
    
        Ok(results)
    }
    
    
    
    fn storage_stage(
        &self,
        tensors: &Vec<(Tensor, bool)>,
        context: &ProcessingContext,
    ) {
        info!("[{}] Storing {} tensors", context.model_id, tensors.len());
        
        tensors.into_par_iter().for_each(|(tensor, is_duplicate)| {
            let original_size = tensor.data.as_ref().unwrap().len() as u64;
    
            if *is_duplicate {
                debug!("[{}] Skipping duplicate tensor: {}", context.model_id, tensor.name);
                return;
            }
    
            let data = tensor.data.as_ref().unwrap().clone();
            let hash = tensor.hash.as_ref().unwrap().clone();
    
            self.storage.store_tensor(&hash, &data).unwrap();
    
            let metadata = TensorMetadata {
                name: tensor.name.clone(),
                hash: (*hash).clone(),
                original_size,
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
            };
    
            self.storage.store_tensor_metadata(&metadata).unwrap();
    
            debug!(
                "[{}] Stored tensor: {} (hash: {})",
                context.model_id, tensor.name, hash
            );
        });
        self.dedup_index.lock().unwrap().store_index();
    }
    
    fn compression_stage(
        &self,
        _tensors: &Vec<(Tensor, bool)>,
        context: &ProcessingContext,
        base_model_id_opt: Option<&str>,
    ) {
        // Create compression engine
        let compression_engine = crate::compression::CompressionEngine::new(Arc::clone(&self.storage));
        
        match base_model_id_opt {
            Some(base_model_id) => {
                // Finetune model compression: compare with base model
                info!("[{}] Starting finetune compression with base model: {}", context.model_id, base_model_id);
                
                match compression_engine.compress_models(base_model_id, &context.model_id) {
                    Ok(result) => {
                        info!("[{}] Finetune compression completed successfully", context.model_id);
                        info!("  - Paired tensors: {}", result.compression_stats.paired_tensors);
                        info!("  - Solo tensors: {}", result.compression_stats.solo_tensors);
                        info!("  - Original size: {} bytes", result.compression_stats.original_size);
                        info!("  - Compressed size: {} bytes", result.compression_stats.compressed_size);
                        info!("  - Compression ratio: {:.2}%", result.compression_stats.compression_ratio * 100.0);
                    }
                    Err(e) => {
                        error!("[{}] Finetune compression failed: {}", context.model_id, e);
                    }
                }
            }
            None => {
                // Base model compression: compress all tensors with zstd
                info!("[{}] Starting base model compression", context.model_id);
                
                match compression_engine.compress_base_model(&context.model_id) {
                    Ok(result) => {
                        info!("[{}] Base model compression completed successfully", context.model_id);
                        info!("  - Total tensors: {}", result.compression_stats.total_tensors);
                        info!("  - Original size: {} bytes", result.compression_stats.original_size);
                        info!("  - Compressed size: {} bytes", result.compression_stats.compressed_size);
                        info!("  - Compression ratio: {:.2}%", result.compression_stats.compression_ratio * 100.0);
                    }
                    Err(e) => {
                        error!("[{}] Base model compression failed: {}", context.model_id, e);
                    }
                }
            }
        }
        
        info!("[{}] Compression stage completed", context.model_id);
    }
    
    fn mark_files_processed(&self, file_hashes: Vec<FileHash>) {
        info!("Marking files as processed: {file_hashes:?}");
        for file_hash in file_hashes {
            if let Ok(mut file_metadata) = self.storage.load_file_metadata(&file_hash) {
                if !file_metadata.is_processed {
                    file_metadata.is_processed = true;
                    if let Err(e) = self.storage.store_file_metadata(&file_metadata) {
                        error!("Failed to update file metadata for {file_hash}: {e}");
                    } else {
                        info!("Marked file as processed: {file_hash}");
                    }
                } else {
                    info!("File already marked as processed: {file_hash}");
                }
            } else {
                error!("Failed to load file metadata for {file_hash}");
            }
        }
    }

    pub fn get_dedup_stats(&self) -> DeduplicationStats {
        let dedup_index = self.dedup_index.lock().unwrap();
        dedup_index.get_stats()
    }
}

#[derive(Debug, Clone)]
pub struct ProcessingStats {
    pub models_processed: usize,
    pub tensors_processed: usize,
    pub total_size_processed: u64,
    pub total_size_saved: u64,
}

impl Default for ProcessingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessingStats {
    pub fn new() -> Self {
        Self {
            models_processed: 0,
            tensors_processed: 0,
            total_size_processed: 0,
            total_size_saved: 0,
        }
    }

    pub fn merge(&mut self, other: ProcessingStats) {
        self.models_processed += other.models_processed;
        self.tensors_processed += other.tensors_processed;
        self.total_size_processed += other.total_size_processed;
        self.total_size_saved += other.total_size_saved;
    }
}

#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_original_size: u64,
    pub total_size_saved: u64,
}

impl Default for StorageStats {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageStats {
    pub fn new() -> Self {
        Self {
            total_original_size: 0,
            total_size_saved: 0,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline() {
        
        let pipeline = ZipLLMPipeline::new(Arc::new(FileSystemStorage::new().unwrap()), Arc::new(XxHasher)).unwrap();
        pipeline.process_models_sequential(&["meta-llama_Llama-3.1-8B-Instruct".to_string()]);

    }
    #[test]
    fn test_file_dedup_stage(){
        let pipeline = ZipLLMPipeline::new(Arc::new(FileSystemStorage::new().unwrap()), Arc::new(XxHasher)).unwrap();
        let context = ProcessingContext {
            model_id: "meta-llama_Llama-3.1-8B-Instruct".to_string(),
            base_model_id: None,
        };
        let (tensors, files, tensor_to_file) = pipeline.file_dedup_stage(&context).unwrap();
        info!("Tensors: {:?}", tensors.len());
        info!("Files: {:?}", files.len());
        info!("Tensor to file mapping: {:?}", tensor_to_file.len());
        
        // Verify that ModelMetadata was created and stored
        match pipeline.storage.load_model_metadata(&context.model_id) {
            Ok(metadata) => {
                info!("ModelMetadata created successfully:");
                info!("  Model ID: {}", metadata.model_id);
                info!("  Base Model ID: {:?}", metadata.base_model_id);
                info!("  Files count: {}", metadata.files.len());
                for (filename, file_hash) in &metadata.files {
                    info!("    {filename} -> {file_hash}");
                }
            }
            Err(e) => {
                error!("Failed to load ModelMetadata: {e}");
            }
        }
    }
    #[test]
    fn test_find_base_model_id(){
        let pipeline = ZipLLMPipeline::new(Arc::new(FileSystemStorage::new().unwrap()), Arc::new(XxHasher)).unwrap();
        let base_model_id = pipeline.find_base_model_id("meta-llama/Llama-3.1-8B-Instruct");
        info!("Base model id: {base_model_id:?}");
    }
    #[test]
    fn test_all_models(){
        let pipeline = ZipLLMPipeline::new(Arc::new(FileSystemStorage::new().unwrap()), Arc::new(XxHasher)).unwrap();
        
        // Read models from the txt file
        let models_file_path = "/home/ubuntu/zipllm_rust/models/meta-llama_Llama-3.1-8B.txt";
        let models = match fs::read_to_string(models_file_path) {
            Ok(content) => {
                content
                    .lines()
                    .map(|line| {
                        let trimmed = line.trim().to_string();
                        if let Some(pos) = trimmed.find('/') {
                            format!("{}{}{}", &trimmed[..pos], "_", &trimmed[pos+1..])
                        } else {
                            trimmed
                        }
                    })
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<String>>()
            }
            Err(e) => {
                error!("Failed to read models file {models_file_path}: {e}");
                return;
            }
        };
        
        info!("Found {} models to process: {:?}", models.len(), models);
        pipeline.process_models_sequential(&models);
    }
    #[test]
    fn test_signle_model(){
        let _ = env_logger::builder().is_test(true).try_init();
        let pipeline = ZipLLMPipeline::new(Arc::new(FileSystemStorage::new().unwrap()), Arc::new(XxHasher)).unwrap();
        pipeline.process_single_model(&mut ProcessingContext {
            model_id: "NousResearch_DeepHermes-3-Llama-3-8B-Preview".to_string(),
            base_model_id: pipeline.find_base_model_id("NousResearch_DeepHermes-3-Llama-3-8B-Preview"),
        });

    }
    
}