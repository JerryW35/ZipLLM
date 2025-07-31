use crate::traits::*;
use crate::types::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use crate::config::CONFIG;
use dashmap::DashMap;


pub struct FileSystemStorage {
    base_path: PathBuf,
    metadata_cache: DashMap<TensorHash, TensorMetadata>,
    file_metadata_cache: DashMap<String, FileMetadata>, // filename -> FileMetadata
}

impl FileSystemStorage {
    pub fn new() -> Result<Self> {
        let base_path = PathBuf::from(CONFIG.storage_dir.clone());
        
        // Create directories if they don't exist
        fs::create_dir_all(&base_path)?;
        fs::create_dir_all(base_path.join("tensors"))?;
        fs::create_dir_all(base_path.join("tensor_metadata"))?;
        fs::create_dir_all(base_path.join("model_metadata"))?;
        fs::create_dir_all(base_path.join("file_metadata"))?;
        fs::create_dir_all(base_path.join("safetensors_headers"))?;
        fs::create_dir_all(base_path.join("compressed_tensors"))?;
        fs::create_dir_all(base_path.join("compressed_metadata"))?;
        
        Ok(Self {
            base_path,
            metadata_cache: DashMap::new(),
            file_metadata_cache: DashMap::new(),
        })
    }

    fn tensor_path(&self, hash: &TensorHash) -> PathBuf {
        self.base_path.join("tensors").join(hash)
    }

    fn tensor_metadata_path(&self, hash: &TensorHash) -> PathBuf {
        self.base_path.join("tensor_metadata").join(format!("{hash}.json"))
    }

    fn model_metadata_path(&self, model_id: &str) -> PathBuf {
        self.base_path.join("model_metadata").join(format!("{model_id}.json"))
    }

    fn file_metadata_path(&self, file_hash: &str) -> PathBuf {
        self.base_path.join("file_metadata").join(format!("{file_hash}.json"))
    }

    fn safetensors_header_path(&self, file_hash: &str) -> PathBuf {
        self.base_path.join("safetensors_headers").join(file_hash)
    }

    fn compressed_tensor_path(&self, hash: &str) -> PathBuf {
        self.base_path.join("compressed_tensors").join(hash)
    }

    fn compressed_metadata_path(&self, hash: &str) -> PathBuf {
        self.base_path.join("compressed_metadata").join(format!("{hash}.json"))
    }

    pub fn store_compressed_tensor(&self, hash: &str, data: &[u8]) -> Result<()> {
        let path = self.compressed_tensor_path(hash);
        fs::write(path, data)?;
        Ok(())
    }

    pub fn load_compressed_tensor(&self, hash: &str) -> Result<Vec<u8>> {
        let path = self.compressed_tensor_path(hash);
        let data = fs::read(path)?;
        Ok(data)
    }

    pub fn exists_compressed_tensor(&self, hash: &str) -> bool {
        self.compressed_tensor_path(hash).exists()
    }

    pub fn store_compressed_metadata(&self, metadata: &CompressedTensorMetadata) -> Result<()> {
        let path = self.compressed_metadata_path(&metadata.compressed_hash);
        let json = serde_json::to_string_pretty(metadata)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load_compressed_metadata(&self, hash: &str) -> Result<CompressedTensorMetadata> {
        let path = self.compressed_metadata_path(hash);
        let json = fs::read_to_string(path)?;
        let metadata = serde_json::from_str(&json)?;
        Ok(metadata)
    }

    pub fn exists_compressed_metadata(&self, hash: &str) -> bool {
        self.compressed_metadata_path(hash).exists()
    }

    pub fn get_model_tensor_info(&self, model_id: &str) -> Result<Vec<TensorInfo>> {
        let model_metadata: ModelMetadata = self.load_model_metadata(model_id)?;
        let mut tensor_infos = Vec::new();
        
        for file_hash in model_metadata.files.values() {
            if let Ok(file_metadata) = self.load_file_metadata(file_hash) {
                // load file metadata and get tensor_hashes , don't load safetensors header
                let tensor_hashes = file_metadata.tensor_hashes;
                for (tensor_name, tensor_hash) in tensor_hashes {
                    let tensor_metadata = self.load_tensor_metadata(&tensor_hash)?;
                    tensor_infos.push(TensorInfo {
                        name: tensor_name,
                        hash: tensor_hash,
                        shape: tensor_metadata.shape,
                        dtype: tensor_metadata.dtype,
                    });
                }
            }
        }
        Ok(tensor_infos)
    }


    pub fn load_file_metadata(&self, file_hash: &str) -> Result<FileMetadata> {
        if let Some(metadata) = self.file_metadata_cache.iter().find(|item| item.value().file_hash == file_hash) {
            return Ok(metadata.value().clone());
        }
        let path = self.file_metadata_path(file_hash);
        let json = fs::read_to_string(path)?;
        let metadata = serde_json::from_str(&json)?;
        Ok(metadata)
    }

    pub fn exists_file_metadata(&self, file_hash: &str) -> bool {
        self.file_metadata_path(file_hash).exists()
    }

    pub fn store_safetensors_header(&self, file_hash: &str, header_data: &[u8]) -> Result<()> {
        let path = self.safetensors_header_path(file_hash);
        fs::write(path, header_data)?;
        Ok(())
    }

    pub fn load_safetensors_header(&self, file_hash: &str) -> Result<Vec<u8>> {
        let path = self.safetensors_header_path(file_hash);
        let data = fs::read(path)?;
        Ok(data)
    }

    pub fn exists_safetensors_header(&self, file_hash: &str) -> bool {
        self.safetensors_header_path(file_hash).exists()
    }


    pub fn load_all_tensor_metadata(&self) -> Result<HashMap<TensorHash, TensorMetadata>> {
        let mut map = HashMap::new();
        let dir = self.base_path.join("tensor_metadata");
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    let hash = stem.to_string();
                    let json = std::fs::read_to_string(&path)?;
                    let metadata: TensorMetadata = serde_json::from_str(&json)?;
                    map.insert(hash, metadata);
                }
            }
        }
        Ok(map)
    }
}

impl StorageBackend for FileSystemStorage {
    fn store_tensor(&self, hash: &TensorHash, data: &[u8]) -> Result<()> {
        let path = self.tensor_path(hash);
        fs::write(&path, data)?;
        Ok(())
    }

    fn store_tensor_metadata(&self, metadata: &TensorMetadata) -> Result<()> {
        let path = self.tensor_metadata_path(&metadata.hash);
        let json = serde_json::to_string_pretty(metadata)?;
        fs::write(path, json)?;
        self.metadata_cache.insert(metadata.hash.clone(), metadata.clone());
        Ok(())
    }

    fn store_model_metadata(&self, metadata: &ModelMetadata) -> Result<()> {
        let path = self.model_metadata_path(&metadata.model_id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(metadata)?;
        fs::write(path, json)?;
        Ok(())
    }
    fn store_file_metadata(&self, metadata: &FileMetadata) -> Result<()> {
        let path = self.file_metadata_path(&metadata.file_hash);
        let json = serde_json::to_string_pretty(metadata)?;
        fs::write(path, json)?;
        self.file_metadata_cache.insert(metadata.filename.clone(), metadata.clone());
        Ok(())    }

    fn load_tensor(&self, hash: &TensorHash) -> Result<Vec<u8>> {
        let path = self.tensor_path(hash);
        let data = fs::read(path)?;
        Ok(data)
    }

    fn load_tensor_metadata(&self, hash: &TensorHash) -> Result<TensorMetadata> {
        if let Some(metadata) = self.metadata_cache.get(hash) {
            return Ok(metadata.value().clone());
        }
        let path = self.tensor_metadata_path(hash);
        let json = fs::read_to_string(path)?;
        let metadata = serde_json::from_str(&json)?;
        Ok(metadata)
    }

    fn load_model_metadata(&self, model_id: &str) -> Result<ModelMetadata> {
        let path = self.model_metadata_path(model_id);
        let json = fs::read_to_string(path)?;
        let metadata = serde_json::from_str(&json)?;
        Ok(metadata)
    }

    fn exists_tensor(&self, hash: &TensorHash) -> bool {
        self.tensor_path(hash).exists()
    }

    fn exists_tensor_metadata(&self, hash: &TensorHash) -> bool {
        self.tensor_metadata_path(hash).exists()
    }

    fn exists_model(&self, model_id: &str) -> bool {
        self.model_metadata_path(model_id).exists()
    }
    fn exists_file(&self, file_hash: &str) -> bool {
        self.file_metadata_path(file_hash).exists()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deduplication::XxHasher;
    use crate::tensor_loader::read_safetensor;
    use crate::types::TensorMetadata;
    use std::path::Path;
    use log::{info, warn, debug};
    #[test]
    fn test_get_model_tensor_info(){
        let _ = env_logger::builder().is_test(true).try_init();
        let storage = FileSystemStorage::new().expect("Failed to create storage");
        let model_id = "NousResearch_DeepHermes-3-Llama-3-8B-Preview";
        let tensor_infos = storage.get_model_tensor_info(model_id).expect("Failed to get model tensor info");
        info!("tensor_infos: {tensor_infos:?}");
    }
    #[test]
    fn test_store_tensor_from_safetensors() {
        let test_file = "./models/microsoft_DialoGPT-small/model.safetensors";
        if !Path::new(test_file).exists() {
            warn!("Test safetensors file not found: {test_file}");
            return;
        }
        // Read tensors from safetensors file
        let tensors = read_safetensor(test_file).expect("Failed to read safetensors");
        assert!(!tensors.is_empty(), "No tensors found in safetensors file");

        // Prepare storage
        let storage = FileSystemStorage::new().expect("Failed to create storage");
        let hasher = XxHasher;

        for tensor in &tensors {
            let data = tensor.data.as_ref().expect("Tensor data missing");
            let hash = hasher.hash_tensor(tensor);
            debug!("Saving tensor: {} hash: {}", tensor.name, hash);
            //check if tensor already exists
            if storage.exists_tensor_metadata(&hash){
                debug!("Tensor already exists: {hash}");
                continue;
            }

            // Store tensor 
            storage.store_tensor(&hash, data).expect("Failed to store tensor");

            // Store metadata
            let metadata = TensorMetadata {
                name: tensor.name.clone(),
                hash: hash.clone(),
                original_size: data.len() as u64,
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
            };
            storage.store_tensor_metadata(&metadata).expect("Failed to store metadata");

            // Load tensor and metadata back
            let loaded = storage.load_tensor(&hash).expect("Failed to load tensor");
            let loaded_meta = storage.load_tensor_metadata(&hash).expect("Failed to load metadata");
            assert_eq!(loaded_meta.hash, hash);
            assert_eq!(loaded_meta.original_size, data.len() as u64);
        }
    }

}