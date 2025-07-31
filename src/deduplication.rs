use crate::traits::*;
use crate::types::*;
use xxhash_rust::xxh3::xxh3_64;
use std::collections::HashMap;
use std::fs;
use crate::storage::FileSystemStorage;
use std::sync::Arc;

pub struct XxHasher;

impl XxHasher {
    pub fn hash_bytes(&self, bytes: &[u8]) -> TensorHash {
        format!("{:016x}", xxh3_64(bytes))
    }
}

impl Hasher for XxHasher {
    fn hash_tensor(&self, tensor: &Tensor) -> TensorHash {
        self.hash_bytes(tensor.data.as_ref().expect("Tensor data should exist"))
    }
    fn hash_file(&self, file_path: &str) -> FileHash {
        let data = fs::read(file_path).expect("Failed to read file");
        
        self.hash_bytes(&data)
    }
    

}

pub struct DeduplicationIndex {
    hash_to_metadata: HashMap<TensorHash, TensorMetadata>,
    storage: Arc<FileSystemStorage>,
}

impl DeduplicationIndex {
    pub fn new(storage: Arc<FileSystemStorage>) -> Self {
        let hash_to_metadata = storage.load_all_tensor_metadata().unwrap();
        Self { hash_to_metadata, storage }
    }

    pub fn add_tensor(&mut self, tensor: &Tensor, hash: &TensorHash) -> bool {
        let is_duplicate = self.hash_to_metadata.contains_key(hash);
        
        if !is_duplicate {
            let metadata = TensorMetadata {
                name: tensor.name.clone(),
                hash: hash.clone(),
                original_size: tensor.shape.iter().product::<usize>() as u64 * 4, 
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
            };
            self.hash_to_metadata.insert(hash.clone(), metadata);
        }
        is_duplicate
    }

    pub fn get_metadata(&self, hash: &TensorHash) -> Option<&TensorMetadata> {
        self.hash_to_metadata.get(hash)
    }

    pub fn update_tensor_metadata(&mut self, hash: TensorHash, metadata: TensorMetadata) {
        self.hash_to_metadata.insert(hash, metadata);
    }

    pub fn get_stats(&self) -> DeduplicationStats {
        let unique_tensors = self.hash_to_metadata.len();
        DeduplicationStats {
            total_tensors: unique_tensors,
            unique_tensors,
            duplicate_tensors: 0,
            space_saved: 0,
        }
    }
    pub fn store_index(&mut self) {
        for (hash, metadata) in &self.hash_to_metadata {
            if !self.storage.exists_tensor_metadata(hash) {
                self.storage.store_tensor_metadata(metadata).unwrap();
            }
        }
    }

}


#[derive(Debug, Clone)]
pub struct DeduplicationStats {
    pub total_tensors: usize,
    pub unique_tensors: usize,
    pub duplicate_tensors: usize,
    pub space_saved: u64,
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_loader::read_model_weights;
    use std::time::Instant;
    use log::{info,debug};

    #[test]
    fn test_xxhasher_throughput() {
        let hasher = XxHasher;
        // read tensor 
        let model_name = "meta-llama_Llama-3.1-8B-Instruct";
        let model_tensors = read_model_weights(model_name).unwrap();
        let mut hashes = Vec::new();
        // time 
        let start = Instant::now();
        for tensor in &model_tensors.tensors {
            let hash = hasher.hash_tensor(tensor);
            hashes.push(hash);
        }
        let elapsed = start.elapsed();
        let mb = (model_tensors.tensors.len() * model_tensors.tensors[0].data.as_ref().unwrap().len()) as f64 / (1024.0 * 1024.0);
        info!("Hashed {:.2} MB in {:?} ({:.2} MB/s)", mb, elapsed, mb / elapsed.as_secs_f64());
        debug!("hashes: {hashes:?}");
    }

    #[test]
    fn test_xxhasher_file_hash() {
        let hasher = XxHasher;
        //time it, report throughput
        let file_path = "./models/meta-llama_Llama-3.1-8B-Instruct/model-00001-of-00004.safetensors";
        let file_size = fs::metadata(file_path).unwrap().len();
        // get file size
        let start = Instant::now();
        let hash = hasher.hash_file(file_path);
        info!("hash: {hash:?}");
        let elapsed = start.elapsed();
        let mb = (file_size as f64) / (1024.0 * 1024.0);
        info!("Hashed {:.2} MB in {:?} ({:.2} MB/s)", mb, elapsed, mb / elapsed.as_secs_f64());
    }
    #[test]
    fn test_deduplication_index() {
        let storage = FileSystemStorage::new().unwrap();
        let index = DeduplicationIndex::new(Arc::new(storage));
        // print some index info, not all hash_to_metadata
        info!("index: {:?}", index.hash_to_metadata.keys().take(10).collect::<Vec<_>>());
    }
}