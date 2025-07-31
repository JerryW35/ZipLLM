use crate::types::*;
use std::error::Error;

pub type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub trait PipelineStage {
    fn name(&self) -> &str;
    fn process(&mut self, context: &ProcessingContext) -> Result<()>;
}

pub trait StorageBackend {
    fn store_tensor(&self, hash: &TensorHash, data: &[u8]) -> Result<()>;
    fn store_tensor_metadata(&self, metadata: &TensorMetadata) -> Result<()>;
    fn store_model_metadata(&self, metadata: &ModelMetadata) -> Result<()>;
    fn store_file_metadata(&self, metadata: &FileMetadata) -> Result<()>;
    fn load_tensor(&self, hash: &TensorHash) -> Result<Vec<u8>>;
    fn load_tensor_metadata(&self, hash: &TensorHash) -> Result<TensorMetadata>;
    fn load_model_metadata(&self, model_id: &str) -> Result<ModelMetadata>;
    fn exists_tensor(&self, hash: &TensorHash) -> bool;
    fn exists_tensor_metadata(&self, hash: &TensorHash) -> bool;
    fn exists_model(&self, model_id: &str) -> bool;
    fn exists_file(&self, file_hash: &str) -> bool;
}

pub trait Compressor {
    fn compress(&self, tensor: &Tensor, base_tensor: Option<&Tensor>) -> Result<Vec<u8>>;
    fn decompress(&self, data: &[u8], base_tensor: Option<&Tensor>) -> Result<Tensor>;
}

pub trait Hasher: Send + Sync {
    fn hash_tensor(&self, tensor: &Tensor) -> TensorHash;
    fn hash_file(&self, file_path: &str) -> TensorHash;
}

pub trait ModelLoader: Send + Sync {
    fn load_model(&self, model_id: &str, temp_dir: &str) -> Result<Vec<Tensor>>;
    fn save_model(&self, tensors: &[Tensor], output_path: &str) -> Result<()>;
}