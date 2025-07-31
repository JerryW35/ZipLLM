use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
pub type FileHash = String;
pub type TensorHash = String;
pub type ModelId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data: Option<Arc<Vec<u8>>>,
    pub hash: Option<Arc<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: ModelId,
    pub base_model_id: Option<ModelId>,
    pub files: HashMap<String, String>, // filename -> file_hash
    pub is_processed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorMetadata {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub filename: String,
    pub file_hash: FileHash,
    pub size: u64,
    pub is_processed: bool,
    pub tensor_hashes: HashMap<String, TensorHash>, // tensor name -> tensor hash
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub hash: TensorHash,
    pub original_size: u64,
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompressionType {
    None,
    BitX,
    Deduplicated { original_hash: TensorHash },
    ZstdSolo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedTensorMetadata {
    pub name: String,
    pub original_hash: TensorHash,
    pub compressed_hash: TensorHash,
    pub compression_type: CompressionType,
    pub original_size: u64,
    pub compressed_size: u64,
    pub base_tensor_hash: Option<TensorHash>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorPair {
    pub base_tensor: TensorInfo,
    pub finetune_tensor: TensorInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub hash: TensorHash,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTensors {
    pub model_id: ModelId,
    pub tensors: Vec<Tensor>,
}

#[derive(Debug)]
pub struct ProcessingContext {
    pub model_id: ModelId,
    pub base_model_id: Option<ModelId>,
}