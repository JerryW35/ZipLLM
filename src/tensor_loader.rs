use std::io::{Read, Seek, SeekFrom};
use std::fs::File;
use std::path::Path;
use crate::types::*;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use anyhow::Result;
use thiserror::Error;
use crate::config::CONFIG;
use rayon::prelude::*;
use std::fs::read_dir;
use log::{info, error, debug};

#[derive(Debug, Error)]
pub enum SafeTensorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("File not found: {0}")]
    FileNotFound(String),
}

type Header = HashMap<String, SafeTensorMetadata>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

fn parse_safetensor_header(path: &Path) -> Result<Header, SafeTensorError> {
    let mut file = File::open(path)?;
    let mut header_len_buf = [0u8; 8];
    file.read_exact(&mut header_len_buf)?;
    let header_len = u64::from_le_bytes(header_len_buf) as usize;

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header_str = std::str::from_utf8(&header_buf)
        .map_err(|e| SafeTensorError::Parse(format!("Invalid UTF-8: {e}")))?;
    
    // Parse the raw JSON first
    let raw_json: serde_json::Value = serde_json::from_str(header_str)
        .map_err(|e| SafeTensorError::Parse(format!("Invalid JSON: {e}")))?;
    
    // Filter out the __metadata__ field and convert to our Header type
    let mut header = Header::new();
    if let serde_json::Value::Object(obj) = raw_json {
        for (key, value) in obj {
            // Skip the __metadata__ field
            if key == "__metadata__" {
                continue;
            }
            
            // Parse the tensor metadata
            let metadata: SafeTensorMetadata = serde_json::from_value(value)
                .map_err(|e| SafeTensorError::Parse(format!("Invalid tensor metadata for '{key}': {e}")))?;
            header.insert(key, metadata);
        }
    }
    
    Ok(header)
}

pub fn read_safetensors_info<P: AsRef<Path>>(path: P) -> Result<Vec<SafeTensorInfo>, SafeTensorError> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(SafeTensorError::FileNotFound(path.display().to_string()));
    }

    let header = parse_safetensor_header(path)?;
    info!("header: {header:?}");
    let infos = header
        .into_iter()
        .map(|(name, meta)| SafeTensorInfo {
            name,
            shape: meta.shape,
            dtype: meta.dtype,
        })
        .collect();

    Ok(infos)
}

pub fn print_safetensors_info<P: AsRef<Path>>(path: P) -> Result<(), SafeTensorError> {
    let path = path.as_ref();
    let tensor_infos = read_safetensors_info(path)?;

    info!("SafeTensor file: {}", path.display());
    info!("Total tensors: {}", tensor_infos.len());

    for info in tensor_infos {
        debug!("Tensor name: {}", info.name);
        debug!("Shape: {:?}", info.shape);
        debug!("Data type: {}", info.dtype);
    }

    Ok(())
}

pub fn read_tensor_bytes<P: AsRef<Path>>(path: P, tensor_name: &str) -> Result<Vec<u8>, SafeTensorError> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(SafeTensorError::FileNotFound(path.display().to_string()));
    }

    let header = parse_safetensor_header(path)?;
    let meta = header.get(tensor_name)
        .ok_or_else(|| SafeTensorError::FileNotFound(tensor_name.to_string()))?;
    let start = meta.data_offsets[0] as u64;
    let end = meta.data_offsets[1] as u64;
    let len = (end - start) as usize;

    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start))?;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)?;
    Ok(buf)
}

pub fn read_safetensor<P: AsRef<Path>>(path: P) -> Result<Vec<Tensor>, SafeTensorError> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(SafeTensorError::FileNotFound(path.display().to_string()));
    }

    let header = parse_safetensor_header(path)?;
    let path_buf = path.to_path_buf();

    let tensors: Vec<Tensor> = header
        .into_par_iter()
        .map(|(name, meta)| {
            let start = meta.data_offsets[0] as u64;
            let end = meta.data_offsets[1] as u64;
            let len = (end - start) as usize;

            let mut file = File::open(&path_buf).unwrap();
            file.seek(SeekFrom::Start(start)).unwrap();
            let mut buf = vec![0u8; len];
            file.read_exact(&mut buf).unwrap();

            Tensor {
                name,
                shape: meta.shape,
                dtype: meta.dtype,
                data: Some(Arc::new(buf)),
                hash: None, 
            }
        })
        .collect();

    Ok(tensors)
}

pub fn read_model_weights(model_name: &str) -> Result<ModelTensors, SafeTensorError> {
    let model_dir = format!("{}/{}", CONFIG.model_dir, model_name);
    let mut tensors: Vec<Tensor> = Vec::new();
    let dir = read_dir(&model_dir).map_err(|_| SafeTensorError::FileNotFound(model_dir.clone()))?;
    for entry in dir {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "safetensors" {
                let mut ts = read_safetensor(&path)?;
                tensors.append(&mut ts);
            }
        }
        
    }
    Ok(ModelTensors {
        model_id: model_name.to_string(),
        tensors,
    })
}

pub fn get_safetensors_header<P: AsRef<Path>>(path: P) -> Result<serde_json::Value, SafeTensorError> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(SafeTensorError::FileNotFound(path.display().to_string()));
    }

    let mut file = File::open(path)?;
    let mut header_len_buf = [0u8; 8];
    file.read_exact(&mut header_len_buf)?;
    let header_len = u64::from_le_bytes(header_len_buf) as usize;

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let header_str = std::str::from_utf8(&header_buf)
        .map_err(|e| SafeTensorError::Parse(format!("Invalid UTF-8: {e}")))?;
    
    // Parse and return the raw JSON
    let raw_json: serde_json::Value = serde_json::from_str(header_str)
        .map_err(|e| SafeTensorError::Parse(format!("Invalid JSON: {e}")))?;
    
    Ok(raw_json)
}
pub fn extract_safetensors_header(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 8 {
        return None;
    }

    let header_len = u64::from_le_bytes(data[0..8].try_into().ok()?);
    let end = 8 + (header_len as usize);

    if data.len() < end {
        return None;
    }

    Some(data[0..end].to_vec())
}

pub fn parse_safetensors_from_bytes(data: &[u8]) -> Result<Vec<Tensor>, SafeTensorError> {
    if data.len() < 8 {
        return Err(SafeTensorError::Parse("Data too short to contain header".to_string()));
    }

    // Read header length
    let header_len = u64::from_le_bytes(
        data[0..8].try_into()
            .map_err(|e| SafeTensorError::Parse(format!("Failed to read header length: {e}")))?
    ) as usize;

    let header_end = 8 + header_len;
    if data.len() < header_end {
        return Err(SafeTensorError::Parse("Data too short to contain complete header".to_string()));
    }

    // Parse header JSON
    let header_str = std::str::from_utf8(&data[8..header_end])
        .map_err(|e| SafeTensorError::Parse(format!("Invalid UTF-8 in header: {e}")))?;
    
    let raw_json: serde_json::Value = serde_json::from_str(header_str)
        .map_err(|e| SafeTensorError::Parse(format!("Invalid JSON in header: {e}")))?;
    
    // Parse tensor metadata
    let mut header = Header::new();
    if let serde_json::Value::Object(obj) = raw_json {
        for (key, value) in obj {
            // Skip the __metadata__ field
            if key == "__metadata__" {
                continue;
            }
            
            let metadata: SafeTensorMetadata = serde_json::from_value(value)
                .map_err(|e| SafeTensorError::Parse(format!("Invalid tensor metadata for '{key}': {e}")))?;
            header.insert(key, metadata);
        }
    }

    // Extract tensors in parallel
    let tensors: Result<Vec<Tensor>, SafeTensorError> = header
        .into_par_iter()
        .map(|(name, meta)| {
            let start = header_end + meta.data_offsets[0];
            let end = header_end + meta.data_offsets[1];
            
            if end > data.len() {
                return Err(SafeTensorError::Parse(
                    format!("Tensor '{name}' data extends beyond file boundary")
                ));
            }
            
            let tensor_data = data[start..end].to_vec();
            
            Ok(Tensor {
                name,
                shape: meta.shape,
                dtype: meta.dtype,
                data: Some(Arc::new(tensor_data)),
                hash: None,
            })
        })
        .collect();

    tensors
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_read_real_safetensors_file() {
        let test_file = "/home/ubuntu/zipllm_rust/models/microsoft_DialoGPT-small/model.safetensors";
        if std::path::Path::new(test_file).exists() {
            let result = read_safetensors_info(test_file);
            match result {
                Ok(tensor_infos) => {
                    assert!(!tensor_infos.is_empty());

                    let first_tensor = &tensor_infos[0];
                    debug!("first tensor info: {first_tensor:?}");
                    assert!(!first_tensor.name.is_empty());
                    assert!(!first_tensor.shape.is_empty());
                    assert!(!first_tensor.dtype.is_empty());
                }
                Err(e) => {
                    panic!("Failed to read safetensors file: {e:?}");
                }
            }
        }
    }

    #[test]
    fn test_tensor_raw_bytes() {
        let test_file = "./models/microsoft_DialoGPT-small/model.safetensors";
        let tensor_name = "transformer.h.3.attn.c_attn.weight";

        if std::path::Path::new(test_file).exists() {
            let result = read_tensor_bytes(test_file, tensor_name);
            assert!(result.is_ok());

            let bytes = result.unwrap();
            debug!("Tensor {} raw bytes len: {}", tensor_name, bytes.len());
            assert!(!bytes.is_empty());
        }
    }

    #[test]
    fn test_read_model_weights() {
        let model_name = "meta-llama_Llama-3.1-8B-Instruct";
        let start = Instant::now();
        let model_tensors = read_model_weights(model_name).unwrap();
        let elapsed = start.elapsed();
        debug!("model_tensors: {:?}", model_tensors.tensors.len());
        debug!("time: {elapsed:?}");
    }
}
