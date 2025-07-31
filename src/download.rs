use std::process::Command;
use std::path::Path;
use std::io;
use crate::config::CONFIG;
use crate::traits::*;
use crate::tensor_loader::*;
use crate::types::*;

use log::{info, error, debug};

pub fn download_model(model_name: &str) -> io::Result<()> {
    download_model_with_options(model_name, &CONFIG.model_dir, 8, false)
}

pub fn download_model_with_options(
    model_name: &str, 
    output_dir: &str, 
    max_workers: u32, 
    check_already_downloaded: bool
) -> io::Result<()> {
    let script_path = Path::new("py_lib/download.py");

    if !script_path.exists() {
        error!("Python download script not found: {script_path:?}");
        return Err(io::Error::new(io::ErrorKind::NotFound, "Python script not found"));
    }

    let mut cmd = Command::new("python3");
    cmd.arg(script_path)
        .arg("--model")
        .arg(model_name)
        .arg("--output_dir")
        .arg(output_dir)
        .arg("--max_workers")
        .arg(max_workers.to_string());

    if check_already_downloaded {
        cmd.arg("--check_already_downloaded");
    }

    let output = cmd.output()?;

    if output.status.success() {
        info!("Model '{model_name}' downloaded successfully to '{output_dir}'.");
        Ok(())
    } else {
        error!("Download failed: {}", String::from_utf8_lossy(&output.stderr));
        Err(io::Error::other("Python download failed"))
    }
}

pub fn download_models_from_file(
    models_txt: &str,
    output_dir: &str,
    max_workers: u32,
    check_already_downloaded: bool
) -> io::Result<()> {
    let script_path = Path::new("py_lib/download.py");

    if !script_path.exists() {
        error!("Python download script not found: {script_path:?}");
        return Err(io::Error::new(io::ErrorKind::NotFound, "Python script not found"));
    }

    let mut cmd = Command::new("python3");
    cmd.arg(script_path)
        .arg("--models_txt")
        .arg(models_txt)
        .arg("--output_dir")
        .arg(output_dir)
        .arg("--max_workers")
        .arg(max_workers.to_string());

    if check_already_downloaded {
        cmd.arg("--check_already_downloaded");
    }

    let output = cmd.output()?;

    if output.status.success() {
        info!("Models from '{models_txt}' downloaded successfully to '{output_dir}'.");
        Ok(())
    } else {
        error!("Download failed: {}", String::from_utf8_lossy(&output.stderr));
        Err(io::Error::other("Python download failed"))
    }
}

pub fn download_models_from_file_simple(models_txt: &str, output_dir: &str) -> io::Result<()> {
    download_models_from_file(models_txt, output_dir, 8, false)
}

pub struct SafeTensorModelLoader;

impl Default for SafeTensorModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl SafeTensorModelLoader {
    pub fn new() -> Self {
        Self
    }
}

impl ModelLoader for SafeTensorModelLoader {
    fn load_model(&self, model_id: &str, _temp_dir: &str) -> Result<Vec<Tensor>> {
        debug!("Loading model tensors from: {model_id}");
        let model_tensors = read_model_weights(model_id)
            .map_err(|e| format!("Failed to read model: {e}"))?;
        Ok(model_tensors.tensors)
    }

    fn save_model(&self, tensors: &[Tensor], output_path: &str) -> Result<()> {
        debug!("Saving {} tensors to: {}", tensors.len(), output_path);
        // Placeholder: implement safetensor file writing
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_single_model() {
        let model_name = "mistralai/Mistral-7B-Instruct-v0.1";
        let result = download_model(model_name);
        assert!(result.is_ok());
    }

    #[test]
    fn test_download_with_options() {
        let model_name = "mistralai/Mistral-7B-Instruct-v0.1";
        let output_dir = "./models";
        let result = download_model_with_options(model_name, output_dir, 4, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_download_from_file() {
        // You can prepare a model list file in tests/resources/models.txt
        let models_txt = "models/models.txt";
        let output_dir = "./models";
        let result = download_models_from_file(models_txt, output_dir, 8, false);
        assert!(result.is_ok());
    }
}
