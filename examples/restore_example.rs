use zipllm::*;
use std::sync::Arc;
use std::env;
use log::{info, error};

fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // Get config path from command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        error!("Config path must be specified as the first argument");
        error!("Usage: {} <config_path>", args[0]);
        return Err("Missing config path argument".into());
    }
    
    // Initialize config from specified path
    let config_path = &args[1];
    config::set_config(config_path);

    info!("ZipLLM Restore Example");

    // Create storage backend
    let storage = Arc::new(FileSystemStorage::new()?);
    
    // Create restore engine
    let restore_engine = RestoreEngine::new(storage.clone());

    // Example model ID (replace with an actual model you have processed)
    let model_id = "meta-llama_Llama-3.1-8B";
    let output_dir = "./restored_example_output";

    // Check if model exists
    if !storage.exists_model(model_id) {
        error!("Model '{model_id}' not found in storage. Please process a model first.");
        return Ok(());
    }

    info!("Restoring model: {model_id}");

    // Restore the model
    match restore_engine.restore_model(model_id, output_dir) {
        Ok(()) => {
            info!("✅ Model restoration completed successfully!");
            info!("Restored safetensors files are available in: {output_dir}");
            
            // List the restored files
            if let Ok(entries) = std::fs::read_dir(output_dir) {
                info!("Restored files:");
                for entry in entries {
                    if let Ok(entry) = entry {
                        let file_name = entry.file_name();
                        if let Some(name) = file_name.to_str() {
                            if name.ends_with(".safetensors") {
                                info!("  📁 {name}");
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            error!("❌ Model restoration failed: {e}");
        }
    }

    Ok(())
} 