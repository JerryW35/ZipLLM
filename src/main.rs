use zipllm::*;
use std::sync::Arc;
use std::fs;
use std::env;
use log::{info, warn, error, debug};

fn main() -> Result<()> {
    // Initialize logger with config level
    env_logger::Builder::from_default_env()
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
    
    info!("Starting ZipLLM - Processing all models...");
    
    // Create pipeline
    let storage = Arc::new(storage::FileSystemStorage::new()?);
    let hasher = Arc::new(deduplication::XxHasher);
    let pipeline = pipeline::ZipLLMPipeline::new(storage.clone(), hasher)?;
    
    // Read models from the txt file
    let models_file_path = CONFIG.get_models_to_process();
    info!("Reading models from: {}", models_file_path.display());
    
    let models = match fs::read_to_string(&models_file_path) {
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
            error!("Failed to read models file {}: {e}", models_file_path.display());
            return Err(e.into());
        }
    };
    
    info!("Found {} models to process", models.len());
    
    // Process all models sequentially
    pipeline.process_models_sequential(&models);
    
    // Verify ModelMetadata was created for each model
    info!("Verifying ModelMetadata for all processed models...");
    for model_id in &models {
        match storage.load_model_metadata(model_id) {
            Ok(metadata) => {
                if metadata.is_processed {
                    info!("✅ FULLY PROCESSED {} - {} files", model_id, metadata.files.len());
                } else {
                    warn!("⏳ PARTIALLY PROCESSED {} - {} files", model_id, metadata.files.len());
                }
                if let Some(ref base_id) = metadata.base_model_id {
                    debug!("   Base model: {base_id}");
                }
            }
            Err(e) => {
                error!("Failed to load ModelMetadata for {model_id}: {e}");
            }
        }
    }
    
    info!("All models processing completed!");
    Ok(())
} 