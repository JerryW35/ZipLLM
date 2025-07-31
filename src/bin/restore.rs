use zipllm::*;
use std::sync::Arc;
use std::env;
use log::{info, error};

fn main() -> Result<()> {
    // Initialize logger with info level by default
    env_logger::Builder::from_default_env()
        .init();

    let args: Vec<String> = env::args().collect();
    
    if args.len() != 3 {
        eprintln!("Usage: {} <model_id> <output_directory>", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} NousResearch_DeepHermes-3-Llama-3-8B-Preview ./restored_models", args[0]);
        eprintln!("  {} meta-llama_Llama-3.1-8B ./output", args[0]);
        eprintln!();
        eprintln!("Performance Monitoring:");
        eprintln!("  Set RUST_LOG=debug to see detailed performance metrics");
        eprintln!("  Set RUST_LOG=info for basic progress information");
        eprintln!();
        eprintln!("Features:");
        eprintln!("  🚀 Parallel processing with memory mapping");
        eprintln!("  📊 Detailed performance analysis (IO vs Processing time)");
        eprintln!("  💾 Zero-copy operations for maximum efficiency");
        std::process::exit(1);
    }

    let raw_model_id = &args[1];
    let output_dir = &args[2];
    
    // Convert real model ID (e.g., "meta-llama/Llama-Guard-3-8B") to storage format (e.g., "meta-llama_Llama-Guard-3-8B")
    let model_id = if let Some(pos) = raw_model_id.find('/') {
        format!("{}{}{}", &raw_model_id[..pos], "_", &raw_model_id[pos+1..])
    } else {
        raw_model_id.to_string()
    };

    info!("🚀 ZipLLM Restore Tool - Parallel Memory Mapping Mode");
    if raw_model_id != &model_id {
        info!("📁 Input Model ID: {raw_model_id}");
        info!("📁 Storage Model ID: {model_id}");
    } else {
    info!("📁 Model ID: {model_id}");
    }
    info!("📁 Output Directory: {output_dir}");
    info!("🏁 Performance monitoring: Pure processing throughput separated from IO");

    // Create storage backend
    let storage = Arc::new(FileSystemStorage::new()?);
    
    // Create restore engine
    let restore_engine = RestoreEngine::new(storage.clone());

    // Check if model exists
    if !storage.exists_model(&model_id) {
        error!("❌ Model '{model_id}' not found in storage!");
        error!("Available models can be found in: {}", config::CONFIG.storage_dir);
        std::process::exit(1);
    }

    // Restore the model with performance monitoring
    match restore_engine.restore_model(&model_id, output_dir) {
        Ok(()) => {
            info!("🎉 Model restoration completed successfully!");
            info!("📁 Restored files are available in: {output_dir}");
        }
        Err(e) => {
            error!("❌ Model restoration failed: {e}");
            std::process::exit(1);
        }
    }

    Ok(())
} 