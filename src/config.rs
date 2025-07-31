use serde::{Serialize, Deserialize};
use once_cell::sync::Lazy;
use std::fs;
use log::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub model_dir: String,
    pub storage_dir: String,
    pub models_to_process: String,
    pub base_ft_path: String,
    pub threads: usize,
}

impl AppConfig {
    /// Load configuration from JSON file, fallback to default if file doesn't exist or parsing fails
    pub fn load() -> Self {
        let config_path = "config.json";
        
        match fs::read_to_string(config_path) {
            Ok(content) => {
                match serde_json::from_str::<AppConfig>(&content) {
                    Ok(config) => {
                        info!("📄 Configuration loaded from {config_path}");
                        info!("  Model directory: {}", config.model_dir);
                        info!("  Storage directory: {}", config.storage_dir);
                        info!("  Models file: {}", config.models_to_process);
                        info!("  Base FT mapping: {}", config.base_ft_path);
                        info!("  Threads: {}", config.threads);
                        config
                    }
                    Err(e) => {
                        error!("❌ Failed to parse {config_path}: {e}");
                        warn!("🔄 Using default configuration");
                        Self::default()
                    }
                }
            }
            Err(e) => {
                warn!("⚠️  Could not read {config_path}: {e}");
                warn!("🔄 Using default configuration");
                info!("💡 Create {config_path} to customize settings");
                Self::default()
            }
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
            
        Self {
            model_dir: "models".to_string(),
            storage_dir: "/mnt/HF_storage".to_string(),
            models_to_process: "./test_models.txt".to_string(),
            base_ft_path: "./base_ft.json".to_string(),
            threads,
        }
    }
}

pub static CONFIG: Lazy<AppConfig> = Lazy::new(|| {
    AppConfig::load()
});