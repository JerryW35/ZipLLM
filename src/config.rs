use serde::{Serialize, Deserialize};
use once_cell::sync::Lazy;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};
use log::{info, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub model_dir: String,
    pub storage_dir: String,
    pub models_to_process: String,
    pub base_ft_path: String,
    pub threads: usize,
    
    #[serde(skip)]
    pub config_dir: PathBuf,
}

impl AppConfig {
    /// Get the absolute path for model_dir, relative to config file location
    pub fn get_model_dir(&self) -> PathBuf {
        if Path::new(&self.model_dir).is_absolute() {
            PathBuf::from(&self.model_dir)
        } else {
            self.config_dir.join(&self.model_dir)
        }
    }
    
    /// Get the absolute path for storage_dir, relative to config file location
    pub fn get_storage_dir(&self) -> PathBuf {
        if Path::new(&self.storage_dir).is_absolute() {
            PathBuf::from(&self.storage_dir)
        } else {
            self.config_dir.join(&self.storage_dir)
        }
    }
    
    /// Get the absolute path for models_to_process, relative to config file location
    pub fn get_models_to_process(&self) -> PathBuf {
        if Path::new(&self.models_to_process).is_absolute() {
            PathBuf::from(&self.models_to_process)
        } else {
            self.config_dir.join(&self.models_to_process)
        }
    }
    
    /// Get the absolute path for base_ft_path, relative to config file location
    pub fn get_base_ft_path(&self) -> PathBuf {
        if Path::new(&self.base_ft_path).is_absolute() {
            PathBuf::from(&self.base_ft_path)
        } else {
            self.config_dir.join(&self.base_ft_path)
        }
    }
    
    /// Load configuration from JSON file, fallback to default if parsing fails
    pub fn load_from_path<P: AsRef<Path>>(config_path: P) -> Self {
        let config_path = config_path.as_ref();
        
        match fs::read_to_string(config_path) {
            Ok(content) => {
                match serde_json::from_str::<AppConfig>(&content) {
                    Ok(config) => {
                        // Store the directory of the config file to resolve relative paths
                        let mut config = config;
                        config.config_dir = config_path.parent().unwrap_or(Path::new(".")).to_path_buf();
                        
                        info!("📄 Configuration loaded from {}", config_path.display());
                        info!("  Model directory: {}", config.get_model_dir().display());
                        info!("  Storage directory: {}", config.get_storage_dir().display());
                        info!("  Models file: {}", config.get_models_to_process().display());
                        info!("  Base FT mapping: {}", config.get_base_ft_path().display());
                        info!("  Threads: {}", config.threads);
                        config
                    }
                    Err(e) => {
                        error!("❌ Failed to parse {}: {e}", config_path.display());
                        warn!("🔄 Using default configuration");
                        Self::default()
                    }
                }
            }
            Err(e) => {
                error!("❌ Could not read {}: {e}", config_path.display());
                panic!("Config file not found or not readable: {}", config_path.display())
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
            config_dir: PathBuf::from("."),
        }
    }
}

static INIT: Once = Once::new();
static CONFIG_INSTANCE: Lazy<Mutex<Option<AppConfig>>> = Lazy::new(|| Mutex::new(None));

pub fn set_config(config_path: &str) {
    INIT.call_once(|| {
        let config = AppConfig::load_from_path(config_path);
        let mut guard = CONFIG_INSTANCE.lock().unwrap();
        *guard = Some(config);
    });
}

pub static CONFIG: Lazy<AppConfig> = Lazy::new(|| {
    CONFIG_INSTANCE.lock().unwrap()
        .clone()
        .expect("CONFIG must be initialized with set_config() before use")
});