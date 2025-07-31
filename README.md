# ZipLLM - Efficient LLM Storage via Model-Aware Synergistic Data Deduplication and Compression

ZipLLM is an efficient LLM storage system that significantly reduces storage cost through tensor-level deduplication and BitX compression.

## Prerequisites

### Install Rust
```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### Python Dependencies
```bash
# Install required Python packages
pip3 install -r requirements.txt
```

### Hugging Face Token
```bash
# Set your HF token for model downloads
export HF_TOKEN=your_token_here
```

## Quick Start

### 1. Setup Test Environment
```bash
# Configure paths (edit if needed)
vim config.json

# Download test models and generate base-finetune mapping
./setup_test_models.sh
```
### 2. Build Project
```bash
cargo build --release
```

### 3. Run Model Compression
```bash
RUST_LOG=info ./target/release/zipllm
```

### 4. Restore Models
```bash
# Restore using real model ID format
RUST_LOG=info ./target/release/restore meta-llama/Llama-3.1-8B-Instruct /tmp/output

# Verify restoration
ls -la /tmp/output/
```

## Performance Testing

Test restore performance across different thread counts:

```bash
cd analysis
python3 throughput_analyzer.py
```

Generates CSV reports and visualization plots showing throughput vs thread count relationships.

## Project Structure

```
zipllm_rust/
├── src/
│   ├── main.rs                # Main compression pipeline
│   ├── restore.rs             # Model restoration binary
│   ├── config.rs              # Configuration loader
│   ├── storage.rs             # Storage backend
│   ├── pipeline.rs            # Processing pipeline
│   ├── deduplication.rs       # Tensor deduplication
│   ├── compression.rs         # Compression strategies
│   └── bitx/bitx_bytes.rs     # BitX differential compression
├── examples/
│   ├── bitx.rs                # Standalone BitX tool
│   └── restore_example.rs     # API usage example
├── analysis/
│   └── throughput_analyzer.py # Performance analysis
├── py_lib/
│   ├── download.py            # Model downloader
│   └── generate_base_ft.py    # Base-finetune mapper
├── config.json                # Configuration file
├── test_models.txt            # Test model list
├── setup_test_models.sh       # Automated setup
├── models/                    # Downloaded models
└── storage/                   # Compressed data
```

## Configuration

Edit `config.json` to customize paths:

```json
{
  "model_dir": "./models",
  "storage_dir": "/mnt/HF_storage",
  "models_to_process": "./models/models.txt",
  "base_ft_path": "./models/base_ft.json"
}
```

## BitX Standalone Tool

```bash
# Build and use BitX for file compression
cargo build --release --example bitx
./target/release/examples/bitx file1.bin file2.bin --compress \
  --compressed-exp exp.zst --compressed-sm mantissa.zst
```

## Important Notes
- **Support Dtype**:⚠️ Current version only supports BF16.
- **Logging**: Use `RUST_LOG=info` to see runtime progress and performance metrics
- **Test Models**: First line in `test_models.txt` must be the base model
- **Model IDs**: Use real Hugging Face format (`org/model-name`) - automatic conversion to storage format
- **HF_token**: To download models, don't forget to set your HF_token in the enviroments.
