# ZipLLM - Model Compression and Restoration System

A Rust-based model compression system using tensor deduplication, BitX differential compression, and Zstd compression for machine learning models.

## Project Structure

```
zipllm_rust/
├── src/
│   ├── lib.rs                 # Main library entry point
│   ├── main.rs                # Main compression pipeline
│   ├── config.rs              # Configuration loader (reads config.json)
│   ├── restore.rs             # Model restoration binary
│   ├── storage.rs             # Storage backend implementation
│   ├── pipeline.rs            # Processing pipeline logic
│   ├── deduplication.rs       # Tensor deduplication algorithms
│   ├── compression.rs         # Compression strategy selection
│   └── bitx/
│       └── bitx_bytes.rs      # BitX differential compression implementation
├── analysis/
│   └── throughput_analyzer.py # Threading performance analysis tool
├── py_lib/                    # Python utilities
│   ├── download.py            # Model download script
│   └── generate_base_ft.py    # Base-finetune mapping generator
├── config.json                # Configuration file (editable without recompiling)
├── test_models.txt            # Test model list (first line = base model)
├── setup_test_models.sh       # Automated test setup script
├── models/                    # Original Model storage directory
└── storage/                   # ZipLLM Model storage
    ├── model_metadata/        # Model information and file lists
    ├── file_metadata/         # Safetensors file structure data
    ├── tensors/               # Original tensor data
    ├── compressed_tensors/    # Compressed tensor storage
    ├── compressed_metadata/   # Compression metadata
    └── safetensors_headers/   # Original file headers
```

### Configuration

Configuration is managed through `config.json` in the project root. The file is automatically loaded at runtime - no recompilation needed when changing settings.

**Example config.json:**
```json
{
  "model_dir": "./models",
  "storage_dir": "/mnt/HF_storage",
  "models_to_process": "./models/meta-llama_Llama-3.1-8B_clean.txt",
  "base_ft_path": "./models/base_ft.json"
}
```

**Settings:**
- **model_dir**: Directory for downloaded original models
- **storage_dir**: ZipLLM compressed storage location  
- **models_to_process**: Text file listing models to compress (one per line)
- **base_ft_path**: Complete path to base-finetune mapping JSON file

If `config.json` doesn't exist, default values are used and a warning is displayed.

## ZipLLM Storage Usage

### 1. Model Compression
```bash
# Process models listed in models file
cargo run --release
```

The storage system:
- Downloads models from Hugging Face
- Analyzes tensors for deduplication opportunities
- Applies optimal compression (BitX/Zstd/None) per tensor
- Stores compressed data with metadata for efficient restoration

### 2. Storage Organization
- **Model Metadata**: Tracks files, compression stats, and processing status
- **Tensor Storage**: Original and compressed tensor data with efficient indexing
- **Deduplication**: References to shared tensors across models
- **Headers**: Safetensors structure preservation for exact restoration

## Model Restoration

### Basic Usage
```bash
# Restore a compressed model using real model ID
./target/release/restore <model_id> <output_directory>

# Examples - both formats work:
./target/release/restore meta-llama/Llama-Guard-3-8B /tmp/restored_model
```

### Input Requirements
- **Model ID**: Real model identifier (e.g., `meta-llama/Llama-Guard-3-8B`) - automatically converted to storage format
- **Output Directory**: Where to restore the original model files
- **Storage**: Existing compressed data in the storage/ directory


### Process Flow
1. **Load Metadata**: Read model structure and compression information
2. **Batch Loading**: Efficiently load compressed tensor data
3. **Parallel Decompression**: Decompress tensors using available CPU cores
4. **File Reconstruction**: Rebuild original safetensors files with exact structure
5. **Verification**: Ensure data integrity and completeness

## Threading Performance Analysis

### Throughput Analysis Tool

Test restore performance across different thread counts:

```bash
# From analysis/ directory
cd analysis
python3 throughput_analyzer.py
```

### Features
- **Automated Testing**: Benchmarks 1, 2, 4, 8, 16, 32, 48 threads
- **Decompression Metrics**: Extracts throughput data from restore logs
- **CSV Reports**: Generates detailed performance statistics
- **Visualization**: Creates plots showing throughput vs thread count
- **Efficiency Analysis**: Compares actual vs ideal linear scaling

### Sample Output
```csv
threads,avg_throughput,min_throughput,max_throughput,num_records
1,0.390,0.390,0.390,4
2,0.425,0.380,0.450,4
4,0.445,0.420,0.480,4
8,0.420,0.390,0.450,4
16,0.435,0.400,0.470,4
32,0.440,0.410,0.480,4
48,0.455,0.430,0.490,4
```

The analysis helps identify optimal thread counts for your hardware configuration and workload characteristics.

## Test Model Setup

### Automated Setup Script

Use the provided bash script to automatically download test models and generate required mapping files:

```bash
# Run the automated setup
./setup_test_models.sh
```

**What the script does:**
1. **Downloads Models**: Downloads all models listed in `test_models.txt` using `py_lib/download.py`
2. **Generates Mapping**: Creates `base_ft.json` mapping file using `py_lib/generate_base_ft.py`
3. **Validates Setup**: Checks all required files and directories exist

### Test Models Configuration

The `test_models.txt` file contains the list of models for testing:
- **First line**: Base model (e.g., `meta-llama/Llama-3.1-8B`)
- **Subsequent lines**: Fine-tuned models that derive from the base model
- **Format**: One model ID per line using Hugging Face format (`org/model-name`)

**Example test_models.txt:**
```
meta-llama/Llama-3.1-8B
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-Guard-3-8B
NousResearch/Hermes-3-Llama-3.1-8B
mlabonne/TwinLlama-3.1-8B
```

## Complete Testing Workflow

### 1. Initial Setup
```bash
# Edit configuration if needed
vim config.json

# Download models and generate mappings
./setup_test_models.sh
```

### 2. Build and Run Compression
```bash
# Build the project
cargo build --release

# Run compression with detailed logging
RUST_LOG=info ./target/release/zipllm
```

### 3. Test Model Restoration
```bash
# Restore a specific model
./target/release/restore meta-llama/Llama-3.1-8B-Instruct /tmp/restored_model

# Verify restored files
ls -la /tmp/restored_model/
```

### 4. Performance Analysis
```bash
# Run threading performance analysis
cd analysis
python3 throughput_analyzer.py
```

## Quick Start

For a complete testing workflow, see [Complete Testing Workflow](#complete-testing-workflow) above.

**Minimal setup:**
1. **Setup**: `./setup_test_models.sh`
2. **Build**: `cargo build --release`
3. **Compress**: `RUST_LOG=info ./target/release/zipllm`
4. **Restore**: `./target/release/restore meta-llama/Llama-3.1-8B-Instruct /tmp/output`