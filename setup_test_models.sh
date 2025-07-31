#!/bin/bash
#
# ZipLLM Test Models Setup Script
# ===============================
#
# This script automates the setup process for ZipLLM testing:
# 1. Downloads test models from test_models.txt
# 2. Generates base_ft.json mapping file
#
# Usage: ./setup_test_models.sh
#

set -e  # Exit on any error

echo "🚀 ZipLLM Test Models Setup"
echo "=========================="

# Check for HF_TOKEN environment variable
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable not set!"
    echo "💡 Please set your Hugging Face token:"
    echo "   export HF_TOKEN='your_hf_token_here'"
    exit 1
fi

# Install Python requirements
echo "📦 Step 0: Installing Python requirements..."
echo "==========================================="
pip install -r requirements.txt || {
    echo "❌ Failed to install Python requirements"
    exit 1
}

# Check if required files exist
if [ ! -f "test_models.txt" ]; then
    echo "❌ Error: test_models.txt not found!"
    echo "💡 Create test_models.txt with model IDs (one per line, first line = base model)"
    exit 1
fi

if [ ! -f "py_lib/download.py" ]; then
    echo "❌ Error: py_lib/download.py not found!"
    exit 1
fi

if [ ! -f "py_lib/generate_base_ft.py" ]; then
    echo "❌ Error: py_lib/generate_base_ft.py not found!"
    exit 1
fi

# Read configuration
MODEL_DIR=$(python3 -c "import json; print(json.load(open('config.json'))['model_dir'])" 2>/dev/null || echo "models")
BASE_FT_PATH=$(python3 -c "import json; print(json.load(open('config.json'))['base_ft_path'])" 2>/dev/null || echo "$MODEL_DIR/base_ft.json")
echo "📁 Model directory: $MODEL_DIR"
echo "📊 Base FT mapping: $BASE_FT_PATH"

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

echo ""
echo "⬇️  Step 1: Downloading test models..."
echo "======================================"

# Download models from test_models.txt with HF_TOKEN
HF_TOKEN=$HF_TOKEN python3 py_lib/download.py --models_txts test_models.txt --output_dir "$MODEL_DIR"

echo ""
echo "📊 Step 2: Generating base-finetune mapping..."
echo "=============================================="

# Generate base_ft.json file
BASE_FT_DIR=$(dirname "$BASE_FT_PATH")
mkdir -p "$BASE_FT_DIR"
python3 py_lib/generate_base_ft.py test_models.txt "$BASE_FT_PATH"

if [ -f "$BASE_FT_PATH" ]; then
    echo "✅ Generated base_ft.json:"
    echo "   📄 Location: $BASE_FT_PATH"
    echo "   📊 Content preview:"
    head -10 "$BASE_FT_PATH" | sed 's/^/      /'
else
    echo "❌ Failed to generate base_ft.json"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo "================================"
echo ""
echo "📋 Next steps:"
echo "  1. Build ZipLLM: cargo build --release"
echo "  2. Run compression: RUST_LOG=info ./target/release/zipllm"
echo "  3. Test restoration: ./target/release/restore <model_id> <output_dir>"
echo "  4. Run analysis: cd analysis && python3 throughput_analyzer.py"
echo ""
echo "💡 Model files are in: $MODEL_DIR/"
echo "💡 Use first model as base: $(head -1 test_models.txt)" 