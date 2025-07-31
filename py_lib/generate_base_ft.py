import os
import json
import sys
from pathlib import Path

def generate_base_ft_json(models_file, output_file):
    """Generate base_ft.json mapping base models to their finetuned versions"""
    
    if not os.path.exists(models_file):
        print(f"❌ Error: Models file {models_file} not found!")
        return False
        
    try:
        with open(models_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
        if len(lines) < 2:
            print(f"❌ Error: Models file must contain at least 2 models (base + finetunes)")
            return False
            
        # First line is the base model, rest are finetunes
        base_model = lines[0]
        finetune_models = lines[1:]
        
        base_ft_data = {
            base_model: finetune_models
        }
        
        print(f"📊 Processing models from {models_file}:")
            
    except Exception as e:
        print(f"❌ Error reading {models_file}: {e}")
        return False
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(base_ft_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Generated {output_file}")
        print(f"📋 Contains mapping:")
        print(f"   {base_model} -> {len(finetune_models)} finetuned models")
        return True
            
    except Exception as e:
        print(f"❌ Error writing {output_file}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 generate_base_ft.py <models.txt> <output_base_ft.json>")
        print("Example: python3 generate_base_ft.py test_models.txt ./models/base_ft.json")
        sys.exit(1)
    
    models_file = sys.argv[1] 
    output_file = sys.argv[2]
    
    success = generate_base_ft_json(models_file, output_file)
    if not success:
        sys.exit(1)