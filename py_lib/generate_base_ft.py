import os
import json
from pathlib import Path

def generate_base_ft_json():
    """Generate base_ft.json mapping base models to their finetuned versions"""
    models_dir = Path(__file__).parent
    base_ft_data = {}
    
    for txt_file in models_dir.glob("*.txt"):
        filename = txt_file.stem
        base_model_name = filename.replace("_", "/", 1)
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                finetune_models = [line.strip() for line in f if line.strip()]
                
            base_ft_data[base_model_name] = finetune_models
            print(f"Processed: {txt_file.name} -> {base_model_name} ({len(finetune_models)} models)")
            
        except Exception as e:
            print(f"Error reading {txt_file.name}: {e}")
    
    output_file = models_dir / "base_ft.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(base_ft_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nGenerated {output_file}")
        print(f"Contains {len(base_ft_data)} base models:")
        for base_model, ft_models in base_ft_data.items():
            print(f"  - {base_model}: {len(ft_models)} finetuned models")
            
    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    generate_base_ft_json()