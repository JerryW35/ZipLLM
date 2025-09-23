#!/usr/bin/env python3
import os
import sys
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / 'config.json'
FASTCDC_DIR = SCRIPT_DIR / 'fastcdc_exp'
FASTCDC_BIN = FASTCDC_DIR / 'target' / 'release' / 'fastcdc_exp'
ZIPNN_PY = SCRIPT_DIR / 'zipnn_exp.py'


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    output_root = cfg.get('output_root') or './'
    hf_root = cfg.get('model_root') or './models'
    fastcdc_avg_size = cfg.get('fastcdc_avg_size') or 65536
    print(f"output_root: {output_root}")
    print(f"hf_root: {hf_root}")
    print(f"fastcdc_avg_size: {fastcdc_avg_size}")
    return {
        'output_root': str(Path(output_root)),
        'hf_root': str(Path(hf_root)),
        'fastcdc_avg_size': int(fastcdc_avg_size),
    }


def ensure_fastcdc_built() -> None:
    if FASTCDC_BIN.exists():
        return
    subprocess.check_call([
        'cargo', 'build', '--manifest-path', str(FASTCDC_DIR / 'Cargo.toml'), '--release'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_fastcdc(input_file: Path, out_txt: Path, avg_size: int) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        str(FASTCDC_BIN), '-s', str(avg_size), '-o', str(out_txt), str(input_file)
    ])


def run_zipnn(input_file: Path, threads: int) -> Dict[str, int]:
    # Capture stdout and parse Original/Compressed sizes
    proc = subprocess.run(
        [sys.executable, str(ZIPNN_PY), str(input_file), "--threads", str(threads)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    out = proc.stdout.splitlines()
    orig, comp = 0, 0
    for line in out:
        line = line.strip()
        if line.startswith('Original size:') and line.endswith('bytes'):
            try:
                orig = int(line.split(':', 1)[1].strip().split()[0])
            except Exception:
                pass
        if line.startswith('Compressed size:') and line.endswith('bytes'):
            try:
                comp = int(line.split(':', 1)[1].strip().split()[0])
            except Exception:
                pass
    return {'original_size': orig, 'compressed_size': comp}


def load_existing_results(result_json_path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Load existing results from JSON file if it exists"""
    if result_json_path.exists():
        try:
            with open(result_json_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing results from {result_json_path}: {e}")
    return {}


def update_results_file(result_json_path: Path, results: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    """Update the results JSON file with current results"""
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(result_json_path, 'w') as f:
            json.dump(results, f, indent=2)
    except IOError as e:
        print(f"Warning: Failed to update results file {result_json_path}: {e}")


def main() -> None:
    cfg = load_config()
    output_root = Path(cfg['output_root']).resolve()
    hf_root = Path(cfg['hf_root']).resolve()

    ensure_fastcdc_built()

    # Prepare results file path
    result_json = output_root / 'zipnn_results.json'
    
    # Load existing results if any
    results = load_existing_results(result_json)
    print(f"Loaded {sum(len(model_files) for model_files in results.values())} existing results")

    files_to_process = []
    for root, _, files in os.walk(hf_root):
        for name in files:
            if not name.endswith('.safetensors'):
                continue
            input_path = Path(root) / name
            rel = input_path.relative_to(hf_root)
            parts = rel.parts
            if len(parts) == 0:
                continue
            model_name = parts[0]
            out_dir = output_root / 'fastcdc_chunks' / model_name
            out_txt = out_dir / f"{name}.chunk.txt"
            
            # Skip if already processed
            if model_name in results and name in results[model_name]:
                print(f"Skipping already processed file: {model_name}/{name}")
                continue
                
            files_to_process.append((input_path, model_name, name, out_txt))

    file_count = len(files_to_process)
    print(f"Found {file_count} new files to process")

    # Parallel FastCDC stage
    if file_count == 0:
        print(f"No new .safetensors files found under: {hf_root}")
    else:
        max_workers = os.cpu_count() or 1
        fastcdc_avg_size = cfg['fastcdc_avg_size']
        print(f"Running FastCDC in parallel with {max_workers} workers for {file_count} file(s)...")
        print(f"Using average chunk size: {fastcdc_avg_size} bytes")
        
        # Create progress bar for FastCDC
        pbar = tqdm(total=file_count, desc="FastCDC", unit="file")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(run_fastcdc, ipath, otxt, fastcdc_avg_size): (ipath, model_name, name, otxt)
                for (ipath, model_name, name, otxt) in files_to_process
            }
            for future in as_completed(future_to_file):
                ipath, model_name, name, otxt = future_to_file[future]
                try:
                    future.result()
                    pbar.update(1)
                    pbar.set_postfix(file=f"{model_name}/{name}")
                except Exception as e:
                    print(f"FastCDC failed for {ipath}: {e}")
                    pbar.update(1)
        
        pbar.close()

    # ZipNN stage (sequential, each with max threads)
    max_zipnn_threads = os.cpu_count() or 1
    
    # Create progress bar for ZipNN
    if file_count > 0:
        print("\nRunning ZipNN compression...")
        pbar = tqdm(total=file_count, desc="ZipNN", unit="file")
    
    processed_count = 0
    
    for (ipath, model_name, name, _otxt) in files_to_process:
        pbar.set_description(f"ZipNN: {model_name}/{name}")
        try:
            res = run_zipnn(ipath, max_zipnn_threads)
            results.setdefault(model_name, {})[name] = res
            
            # Update JSON file after each file is processed
            update_results_file(result_json, results)
            
            processed_count += 1
            
            # Update progress bar with compression ratio
            orig = res.get('original_size', 0)
            comp = res.get('compressed_size', 0)
            ratio = (orig - comp) / orig if orig > 0 else 0
            pbar.set_postfix(file=f"{model_name}/{name}", 
                            ratio=f"{ratio:.2%}", 
                            orig=f"{orig/(1024*1024):.2f}MB", 
                            comp=f"{comp/(1024*1024):.2f}MB")
            pbar.update(1)
            
        except Exception as e:
            print(f"ZipNN failed for {ipath}: {e}")
            pbar.update(1)
    
    if file_count > 0:
        pbar.close()

    if processed_count > 0:
        print(f"Processed {processed_count} new .safetensors file(s)")
    
    print(f"Results saved to {result_json}")


if __name__ == '__main__':
    main()