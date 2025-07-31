import os
import json
import hashlib
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
import requests
import time
from urllib.parse import quote
import shutil
from threading import Lock
import safetensors.torch

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set")

def fetch_repo_files_with_sizes(repo_id, repo_type="model", revision="main", token=HF_TOKEN):
    BASE_URL = "https://huggingface.co"
    encoded_repo_id = quote(repo_id, safe="")
    encoded_revision = quote(revision, safe="")

    tree_url = f"{BASE_URL}/api/{repo_type}s/{encoded_repo_id}/tree/{encoded_revision}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    while True:
        try:
            response = requests.get(tree_url, params={"recursive": True, "expand": True}, headers=headers, timeout=30)
            if response.status_code == 200:
                files_data = response.json()
                return [{"path": file["path"], "size": file["size"]}
                        for file in files_data if file["type"] == "file"]
            elif response.status_code == 429:
                print("Rate limit reached. Sleeping for 10s...")
                time.sleep(10)
            else:
                print(f"Failed to fetch files for {repo_id}. Status code: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error fetching files for {repo_id}: {e}")
            return []

def download_a_file(model_name, file_rel_path, output_dir, token=HF_TOKEN):
    try:
        local_path = hf_hub_download(
            repo_id=model_name,
            filename=file_rel_path,
            local_dir=output_dir,
            use_auth_token=token
        )
        if os.path.islink(local_path):
            real_path = os.path.realpath(local_path)
            os.unlink(local_path)
            shutil.move(real_path, local_path)
        return local_path
    except Exception as e:
        print(f"Failed to download {file_rel_path}: {e}")
        return None

def download_files(model_name, files, output_dir, token=HF_TOKEN, suffix=[".safetensors"], max_workers=8):
    downloaded_files = []
    with ThreadPoolExecutor(max_workers) as executor:
        futures = []
        files = [file for file in files if file["path"].endswith(tuple(suffix))]
        for file in files:
            file_rel_path = file["path"]
            futures.append(executor.submit(download_a_file, model_name, file_rel_path, output_dir, token))

        for future in tqdm(futures, total=len(futures), desc=f"Downloading {model_name}"):
            local_path = future.result()
            if local_path:
                downloaded_files.append(local_path)
    return downloaded_files

def download_model_repo(model_name, output_dir, token=HF_TOKEN, suffix=[".safetensors"], max_workers=8):
    files = fetch_repo_files_with_sizes(model_name, token=token)
    model_path = model_name.replace("/", "_", 1)
    output_dir = os.path.join(output_dir, model_path)
    os.makedirs(output_dir, exist_ok=True)
    local_files = download_files(model_name, files, output_dir, token, suffix, max_workers)

    cache_folder = os.path.join(output_dir, ".cache")
    if os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)
    return local_files

def process_model_download(model_name, output_dir, token, suffix, already_downloads_file, lock, check_already_downloaded=True):
    model_name = model_name.strip()
    if not model_name:
        return

    if check_already_downloaded:
        with lock:
            if os.path.exists(already_downloads_file):
                with open(already_downloads_file, "r") as f:
                    already_downloaded = {line.strip() for line in f}
                if model_name in already_downloaded:
                    print(f"Model {model_name} already downloaded. Skipping.")
                    return

    downloaded_files = download_model_repo(model_name, output_dir, token, suffix)

    if downloaded_files:
        with lock:
            with open(already_downloads_file, "a") as f:
                f.write(model_name + "\n")

def read_safetensors_layers(folder_path):
    all_tensors = {}
    for file in os.listdir(folder_path):
        if file.endswith(".safetensors"):
            file_path = os.path.join(folder_path, file)
            tensors = safetensors.torch.load_file(file_path)
            all_tensors.update(tensors)
    return all_tensors

def save_combined_safetensors(folder_path, output_file):
    all_tensors = read_safetensors_layers(folder_path)
    sorted_tensors = dict(sorted(all_tensors.items(), key=lambda x: x[0]))
    safetensors.torch.save_file(sorted_tensors, output_file)
    print(f"Combined safetensors saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download models from Hugging Face')

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--model', type=str, help='Single model name to download')
    input_group.add_argument('--models_txts', type=str, nargs='+', help='One or more txt files containing model names')

    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save downloaded models')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of worker threads (default: 8)')
    parser.add_argument('--check_already_downloaded', action='store_true', help='Check if models are already downloaded')
    parser.add_argument('--already_downloads_file', type=str, default='already_downloads.txt', help='File to track already downloaded models')
    parser.add_argument('--suffix', nargs='+', default=['.safetensors'], help='File extensions to download (default: .safetensors)')

    args = parser.parse_args()

    output_dir = args.output_dir
    max_workers = args.max_workers
    check_already_downloaded = args.check_already_downloaded
    already_downloads_file = args.already_downloads_file
    suffix = args.suffix

    model_names = set()
    if args.model:
        model_names.add(args.model.strip())
    else:
        for file in args.models_txts:
            if not os.path.exists(file):
                print(f"Error: File not found: {file}")
                exit(1)
            with open(file, "r") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        model_names.add(name)

    if check_already_downloaded and os.path.exists(already_downloads_file):
        with open(already_downloads_file, "r") as f:
            already_downloaded = set(line.strip() for line in f)
        model_names = model_names - already_downloaded

    if not model_names:
        print("No models to download!")
        exit(0)

    print(f"Starting download of {len(model_names)} models...")
    print(f"Output directory: {output_dir}")
    print(f"Max workers: {max_workers}")
    print(f"Check already downloaded: {check_already_downloaded}")
    print(f"File suffixes: {suffix}")

    num_models = len(model_names)
    num_workers = min(max_workers, 4)
    models_per_worker = (num_models + num_workers - 1) // num_workers

    lock = Lock()
    model_list = list(model_names)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            chunk = model_list[i * models_per_worker:(i + 1) * models_per_worker]
            for model_name in chunk:
                futures.append(executor.submit(
                    process_model_download,
                    model_name,
                    output_dir,
                    HF_TOKEN,
                    suffix,
                    already_downloads_file,
                    lock,
                    check_already_downloaded
                ))
        for future in futures:
            future.result()

    print("Download completed!")
