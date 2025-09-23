#!/usr/bin/env python3
import os
import json
import glob
import hashlib
import re
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any


def analyze_fastcdc_chunks(fastcdc_dir: str) -> Dict[str, Any]:
    """
    Analyze FastCDC chunk statistics
    
    Args:
        fastcdc_dir: Path to the fastcdc_chunks directory
        
    Returns:
        Dictionary with statistics
    """
    print("\n===== FastCDC Chunk Deduplication Statistics =====")
    
    # Get all models
    model_dirs = sorted(os.listdir(fastcdc_dir))
    
    all_chunks = []  # Store all chunks (with duplicates)
    unique_chunks = set()  # Store unique chunk hashes
    chunk_sizes = {}  # chunk_hash -> size
    
    for model_dir in model_dirs:
        model_path = os.path.join(fastcdc_dir, model_dir)
        chunk_files = glob.glob(os.path.join(model_path, "*.chunk.txt"))
        
        for chunk_file in chunk_files:
            with open(chunk_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        chunk_hash, chunk_size = parts[0], int(parts[1])
                        all_chunks.append(chunk_hash)
                        unique_chunks.add(chunk_hash)
                        chunk_sizes[chunk_hash] = chunk_size
    
    # Calculate statistics
    total_chunks = len(all_chunks)
    unique_chunk_count = len(unique_chunks)
    
    # Calculate size statistics
    chunk_size_values = list(chunk_sizes.values())
    max_chunk_size = max(chunk_size_values) if chunk_size_values else 0
    avg_chunk_size = sum(chunk_size_values) / len(chunk_size_values) if chunk_size_values else 0
    
    print(f"Total chunks: {total_chunks}")
    print(f"Unique chunks: {unique_chunk_count}")
    print(f"Unique ratio: {unique_chunk_count / total_chunks:.4f}")
    print(f"Max chunk size: {max_chunk_size} bytes ({max_chunk_size / 1024 / 1024:.2f} MB)")
    print(f"Average chunk size: {avg_chunk_size:.2f} bytes ({avg_chunk_size / 1024 / 1024:.2f} MB)")
    
    return {
        "total_chunks": total_chunks,
        "unique_chunks": unique_chunk_count,
        "unique_ratio": unique_chunk_count / total_chunks if total_chunks else 0,
        "max_chunk_size": max_chunk_size,
        "avg_chunk_size": avg_chunk_size
    }


def analyze_tensor_dedup(storage_dir: str) -> Dict[str, Any]:
    """
    Analyze tensor deduplication statistics
    
    Args:
        storage_dir: Path to the HF_storage directory
        
    Returns:
        Dictionary with statistics
    """
    print("\n===== Tensor Deduplication Statistics =====")
    
    # Get all models
    model_metadata_dir = os.path.join(storage_dir, "model_metadata")
    model_files = sorted(os.listdir(model_metadata_dir))
    
    # Track all tensors and unique tensors
    all_tensors = []  # Store all tensors (with duplicates)
    unique_tensors = set()  # Store unique tensor hashes
    tensor_sizes = {}  # tensor_hash -> size
    
    for model_file in model_files:
        model_path = os.path.join(model_metadata_dir, model_file)
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Get all tensors for this model
        for file_name, file_hash in model_data["files"].items():
            file_metadata_path = os.path.join(storage_dir, "file_metadata", f"{file_hash}.json")
            
            try:
                with open(file_metadata_path, 'r') as f:
                    file_data = json.load(f)
                
                # Add all tensors from this file
                for tensor_name, tensor_hash in file_data.get("tensor_hashes", {}).items():
                    all_tensors.append(tensor_hash)
                    unique_tensors.add(tensor_hash)
                    
                    # Get tensor size if not already recorded
                    if tensor_hash not in tensor_sizes:
                        tensor_metadata_path = os.path.join(storage_dir, "tensor_metadata", f"{tensor_hash}.json")
                        try:
                            with open(tensor_metadata_path, 'r') as f:
                                tensor_data = json.load(f)
                                tensor_sizes[tensor_hash] = tensor_data.get("original_size", 0)
                        except (FileNotFoundError, json.JSONDecodeError):
                            # If we can't find the tensor metadata, set size to 0
                            tensor_sizes[tensor_hash] = 0
            except (FileNotFoundError, json.JSONDecodeError):
                continue
    
    # Calculate statistics
    total_tensors = len(all_tensors)
    unique_tensor_count = len(unique_tensors)
    
    # Calculate size statistics
    tensor_size_values = list(tensor_sizes.values())
    max_tensor_size = max(tensor_size_values) if tensor_size_values else 0
    avg_tensor_size = sum(tensor_size_values) / len(tensor_size_values) if tensor_size_values else 0
    
    print(f"Total tensors: {total_tensors}")
    print(f"Unique tensors: {unique_tensor_count}")
    print(f"Unique ratio: {unique_tensor_count / total_tensors:.4f}")
    print(f"Max tensor size: {max_tensor_size} bytes ({max_tensor_size / 1024 / 1024:.2f} MB)")
    print(f"Average tensor size: {avg_tensor_size:.2f} bytes ({avg_tensor_size / 1024 / 1024:.2f} MB)")
    
    return {
        "total_tensors": total_tensors,
        "unique_tensors": unique_tensor_count,
        "unique_ratio": unique_tensor_count / total_tensors if total_tensors else 0,
        "max_tensor_size": max_tensor_size,
        "avg_tensor_size": avg_tensor_size
    }


def analyze_file_dedup(storage_dir: str) -> Dict[str, Any]:
    """
    Analyze file deduplication statistics
    
    Args:
        storage_dir: Path to the HF_storage directory
        
    Returns:
        Dictionary with statistics
    """
    print("\n===== File Deduplication Statistics =====")
    
    # Get all models
    model_metadata_dir = os.path.join(storage_dir, "model_metadata")
    model_files = sorted(os.listdir(model_metadata_dir))
    
    # Track all files and unique files
    all_files = []  # Store all files (with duplicates)
    unique_files = set()  # Store unique file hashes
    file_sizes = {}  # file_hash -> size
    
    for model_file in model_files:
        model_path = os.path.join(model_metadata_dir, model_file)
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Get all files for this model
        for file_name, file_hash in model_data["files"].items():
            all_files.append(file_hash)
            unique_files.add(file_hash)
            
            # Get file size if not already recorded
            if file_hash not in file_sizes:
                file_metadata_path = os.path.join(storage_dir, "file_metadata", f"{file_hash}.json")
                try:
                    with open(file_metadata_path, 'r') as f:
                        file_data = json.load(f)
                        file_sizes[file_hash] = file_data.get("size", 0)
                except (FileNotFoundError, json.JSONDecodeError):
                    # If we can't find the file metadata, set size to 0
                    file_sizes[file_hash] = 0
    
    # Calculate statistics
    total_files = len(all_files)
    unique_file_count = len(unique_files)
    
    # Calculate size statistics
    file_size_values = list(file_sizes.values())
    max_file_size = max(file_size_values) if file_size_values else 0
    avg_file_size = sum(file_size_values) / len(file_size_values) if file_size_values else 0
    
    print(f"Total files: {total_files}")
    print(f"Unique files: {unique_file_count}")
    print(f"Unique ratio: {unique_file_count / total_files:.4f}")
    print(f"Max file size: {max_file_size} bytes ({max_file_size / 1024 / 1024:.2f} MB)")
    print(f"Average file size: {avg_file_size:.2f} bytes ({avg_file_size / 1024 / 1024:.2f} MB)")
    
    return {
        "total_files": total_files,
        "unique_files": unique_file_count,
        "unique_ratio": unique_file_count / total_files if total_files else 0,
        "max_file_size": max_file_size,
        "avg_file_size": avg_file_size
    }


def analyze_layer_dedup(storage_dir: str) -> Dict[str, Any]:
    """
    Analyze layer deduplication statistics
    
    Args:
        storage_dir: Path to the HF_storage directory
        
    Returns:
        Dictionary with statistics
    """
    print("\n===== Layer Deduplication Statistics =====")
    
    # Get all models
    model_metadata_dir = os.path.join(storage_dir, "model_metadata")
    model_files = sorted(os.listdir(model_metadata_dir))
    
    # Track layers by their structure pattern, not just by layer number
    # A layer is defined by its tensor name patterns (e.g., self_attn.q_proj, mlp.gate_proj, etc.)
    all_models_layers = []  # List of all layers across all models
    tensor_sizes = {}  # tensor_hash -> size
    
    # First pass: collect all tensors and their metadata
    for model_file in model_files:
        model_path = os.path.join(model_metadata_dir, model_file)
        with open(model_path, 'r') as f:
            model_data = json.load(f)
        
        # Group tensors by layer for this model
        model_layers = defaultdict(dict)  # layer_key -> {tensor_type: tensor_hash}
        non_layer_tensors = {}  # tensor_name -> tensor_hash
        
        # Get all tensors for this model
        for file_name, file_hash in model_data["files"].items():
            file_metadata_path = os.path.join(storage_dir, "file_metadata", f"{file_hash}.json")
            
            try:
                with open(file_metadata_path, 'r') as f:
                    file_data = json.load(f)
                
                # Process all tensors from this file
                for tensor_name, tensor_hash in file_data.get("tensor_hashes", {}).items():
                    # Get tensor size if not already recorded
                    if tensor_hash not in tensor_sizes:
                        tensor_metadata_path = os.path.join(storage_dir, "tensor_metadata", f"{tensor_hash}.json")
                        try:
                            with open(tensor_metadata_path, 'r') as f:
                                tensor_data = json.load(f)
                                tensor_sizes[tensor_hash] = tensor_data.get("original_size", 0)
                        except (FileNotFoundError, json.JSONDecodeError):
                            tensor_sizes[tensor_hash] = 0
                    
                    # Parse tensor name to identify layer structure
                    # Examples:
                    # model.layers.0.self_attn.q_proj.weight -> layer_id=0, tensor_type=self_attn.q_proj.weight
                    # model.layers.1.mlp.gate_proj.weight -> layer_id=1, tensor_type=mlp.gate_proj.weight
                    parts = tensor_name.split('.')
                    if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers' and parts[2].isdigit():
                        layer_id = int(parts[2])
                        tensor_type = '.'.join(parts[3:])  # e.g., self_attn.q_proj.weight
                        model_layers[layer_id][tensor_type] = tensor_hash
                    else:
                        # Non-layer tensor (e.g., embedding)
                        non_layer_tensors[tensor_name] = tensor_hash
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        
        # Add this model's layers to the global list
        all_models_layers.append((model_layers, non_layer_tensors))
    
    # Second pass: analyze layer structures and identify unique layers
    # A layer is considered unique if it has a unique set of tensor types and values
    unique_layer_structures = {}  # layer_structure_hash -> (layer_info, size)
    
    # Process each model's layers
    for model_layers, non_layer_tensors in all_models_layers:
        # Process regular layers
        for layer_id, tensor_dict in model_layers.items():
            # Sort tensor types for consistent hashing
            tensor_types = sorted(tensor_dict.keys())
            
            # Create a deterministic representation of this layer's structure
            layer_structure = []
            for tensor_type in tensor_types:
                tensor_hash = tensor_dict[tensor_type]
                layer_structure.append(f"{tensor_type}:{tensor_hash}")
            
            # Hash the layer structure
            layer_structure_str = "|".join(layer_structure)
            layer_hash = hashlib.sha256(layer_structure_str.encode()).hexdigest()
            
            # Calculate layer size
            layer_size = sum(tensor_sizes.get(tensor_dict[tensor_type], 0) for tensor_type in tensor_types)
            
            # Store unique layer structure
            unique_layer_structures[layer_hash] = (f"layer_{layer_id}", layer_size)
        
        # Process non-layer tensors (each as its own "layer")
        for tensor_name, tensor_hash in non_layer_tensors.items():
            layer_hash = hashlib.sha256(f"{tensor_name}:{tensor_hash}".encode()).hexdigest()
            unique_layer_structures[layer_hash] = (tensor_name, tensor_sizes.get(tensor_hash, 0))
    
    # Calculate statistics
    total_layers = sum(len(model_layers) + len(non_layer_tensors) for model_layers, non_layer_tensors in all_models_layers)
    unique_layer_count = len(unique_layer_structures)
    
    # Calculate layer size statistics
    layer_sizes = [size for _, size in unique_layer_structures.values()]
    max_layer_size = max(layer_sizes) if layer_sizes else 0
    avg_layer_size = sum(layer_sizes) / len(layer_sizes) if layer_sizes else 0
    
    print(f"Total layers: {total_layers}")
    print(f"Unique layers: {unique_layer_count}")
    print(f"Unique ratio: {unique_layer_count / total_layers if total_layers else 0:.4f}")
    print(f"Max layer size: {max_layer_size} bytes ({max_layer_size / 1024 / 1024:.2f} MB)")
    print(f"Average layer size: {avg_layer_size:.2f} bytes ({avg_layer_size / 1024 / 1024:.2f} MB)")
    
    
    return {
        "total_layers": total_layers,
        "unique_layers": unique_layer_count,
        "unique_ratio": unique_layer_count / total_layers if total_layers else 0,
        "max_layer_size": max_layer_size,
        "avg_layer_size": avg_layer_size
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze deduplication statistics')
    parser.add_argument('--zipllm_dir', required=True, help='Path to the HF_storage directory')
    parser.add_argument('--fastcdc_dir', required=True, help='Path to the fastcdc_chunks directory')
    
    args = parser.parse_args()
    
    # Analyze FastCDC chunks
    chunk_stats = analyze_fastcdc_chunks(args.fastcdc_dir)
    
    # Analyze tensor deduplication
    tensor_stats = analyze_tensor_dedup(args.zipllm_dir)
    
    # Analyze file deduplication
    file_stats = analyze_file_dedup(args.zipllm_dir)
    
    # Analyze layer deduplication
    layer_stats = analyze_layer_dedup(args.zipllm_dir)
    
    print("\n===== Summary =====")
    print(f"FastCDC: {chunk_stats['total_chunks']} total, {chunk_stats['unique_chunks']} unique chunks, {chunk_stats['unique_ratio']:.4f} ratio, {chunk_stats['avg_chunk_size']/1024/1024:.2f} MB avg size")
    print(f"Tensor: {tensor_stats['total_tensors']} total, {tensor_stats['unique_tensors']} unique tensors, {tensor_stats['unique_ratio']:.4f} ratio, {tensor_stats['avg_tensor_size']/1024/1024:.2f} MB avg size")
    print(f"File: {file_stats['total_files']} total, {file_stats['unique_files']} unique files, {file_stats['unique_ratio']:.4f} ratio, {file_stats['avg_file_size']/1024/1024:.2f} MB avg size")
    print(f"Layer: {layer_stats['total_layers']} total, {layer_stats['unique_layers']} unique layers, {layer_stats['unique_ratio']:.4f} ratio, {layer_stats['avg_layer_size']/1024/1024:.2f} MB avg size")


if __name__ == "__main__":
    main()
