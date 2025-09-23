#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


def check_and_install_zipnn():
    try:
        import zipnn 
    except ImportError:
        print("zipnn not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "zipnn", "--upgrade"])


def parse_streaming_chunk_size(streaming_chunk_size):
    s = str(streaming_chunk_size).strip()
    if s.isdigit():
        return int(s)

    s = s.lower()
    import re
    m = re.fullmatch(r"(\d+)\s*([a-zA-Z]+)", s)
    if not m:
        raise ValueError(f"Invalid size format: {streaming_chunk_size}. Use e.g. 512k, 1m, 1g, or raw bytes.")

    size_value = int(m.group(1))
    unit = m.group(2)
    unit_map = {
        "k": KB,
        "kb": KB,
        "m": MB,
        "mb": MB,
        "g": GB,
        "gb": GB,
    }

    if unit not in unit_map:
        raise ValueError(f"Invalid size unit: {unit}. Use 'k', 'm', or 'g' (optionally with 'b').")

    return size_value * unit_map[unit]


def compress_once(input_file: str, dtype: str = "bfloat16", method: str = "HUFFMAN", threads: int = 32,
                  streaming_chunk_size: str | int | None = None) -> tuple[int, int]:
    import zipnn

    full_path = os.path.abspath(input_file)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")

    if streaming_chunk_size is None:
        chunk_bytes = MB
    else:
        chunk_bytes = parse_streaming_chunk_size(streaming_chunk_size)

    zpn = zipnn.ZipNN(
        bytearray_dtype=dtype,
        is_streaming=False,
        streaming_chunk=chunk_bytes,
        method=method,
        threads=threads,
    )

    size_before = 0
    size_after = 0

    with open(full_path, "rb") as infile:
        data = infile.read()
        size_before = len(data)
        compressed = zpn.compress(data)
        if compressed:
            size_after = len(compressed)

    return size_before, size_after


def main():
    parser = argparse.ArgumentParser(description="Compress a .safetensors file and print sizes before/after.")
    parser.add_argument("input_file", type=str, help="Path to a .safetensors file")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32", "float8_e4m3fn", "float8_e5m2"],
                        help="Data type for zipnn")
    parser.add_argument("--method", type=str, default="HUFFMAN", choices=["HUFFMAN", "ZSTD", "FSE", "AUTO"],
                        help="Compression method")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads for compression")
    parser.add_argument("--streaming_chunk_size", type=str, default="1mb",
                        help="Chunk size like 512k/1m/1g or raw bytes integer")

    args = parser.parse_args()

    if not args.input_file.endswith(".safetensors"):
        print(f"{YELLOW}Warning: input is not a .safetensors file{RESET}")

    check_and_install_zipnn()

    try:
        size_before, size_after = compress_once(
            args.input_file,
            dtype=args.dtype,
            method=args.method,
            threads=args.threads,
            streaming_chunk_size=args.streaming_chunk_size,
        )
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        sys.exit(1)

    print(f"Original size: {size_before} bytes")
    print(f"Compressed size: {size_after} bytes")
    if size_before > 0 and size_after > 0:
        ratio = size_after / size_before * 100
        print(f"Remaining: {ratio:.2f}% of original")


if __name__ == "__main__":
    main()

