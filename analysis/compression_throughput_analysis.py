#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BitX Compression Benchmark Analysis Script

This script runs the benchmark_bitx executable with different thread counts
and analyzes the compression and decompression throughput.
"""

import os
import re
import subprocess
import argparse
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run BitX compression benchmark with different thread counts'
    )
    parser.add_argument(
        'base_dir',
        help='Path to base model directory'
    )
    parser.add_argument(
        'finetune_dir',
        help='Path to finetune model directory'
    )
    parser.add_argument(
        '--threads',
        type=str,
        default='1,2,4,8,16,32,64',
        help='Comma-separated list of thread counts to test (default: 1,2,4,8,16,32,64)'
    )
    parser.add_argument(
        '--no-decompress',
        action='store_true',
        help='Skip decompression test'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='bitx_benchmark_results',
        help='Output prefix for CSV and PNG files (default: bitx_benchmark_results)'
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help='Number of times to repeat each test (default: 1)'
    )
    
    return parser.parse_args()


def run_benchmark(base_dir, finetune_dir, threads, run_decompress=True, benchmark_path=None):
    """Run the benchmark_bitx executable with specified parameters."""
    env = os.environ.copy()
    env['BITX_THREADS'] = str(threads)
    env['RUST_LOG'] = 'info'
    
    if not benchmark_path:
        raise ValueError("benchmark_path must be provided")
    
    cmd = [str(benchmark_path), base_dir, finetune_dir]
    if run_decompress:
        cmd.append('--decompress')
    
    print(f"\n=== Running benchmark with {threads} threads ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment: BITX_THREADS={threads}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        stdout = result.stdout
        stderr = ""
        elapsed = time.time() - start_time
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        stdout = e.stdout if e.stdout else ""
        stderr = e.stderr if e.stderr else ""
        print(f"Error output: {stderr}")
        return None
    
    results = {
        'threads': threads,
        'compression_throughput': None,
        'decompression_throughput': None,
        'compression_time': None,
        'decompression_time': None,
        'compression_ratio': None,
        'success': False
    }
    
    # Parse compression throughput - support two formats
    comp_match = re.search(r'Compression throughput: ([\d\.]+) MB/s', stdout)
    if not comp_match:
        comp_match = re.search(r'Throughput: ([\d\.]+) MB/s', stdout)
    if comp_match:
        results['compression_throughput'] = float(comp_match.group(1))
    
    # Parse compression time
    time_match = re.search(r'Compression time: ([\d\.]+) seconds', stdout)
    if time_match:
        results['compression_time'] = float(time_match.group(1))
    
    # Parse compression ratio
    ratio_match = re.search(r'Compression ratio: ([\d\.]+)', stdout)
    if ratio_match:
        results['compression_ratio'] = float(ratio_match.group(1))
    
    # Parse decompression throughput if available
    if run_decompress:
        decomp_match = re.search(r'Decompression throughput: ([\d\.]+) MB/s', stdout)
        if decomp_match:
            results['decompression_throughput'] = float(decomp_match.group(1))
        
        decomp_time_match = re.search(r'Decompression time: ([\d\.]+) seconds', stdout)
        if decomp_time_match:
            results['decompression_time'] = float(decomp_time_match.group(1))
        
        if "All tensors passed bit-exact verification" in stdout:
            results['verification'] = "PASS"
        elif "WARNING" in stdout and "failed bit-exact verification" in stdout:
            results['verification'] = "FAIL"
        else:
            results['verification'] = "UNKNOWN"
    
    if results['compression_throughput'] is not None:
        results['success'] = True
    
    print(f"Compression throughput: {results['compression_throughput']} MB/s" if results['compression_throughput'] is not None else "Compression throughput: Not found")
    if run_decompress and results['decompression_throughput'] is not None:
        print(f"Decompression throughput: {results['decompression_throughput']} MB/s")
    print(f"Compression ratio: {results['compression_ratio']}" if results['compression_ratio'] is not None else "Compression ratio: Not found")
    if run_decompress and 'verification' in results:
        print(f"Verification: {results['verification']}")
        
    if not results['success']:
        print("\nDebug - Command output first 5 lines:")
        for i, line in enumerate(stdout.split('\n')[:5]):
            print(f"  {i+1}: {line}")
        print("Debug - Command output last 5 lines:")
        for i, line in enumerate(stdout.split('\n')[-5:]):
            print(f"  {len(stdout.split('\n'))-5+i+1}: {line}")
    
    return results


def run_all_benchmarks(base_dir, finetune_dir, thread_counts, run_decompress=True, repeat=1, benchmark_path=None):
    """Run benchmarks for all specified thread counts."""
    results = []
    
    for threads in thread_counts:
        thread_results = []
        for i in range(repeat):
            print(f"\nRun {i+1}/{repeat} with {threads} threads")
            result = run_benchmark(base_dir, finetune_dir, threads, run_decompress, benchmark_path)
            if result and result['success']:
                thread_results.append(result)
            else:
                print(f"Failed to get results for {threads} threads, run {i+1}")
        
        if thread_results:
            avg_result = {
                'threads': threads,
                'compression_throughput': np.mean([r['compression_throughput'] for r in thread_results]),
                'compression_time': np.mean([r['compression_time'] for r in thread_results]),
                'compression_ratio': np.mean([r['compression_ratio'] for r in thread_results]),
            }
            
            if run_decompress:
                decomp_throughputs = [r['decompression_throughput'] for r in thread_results if r['decompression_throughput'] is not None]
                if decomp_throughputs:
                    avg_result['decompression_throughput'] = np.mean(decomp_throughputs)
                
                decomp_times = [r['decompression_time'] for r in thread_results if r['decompression_time'] is not None]
                if decomp_times:
                    avg_result['decompression_time'] = np.mean(decomp_times)
                
                verifications = [r.get('verification', 'UNKNOWN') for r in thread_results]
                if all(v == 'PASS' for v in verifications):
                    avg_result['verification'] = 'PASS'
                elif any(v == 'FAIL' for v in verifications):
                    avg_result['verification'] = 'FAIL'
                else:
                    avg_result['verification'] = 'UNKNOWN'
            
            results.append(avg_result)
    
    return results


def save_results_to_csv(results, output_prefix):
    """Save benchmark results to a CSV file."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create results directory in the current (analysis) directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    filename = results_dir / f"{output_prefix}_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['threads', 'compression_throughput', 'decompression_throughput', 
                     'compression_time', 'decompression_time', 'compression_ratio']
        
        if results and 'verification' in results[0]:
            fieldnames.append('verification')
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {field: result.get(field) for field in fieldnames}
            writer.writerow(row)
    
    print(f"\nResults saved to {filename}")
    return filename


def plot_results(results, output_prefix):
    """Create a plot of throughput vs thread count."""
    if not results:
        print("No results to plot")
        return
    
    df = pd.DataFrame(results)
    
    df = df.sort_values('threads')
    
    # Create results directory in the current (analysis) directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.figure(figsize=(10, 6))
    plt.plot(df['threads'], df['compression_throughput'], 'b-o', label='Compression')
    if 'decompression_throughput' in df.columns and not df['decompression_throughput'].isna().all():
        plt.plot(df['threads'], df['decompression_throughput'], 'r-o', label='Decompression')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('Thread Count')
    plt.ylabel('Throughput (MB/s)')
    plt.title('BitX Compression and Decompression Throughput vs Thread Count')
    plt.legend()
    plt.xticks(df['threads'], df['threads'])
    filename = results_dir / f"{output_prefix}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {filename}")
    
    
    return filename


def main():
    """Main function."""
    args = parse_args()
    
    thread_counts = [int(t) for t in args.threads.split(',')]
    
    # Use absolute path to find benchmark_bitx
    # First check relative to current script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Try several possible locations
    possible_paths = [
        project_root / 'target' / 'release' / 'examples' / 'benchmark_bitx',  
        project_root / 'target' / 'release' / 'benchmark_bitx',              
    ]
    
    benchmark_path = None
    for path in possible_paths:
        if path.exists():
            benchmark_path = path
            break
    
    if not benchmark_path:
        print("Error: benchmark_bitx not found in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("Please build the project first with 'cargo build --release'")
        return 1
    
    print(f"Using benchmark executable: {benchmark_path}")
    
    print(f"Starting BitX benchmark analysis")
    print(f"Base directory: {args.base_dir}")
    print(f"Finetune directory: {args.finetune_dir}")
    print(f"Thread counts: {thread_counts}")
    print(f"Decompression test: {'disabled' if args.no_decompress else 'enabled'}")
    print(f"Repetitions: {args.repeat}")
    
    results = run_all_benchmarks(
        args.base_dir,
        args.finetune_dir,
        thread_counts,
        not args.no_decompress,
        args.repeat,
        benchmark_path
    )
    
    if not results:
        print("No valid results obtained")
        return 1
    
    csv_file = save_results_to_csv(results, args.output)
    
    plot_file = plot_results(results, args.output)
    
    print("\nBenchmark analysis completed successfully")
    print(f"Results saved to {csv_file}")
    print(f"Plot saved to {plot_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
