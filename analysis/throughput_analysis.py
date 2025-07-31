#!/usr/bin/env python3
"""
ZipLLM Restore Performance Analysis Tool
========================================

This script runs restore experiments with different thread counts,
clears cache before each run, and analyzes the throughput results.

Requirements:
- Rust project built with --release
- sudo access for cache clearing
- Python 3.6+ with matplotlib, pandas
- Must be run from the analysis/ subdirectory

Features:
- Runs single restore experiment per thread count
- Clears system cache before each experiment
- Saves outputs to threads_*.txt files
- Parses decompression throughput from log files
- Generates CSV reports with statistics
- Creates throughput vs thread count plots
- Shows threading efficiency analysis

Usage:
    python3 throughput_analysis.py --threads 1 2 4 8 16 32 48
    python3 throughput_analysis.py --analyze-only
    python3 throughput_analysis.py --threads 1 4 8 --model "bambisheng_UltraIF-8B-SFT" --output "/tmp/zipllm_test"
"""

import os
import subprocess
import time
import argparse
import sys
from datetime import datetime
import re
import csv
import glob
import matplotlib.pyplot as plt
import pandas as pd

def clear_system_cache():
    """Clear system cache using sudo"""
    print("🧹 Clearing system cache...")
    try:
        result = subprocess.run([
            'sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'
        ], check=True, capture_output=True, text=True)
        print("✅ Cache cleared successfully")
        time.sleep(2)  # Wait a bit after cache clear
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to clear cache: {e}")
        print("💡 Make sure you have sudo privileges")
        return False
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return False

def run_restore_experiment(threads, model_id, output_path):
    """Run restore experiment with specified thread count"""
    print(f"\n🚀 Running restore experiment with {threads} threads...")
    print(f"📦 Model: {model_id}")
    print(f"📁 Output: {output_path}")
    
    # Clear cache before experiment
    if not clear_system_cache():
        print("⚠️  Continuing without cache clear...")
    
    # Prepare environment
    env = os.environ.copy()
    env['RUST_LOG'] = 'info'
    env['RAYON_NUM_THREADS'] = str(threads)
    
    # Prepare command
    restore_cmd = [
        '../target/release/restore',
        model_id,
        output_path
    ]
    
    # Output file
    output_file = f"../threads_{threads}.txt"
    
    try:
        # Run the restore command
        start_time = time.time()
        result = subprocess.run(
            restore_cmd,
            env=env,
            cwd='.',  # Run from analysis directory
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        end_time = time.time()
        
        if result.returncode == 0:
            duration = end_time - start_time
            print(f"✅ Experiment completed successfully in {duration:.2f}s")
            
            # Collect output
            output_text = f"=== Experiment with {threads} threads ===\n"
            output_text += f"Timestamp: {datetime.now().isoformat()}\n"
            output_text += f"Command: {' '.join(restore_cmd)}\n"
            output_text += f"Environment: RAYON_NUM_THREADS={threads}\n"
            output_text += f"Duration: {duration:.2f}s\n"
            output_text += "=== STDOUT ===\n"
            output_text += result.stdout
            output_text += "\n=== STDERR ===\n"
            output_text += result.stderr
            output_text += "\n" + "="*80 + "\n\n"
            
            # Clean up output directory
            if os.path.exists(output_path):
                subprocess.run(['rm', '-rf', output_path], check=False)
                
        else:
            print(f"❌ Experiment failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            
            # Still save the failed output for debugging
            output_text = f"=== FAILED Experiment with {threads} threads ===\n"
            output_text += f"Timestamp: {datetime.now().isoformat()}\n"
            output_text += f"Return code: {result.returncode}\n"
            output_text += f"STDERR: {result.stderr}\n"
            output_text += f"STDOUT: {result.stdout}\n"
            output_text += "\n" + "="*80 + "\n\n"
    
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment timed out after 1 hour")
        output_text = f"=== TIMEOUT Experiment with {threads} threads ===\n"
        output_text += f"Timestamp: {datetime.now().isoformat()}\n"
        output_text += f"Timeout: 1 hour\n"
        output_text += "\n" + "="*80 + "\n\n"
        result = None
    except Exception as e:
        print(f"❌ Experiment failed with exception: {e}")
        output_text = f"=== ERROR Experiment with {threads} threads ===\n"
        output_text += f"Timestamp: {datetime.now().isoformat()}\n"
        output_text += f"Error: {e}\n"
        output_text += "\n" + "="*80 + "\n\n"
        result = None
    
    # Save output to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# ZipLLM Restore Experiment Results\n")
            f.write(f"# Model: {model_id}\n")
            f.write(f"# Threads: {threads}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Command: {' '.join(restore_cmd)}\n\n")
            f.write(output_text)
        
        print(f"📄 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return False
    
    return result is not None and result.returncode == 0

def run_experiments(thread_counts, model_id, output_path):
    """Run experiments for all specified thread counts"""
    print("🔬 Starting ZipLLM Restore Performance Experiments")
    print("="*60)
    print(f"📊 Thread counts to test: {thread_counts}")
    print(f"📦 Model: {model_id}")
    print(f"📁 Output path: {output_path}")
    print("="*60)
    
    # Check if restore binary exists
    restore_binary = "../target/release/restore"
    if not os.path.exists(restore_binary):
        print(f"❌ Restore binary not found: {restore_binary}")
        print("💡 Run 'cargo build --release' from the project root")
        return False
    
    successful_experiments = 0
    total_experiments = len(thread_counts)
    
    start_time = time.time()
    
    for i, threads in enumerate(thread_counts):
        print(f"\n🧪 Experiment {i+1}/{total_experiments}: Testing {threads} threads")
        
        if run_restore_experiment(threads, model_id, output_path):
            successful_experiments += 1
        else:
            print(f"❌ Experiment with {threads} threads failed!")
        
        # Show progress
        progress = (i + 1) / total_experiments * 100
        print(f"📈 Progress: {progress:.1f}% ({i+1}/{total_experiments})")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\n" + "="*60)
    print("🎯 EXPERIMENT SUMMARY")
    print("="*60)
    print(f"✅ Successful experiments: {successful_experiments}/{total_experiments}")
    print(f"⏱️  Total duration: {total_duration/60:.1f} minutes")
    print(f"📄 Output files: threads_*.txt in parent directory")
    
    if successful_experiments > 0:
        print("🎉 Experiments completed! Ready for analysis.")
        return True
    else:
        print("❌ All experiments failed!")
        return False

def parse_decompression_throughput(file_path):
    """Parse decompression throughput data from txt files"""
    throughputs = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Match pattern: Decompression: X.XXXs (Y.YY GB/s FT throughput)
        pattern = r'Decompression:\s+[\d.]+s\s+\((\d+\.?\d*)\s+GB/s\s+FT\s+throughput\)'
        matches = re.findall(pattern, content)
        
        # Convert to float
        throughputs = [float(match) for match in matches]
        
        print(f"📊 Found {len(throughputs)} decompression records in {file_path}")
        print(f"    Throughputs: {throughputs} GB/s")
        
        return throughputs
        
    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        return []

def extract_thread_count_from_filename(filename):
    """Extract thread count from filename"""
    # Match pattern: threads_X.txt
    match = re.search(r'threads_(\d+)\.txt', filename)
    if match:
        return int(match.group(1))
    return None

def analyze_throughput_data():
    """Analyze throughput data from all txt files"""
    print("\n🔍 Analyzing throughput data from txt files...")
    
    # Look for all threads_*.txt files in parent directory
    txt_files = glob.glob("../threads_*.txt")
    
    if not txt_files:
        print("❌ No threads_*.txt files found in parent directory!")
        print("💡 Make sure to run this script from the analysis/ subdirectory")
        return
    
    print(f"📄 Found {len(txt_files)} files: {txt_files}")
    
    results = []
    
    for file_path in txt_files:
        # Extract just the filename for thread count extraction
        filename = os.path.basename(file_path)
        thread_count = extract_thread_count_from_filename(filename)
        if thread_count is None:
            print(f"⚠️  Could not extract thread count from {filename}")
            continue
            
        throughputs = parse_decompression_throughput(file_path)
        
        if throughputs:
            # Calculate statistics
            avg_throughput = sum(throughputs) / len(throughputs)
            min_throughput = min(throughputs)
            max_throughput = max(throughputs)
            
            results.append({
                'threads': thread_count,
                'avg_throughput': avg_throughput,
                'min_throughput': min_throughput,
                'max_throughput': max_throughput,
                'num_records': len(throughputs),
                'all_throughputs': throughputs
            })
            
            print(f"✅ Thread {thread_count}: Avg={avg_throughput:.3f} GB/s, Min={min_throughput:.3f}, Max={max_throughput:.3f}")
        else:
            print(f"⚠️  No throughput data found in {file_path}")
    
    if not results:
        print("❌ No valid throughput data found!")
        return
    
    # Sort by thread count
    results.sort(key=lambda x: x['threads'])
    
    # Generate CSV file
    csv_filename = "decompression_throughput_analysis.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['threads', 'avg_throughput', 'min_throughput', 'max_throughput', 'num_records']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'threads': result['threads'],
                'avg_throughput': f"{result['avg_throughput']:.3f}",
                'min_throughput': f"{result['min_throughput']:.3f}",
                'max_throughput': f"{result['max_throughput']:.3f}",
                'num_records': result['num_records']
            })
    
    print(f"📊 CSV data saved to: {csv_filename}")
    
    # Create plots
    create_throughput_plot(results)
    
    return results

def create_throughput_plot(results):
    """Create throughput vs threads visualization"""
    print("📈 Creating throughput visualization...")
    
    # Extract data
    threads = [r['threads'] for r in results]
    avg_throughputs = [r['avg_throughput'] for r in results]
    min_throughputs = [r['min_throughput'] for r in results]
    max_throughputs = [r['max_throughput'] for r in results]
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    # Main plot: average throughput
    plt.subplot(2, 1, 1)
    plt.plot(threads, avg_throughputs, 'o-', linewidth=2, markersize=8, label='Average Throughput')
    plt.fill_between(threads, min_throughputs, max_throughputs, alpha=0.3, label='Min-Max Range')
    plt.xlabel('Number of Threads')
    plt.ylabel('Decompression Throughput (GB/s)')
    plt.title('Decompression Throughput vs Thread Count')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add value labels
    for i, (x, y) in enumerate(zip(threads, avg_throughputs)):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    
    # Subplot: efficiency analysis (relative to single thread)
    plt.subplot(2, 1, 2)
    baseline_throughput = None
    for r in results:
        if r['threads'] == 1:
            baseline_throughput = r['avg_throughput']
            break
    
    if baseline_throughput:
        efficiency = [r['avg_throughput'] / baseline_throughput for r in results]
        ideal_efficiency = [t / 1 for t in threads]  # Ideal linear scaling
        
        plt.plot(threads, efficiency, 'o-', linewidth=2, markersize=8, label='Actual Efficiency', color='green')
        plt.plot(threads, ideal_efficiency, '--', linewidth=1, alpha=0.7, label='Ideal Linear Scaling', color='red')
        plt.xlabel('Number of Threads')
        plt.ylabel('Speedup Ratio (vs 1 thread)')
        plt.title('Threading Efficiency Analysis')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add value labels
        for i, (x, y) in enumerate(zip(threads, efficiency)):
            plt.annotate(f'{y:.2f}x', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = "decompression_throughput_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📈 Plot saved to: {plot_filename}")
    
    # Display plot
    plt.show()

def print_analysis_summary(results):
    """Print analysis summary"""
    print("\n" + "="*60)
    print("📊 THROUGHPUT ANALYSIS SUMMARY")
    print("="*60)
    
    best_avg = max(results, key=lambda x: x['avg_throughput'])
    worst_avg = min(results, key=lambda x: x['avg_throughput'])
    
    print(f"🏆 Best average throughput: {best_avg['avg_throughput']:.3f} GB/s ({best_avg['threads']} threads)")
    print(f"🐌 Worst average throughput: {worst_avg['avg_throughput']:.3f} GB/s ({worst_avg['threads']} threads)")
    
    # Find single thread baseline
    single_thread = None
    for r in results:
        if r['threads'] == 1:
            single_thread = r
            break
    
    if single_thread:
        speedup = best_avg['avg_throughput'] / single_thread['avg_throughput']
        print(f"📈 Best speedup vs 1 thread: {speedup:.2f}x")
        print(f"💡 Threading efficiency: {speedup / best_avg['threads'] * 100:.1f}%")
    
    print(f"📄 Data points: {len(results)} thread configurations")
    total_records = sum(r['num_records'] for r in results)
    print(f"📊 Total decompression records analyzed: {total_records}")

def check_dependencies():
    """Check if required Python packages are available"""
    missing_packages = []
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        missing_packages.append('matplotlib')
    
    try:
        import pandas as pd
    except ImportError:
        missing_packages.append('pandas')
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip3 install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='ZipLLM Restore Performance Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments with different thread counts
  python3 throughput_analysis.py --threads 1 2 4 8 16 32 48
  
  # Run experiments with custom model and output
  python3 throughput_analysis.py --threads 1 4 8 --model "your_model_id" --output "/your/output/path"
  
  # Only analyze existing results
  python3 throughput_analysis.py --analyze-only
  
  # Run experiments and then analyze
  python3 throughput_analysis.py --threads 1 4 8 16 --analyze
        """
    )
    
    parser.add_argument('--threads', type=int, nargs='+', 
                       help='Thread counts to test (e.g., --threads 1 2 4 8 16 32 48)')
    parser.add_argument('--model', type=str, default='bambisheng_UltraIF-8B-SFT',
                       help='Model ID to restore (default: bambisheng_UltraIF-8B-SFT)')
    parser.add_argument('--output', type=str, default='/tmp/zipllm_test',
                       help='Output path for restored model (default: /tmp/zipllm_test)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze existing results, do not run experiments')
    parser.add_argument('--analyze', action='store_true',
                       help='Run analysis after experiments')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies: pip3 install matplotlib pandas")
        return 1
    
    # Check if we're in the right directory
    if not os.path.basename(os.getcwd()) == 'analysis':
        print("❌ This script must be run from the analysis/ subdirectory")
        print(f"💡 Current directory: {os.getcwd()}")
        print("💡 Please cd to the analysis/ directory and run again")
        return 1
    
    if args.analyze_only:
        # Only run analysis
        print("📊 Running analysis on existing files...")
        results = analyze_throughput_data()
        if results:
            print_analysis_summary(results)
            print("🎉 Analysis completed successfully!")
        else:
            print("❌ No data found to analyze!")
            return 1
    
    elif args.threads:
        # Run experiments
        print(f"🧪 Running experiments with threads: {args.threads}")
        
        success = run_experiments(
            thread_counts=args.threads,
            model_id=args.model,
            output_path=args.output
        )
        
        if not success:
            print("❌ Experiments failed!")
            return 1
        
        # Run analysis if requested
        if args.analyze:
            print("\n📊 Running analysis...")
            results = analyze_throughput_data()
            if results:
                print_analysis_summary(results)
                print("🎉 Experiments and analysis completed successfully!")
            else:
                print("⚠️  Experiments completed but analysis failed!")
    
    else:
        print("❌ Please specify either --threads for experiments or --analyze-only for analysis")
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 