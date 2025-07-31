#!/usr/bin/env python3
"""
RAYON Thread Scaling Analysis Tool
==================================

This script analyzes the decompression throughput data from benchmark txt files,
generates CSV reports, and creates visualization plots.

Requirements:
- Python 3.6+ with matplotlib, pandas
- Benchmark output files (threads_*.txt) in the parent directory
- Must be run from the analysis/ subdirectory

Features:
- Parses decompression throughput from log files
- Generates CSV reports with statistics
- Creates throughput vs thread count plots
- Shows threading efficiency analysis

Usage:
    python3 throughput_analyzer.py [--help]
    python3 throughput_analyzer.py --analyze-only
"""

import os
import subprocess
import time
from datetime import datetime
import re
import csv
import glob
import matplotlib.pyplot as plt
import pandas as pd
import sys

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

def run_analysis():
    """Run the complete throughput analysis"""
    print("🔍 Running throughput analysis on existing files...")
    results = analyze_throughput_data()
    
    if results:
        print_analysis_summary(results)
        print("\n🎉 Analysis completed successfully!")
        print("📊 Check the generated CSV and PNG files for detailed results.")
        
        # Print CSV data summary
        print("\n📋 CSV Summary:")
        print("threads,avg_throughput,min_throughput,max_throughput,num_records")
        for r in results:
            print(f"{r['threads']},{r['avg_throughput']:.3f},{r['min_throughput']:.3f},{r['max_throughput']:.3f},{r['num_records']}")
    else:
        print("❌ No data found to analyze!")
        print("💡 Make sure threads_*.txt files exist in the parent directory")

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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RAYON Thread Scaling Analysis Tool')
    parser.add_argument('--analyze-only', action='store_true', 
                       help='Run analysis on existing txt files (default behavior)')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies before running the analysis.")
        exit(1)
    
    # Check if we're in the right directory
    if not os.path.exists("../threads_1.txt"):
        print("❌ Could not find threads_*.txt files in parent directory")
        print("💡 Make sure to run this script from the analysis/ subdirectory")
        print("💡 And ensure benchmark files exist in the project root")
        exit(1)
    
    # Run analysis
    run_analysis() 