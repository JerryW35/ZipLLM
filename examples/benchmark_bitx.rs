use std::env;
use std::path::Path;
use std::collections::HashMap;
use std::time::Instant;
use std::io::{Write, Read};
use log::info;
use zipllm::config::set_config;
use zipllm::bitx::bitx_bytes::{bitx_transform_only, optimized_bitx_compress};
use zstd::stream::{Encoder, Decoder};
use rayon::prelude::*;
use anyhow::Result;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub fn zstd_compress_data(data: &[u8], level: i32) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() / 3);
    let mut encoder = Encoder::new(&mut output, level).expect("Zstd encoder failed");
    let threads = env::var("BITX_THREADS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1)
        });
    encoder.multithread(threads).expect("Failed to set threads");
    encoder.write_all(data).expect("Write failed");
    encoder.finish().expect("Finish failed");
    output
}

pub fn zstd_decompress_data(data: &[u8]) -> Vec<u8> {
    let mut decoder = Decoder::new(data).expect("Failed to create decoder");
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).expect("Failed to decompress");
    decompressed
}

fn zstd_decompress_all_frames(input: &[u8]) -> Vec<u8> {
    if input.len() >= 8 && &input[0..4] == b"ZIDX" {
        let num = u32::from_le_bytes([input[4], input[5], input[6], input[7]]) as usize;
        let lens_start = 8;
        let lens_end = lens_start + num * 8;
        if input.len() >= lens_end {
            let mut clens = Vec::with_capacity(num);
            let mut dlens = Vec::with_capacity(num);
            for i in 0..num {
                let o = lens_start + i * 8;
                clens.push(u32::from_le_bytes([input[o], input[o+1], input[o+2], input[o+3]]) as usize);
                dlens.push(u32::from_le_bytes([input[o+4], input[o+5], input[o+6], input[o+7]]) as usize);
            }
            let mut coffs = Vec::with_capacity(num);
            let mut off = lens_end;
            let mut ok = true;
            let total: usize = dlens.iter().sum();
            for &l in &clens {
                if off + l > input.len() {
                    ok = false;
                    break;
                }
                coffs.push(off);
                off += l;
            }
            if ok {
                let mut doffs = Vec::with_capacity(num);
                let mut acc: usize = 0;
                for &d in &dlens {
                    doffs.push(acc);
                    acc += d;
                }
                let mut out = vec![0u8; total];
                struct OutPtr(*mut u8);
                unsafe impl Send for OutPtr {}
                unsafe impl Sync for OutPtr {}
                let out_ptr = std::sync::Arc::new(OutPtr(out.as_mut_ptr()));
                (0..num).into_par_iter().for_each(|i| {
                    let start = coffs[i];
                    let end = start + clens[i];
                    let dlen = dlens[i];
                    let doff = doffs[i];
                    let src = &input[start..end];
                    let base_ptr = (*out_ptr).0;
                    let dst = unsafe { std::slice::from_raw_parts_mut(base_ptr.add(doff), dlen) };
                    let mut dctx = zstd_safe::DCtx::create();
                    let written = dctx.decompress(dst, src).expect("zstd decompress");
                    debug_assert_eq!(written, dlen);
                });
                return out;
            }
        }
    }
    zstd_decompress_data(input)
}

fn reconstruct_tensor_from_bitx(base: &[u8], exp_frames: &[u8], sm_frames: &[u8]) -> Vec<u8> {
    let exp = zstd_decompress_all_frames(exp_frames);
    let sm = zstd_decompress_all_frames(sm_frames);
    assert_eq!(exp.len(), sm.len(), "Exponent and sign-mantissa lengths must match");
    let pair_count = exp.len();
    assert!(base.len() >= pair_count * 2, "Base tensor too small for reconstruction");
    let mut output = vec![0u8; pair_count * 2];
    output.par_chunks_mut(2).enumerate().for_each(|(i, chunk)| {
        let sign = ((sm[i] >> 7) & 0x1) as u16;
        let mantissa = (sm[i] & 0x7F) as u16;
        let expv = exp[i] as u16;
        let xor = (sign << 15) | (expv << 7) | mantissa;
        let idx = i * 2;
        let base_v = u16::from_le_bytes([base[idx], base[idx + 1]]);
        let rec = base_v ^ xor;
        chunk[0] = (rec & 0x00FF) as u8;
        chunk[1] = (rec >> 8) as u8;
    });
    output
}

fn tensors_equal(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a == b
}

#[derive(Debug)]
pub struct BitxBenchmarkResult {
    pub total_tensors: usize,
    pub processed_tensors: usize,
    pub skipped_tensors: usize,
    pub total_original_size_mb: f64,
    pub total_compressed_size_mb: f64,
    pub compression_ratio: f64,
    pub compression_time_seconds: f64,
    pub throughput_mb_per_sec: f64,
    pub decompression_time_seconds: Option<f64>,
    pub decompression_throughput_mb_per_sec: Option<f64>,
}

pub fn optimized_bitx_compress_chunked(data1: &[u8], data2: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    let min_len = std::cmp::min(data1.len(), data2.len()) & !1;
    if min_len < 2 {
        return Err(anyhow::anyhow!("Input data too small for bitx compression"));
    }

    // Chunk by bytes (default 16MB), even alignment
    let chunk_bytes_env = std::env::var("BITX_TENSOR_CHUNK_SIZE")
        .ok().and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16 * 1024 * 1024);
    let chunk_bytes = std::cmp::max(2, chunk_bytes_env & !1usize);

    if min_len <= chunk_bytes {
        return optimized_bitx_compress(data1, data2);
    }

    let total_chunks = (min_len + chunk_bytes - 1) / chunk_bytes;

    // Parallel: transform -> record uncompressed length -> compress -> collect
    // parts: (idx, exp_c, exp_u, sm_c, sm_u)
    let mut parts: Vec<(usize, Vec<u8>, usize, Vec<u8>, usize)> =
        (0..total_chunks).into_par_iter().map(|idx| {
            let start = idx * chunk_bytes;
            let end = std::cmp::min(start + chunk_bytes, min_len);

            let (exp, sm) = bitx_transform_only(&data1[start..end], &data2[start..end])
                .unwrap_or_else(|_| (Vec::new(), Vec::new()));
            let exp_u = exp.len();
            let sm_u  = sm.len();

            let zstd_level = 3;
            let (exp_c, sm_c) = rayon::join(
                || zstd::bulk::compress(&exp, zstd_level)
                        .unwrap_or_else(|_| zstd_compress_data(&exp, zstd_level)),
                || zstd::bulk::compress(&sm,  zstd_level)
                        .unwrap_or_else(|_| zstd_compress_data(&sm,  zstd_level)),
            );
            (idx, exp_c, exp_u, sm_c, sm_u)
        }).collect();

    parts.sort_by_key(|p| p.0);
    let num = parts.len() as u32;

    // EXP container
    let exp_payload: usize = parts.iter().map(|p| p.1.len()).sum();
    let exp_header_size = 4 + 4 + (num as usize) * 8;
    let mut out_exp = Vec::with_capacity(exp_header_size + exp_payload);
    out_exp.extend_from_slice(b"ZIDX");
    out_exp.extend_from_slice(&num.to_le_bytes());
    for p in &parts {
        out_exp.extend_from_slice(&(p.1.len() as u32).to_le_bytes()); // clen
        out_exp.extend_from_slice(&(p.2 as u32).to_le_bytes());       // dlen
    }
    for p in &parts { out_exp.extend_from_slice(&p.1); }

    // SM container
    let sm_payload: usize = parts.iter().map(|p| p.3.len()).sum();
    let sm_header_size = 4 + 4 + (num as usize) * 8;
    let mut out_sm = Vec::with_capacity(sm_header_size + sm_payload);
    out_sm.extend_from_slice(b"ZIDX");
    out_sm.extend_from_slice(&num.to_le_bytes());
    for p in &parts {
        out_sm.extend_from_slice(&(p.3.len() as u32).to_le_bytes()); // clen
        out_sm.extend_from_slice(&(p.4 as u32).to_le_bytes());       // dlen
    }
    for p in &parts { out_sm.extend_from_slice(&p.3); }

    Ok((out_exp, out_sm))
}

/// Optimized BitX compression benchmark
pub fn benchmark_optimized_bitx_compression(base_dir: &str, finetune_dir: &str, run_decompress: bool) -> Result<BitxBenchmarkResult> {
    info!("Starting optimized BitX compression benchmark (chunked big tensors)\nBase model: {}\nFinetune model: {}", base_dir, finetune_dir);

    let base_path = Path::new(base_dir);
    let finetune_path = Path::new(finetune_dir);

    // Load tensors from base and finetune directories
    let mut base_tensors = HashMap::new();
    for entry in std::fs::read_dir(base_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            let tensors = zipllm::tensor_loader::read_safetensor(&path)?;
            for tensor in tensors {
                base_tensors.insert(tensor.name.clone(), tensor);
            }
        }
    }

    let mut finetune_tensors = HashMap::new();
    for entry in std::fs::read_dir(finetune_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            let tensors = zipllm::tensor_loader::read_safetensor(&path)?;
            for tensor in tensors {
                finetune_tensors.insert(tensor.name.clone(), tensor);
            }
        }
    }

    // Benchmark compression
    let start_time = Instant::now();
    let results: Vec<(usize, usize, bool)> = finetune_tensors.par_iter().filter_map(|(name, finetune_tensor)| {
        if let Some(base_tensor) = base_tensors.get(name) {
            if base_tensor.shape == finetune_tensor.shape && base_tensor.dtype == finetune_tensor.dtype {
                let orig = finetune_tensor.data.as_ref().map_or(0, |d| d.len());
                let base_data = base_tensor.data.as_ref().unwrap();
                let finetune_data = finetune_tensor.data.as_ref().unwrap();
                
                match optimized_bitx_compress_chunked(base_data, finetune_data) {
                    Ok((ce, cs)) => Some((orig, ce.len() + cs.len(), true)),
                    Err(_) => Some((0, 0, false)),
                }
            } else { Some((0, 0, false)) }
        } else { Some((0, 0, false)) }
    }).collect();

    // Collect compression results
    let (total_orig_size, total_comp_size, processed, skipped) = results.into_iter()
        .fold((0usize, 0usize, 0usize, 0usize), |acc, (o, c, p)| { 
            let (ao, ac, ap, as_) = acc; 
            if p { 
                (ao+o, ac+c, ap+1, as_) 
            } else { 
                (ao, ac, ap, as_+1) 
            } 
        });

    let elapsed_seconds = start_time.elapsed().as_secs_f64();
    let total_original_size_mb = total_orig_size as f64 / (1024.0 * 1024.0);
    let total_compressed_size_mb = total_comp_size as f64 / (1024.0 * 1024.0);
    let compression_ratio = if total_orig_size > 0 { total_comp_size as f64 / total_orig_size as f64 } else { 1.0 };
    let throughput = total_original_size_mb / elapsed_seconds;

    info!("Optimized BitX compression completed in {:.2} seconds", elapsed_seconds);
    info!("Throughput: {:.2} MB/s", throughput);
    info!("Processed tensors: {}, Skipped tensors: {}", processed, skipped);
    info!("Original size: {:.2} MB, Compressed size: {:.2} MB", total_original_size_mb, total_compressed_size_mb);
    info!("Compression ratio: {:.4} ({:.2}% of original)", compression_ratio, compression_ratio * 100.0);
    
    // Run decompression test if requested
    let (decomp_seconds, decomp_throughput) = if run_decompress {
        info!("Running decompression test...");
        
        // For decompression test, we need to compress the tensors again to get the compressed data
        let compressed_data: Vec<(String, Vec<u8>, Vec<u8>)> = finetune_tensors
            .par_iter()
            .filter_map(|(name, finetune_tensor)| {
                if let Some(base_tensor) = base_tensors.get(name) {
                    if base_tensor.shape == finetune_tensor.shape && base_tensor.dtype == finetune_tensor.dtype {
                        let base_data = base_tensor.data.as_ref().unwrap();
                        let finetune_data = finetune_tensor.data.as_ref().unwrap();
                        
                        match optimized_bitx_compress_chunked(base_data, finetune_data) {
                            Ok((ce, cs)) => Some((name.clone(), ce, cs)),
                            Err(_) => None,
                        }
                    } else { None }
                } else { None }
            })
            .collect();
        
        // Prepare data structures for decompression and verification
        let mut comp_map: HashMap<String, (Vec<u8>, Vec<u8>)> = HashMap::new();
        let decomp_results: HashMap<String, Vec<u8>> = HashMap::new();
        
        // Build lookup map for compressed data
        for (n, ce, cs) in compressed_data {
            comp_map.insert(n, (ce, cs));
        }
        
        // Decompression benchmark - only decompress, do not verify yet
        let start_decomp = Instant::now();
        
        // Use a thread-safe collector for decompression results
        let decomp_results_mutex = Arc::new(std::sync::Mutex::new(decomp_results));
        
        // Parallel reconstruction of tensors, no verification yet
        finetune_tensors.par_iter().for_each(|(name, finetune_tensor)| {
            if let (Some(base_tensor), Some((ce, cs))) = (base_tensors.get(name), comp_map.get(name)) {
                if base_tensor.shape == finetune_tensor.shape && base_tensor.dtype == finetune_tensor.dtype {
                    let base_data = base_tensor.data.as_ref().unwrap();
                    
                    // Reconstruct tensor from compressed data
                    let recon = reconstruct_tensor_from_bitx(base_data, ce, cs);
                    
                    // Store reconstruction result, verification is done later
                    let name_clone = name.clone();
                    decomp_results_mutex.lock().unwrap().insert(name_clone, recon);
                }
            }
        });
        
        // Record decompression time and throughput
        let decomp_seconds = start_decomp.elapsed().as_secs_f64();
        let decomp_throughput = total_original_size_mb / decomp_seconds;
        
        // Unlock the result collector
        let decomp_results = Arc::try_unwrap(decomp_results_mutex)
            .expect("Failed to unwrap mutex")
            .into_inner()
            .expect("Failed to get inner value");
        
        // After decompression, verify results separately
        let start_verify = Instant::now();
        
        // Atomic counters for verification
        let success_count = Arc::new(AtomicUsize::new(0));
        let failure_count = Arc::new(AtomicUsize::new(0));
        let total_count = Arc::new(AtomicUsize::new(0));
        
        // Verify decompression results
        finetune_tensors.par_iter().for_each(|(name, finetune_tensor)| {
            if let Some(recon) = decomp_results.get(name) {
                let finetune_data = finetune_tensor.data.as_ref().unwrap();
                
                // Count total
                total_count.fetch_add(1, Ordering::Relaxed);
                
                // Verify reconstruction - must be byte-exact
                if recon.len() == finetune_data.len() {
                    let mut is_equal = true;
                    for i in 0..recon.len() {
                        if recon[i] != finetune_data[i] {
                            is_equal = false;
                            break;
                        }
                    }
                    
                    if is_equal {
                        success_count.fetch_add(1, Ordering::Relaxed);
                    } else {
                        failure_count.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    failure_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        
        let verify_seconds = start_verify.elapsed().as_secs_f64();
        
        // Output verification results
        let success = success_count.load(Ordering::Relaxed);
        let failure = failure_count.load(Ordering::Relaxed);
        let total = total_count.load(Ordering::Relaxed);
        
        info!("Decompression verification: {} tensors processed", total);
        info!("Successful: {}, Failed: {}", success, failure);
        info!("Verification time: {:.2} seconds", verify_seconds);
        
        if failure > 0 {
            info!("WARNING: {} tensors failed bit-exact verification!", failure);
        } else {
            info!("All tensors passed bit-exact verification");
        }
        
        info!("Decompression completed in {:.2} seconds", decomp_seconds);
        info!("Decompression throughput: {:.2} MB/s", decomp_throughput);
        
        (Some(decomp_seconds), Some(decomp_throughput))
    } else {
        (None, None)
    };

    Ok(BitxBenchmarkResult {
        total_tensors: finetune_tensors.len(),
        processed_tensors: processed,
        skipped_tensors: skipped,
        total_original_size_mb,
        total_compressed_size_mb,
        compression_ratio,
        compression_time_seconds: elapsed_seconds,
        throughput_mb_per_sec: throughput,
        decompression_time_seconds: decomp_seconds,
        decompression_throughput_mb_per_sec: decomp_throughput,
    })
}

fn main() -> Result<()> {
    // If RUST_LOG is not set, default to info level
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // Print a message directly to the console to ensure the user sees it
    info!("Starting optimized BitX compression benchmark...");
    
    // Read thread count from environment variable, or use system default
    let threads = env::var("BITX_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| {
            // Use system available threads by default
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
    
    // Set global thread pool size for rayon
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .expect("Failed to set global thread pool");
    
    
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("Usage: {} <base_dir> <finetune_dir> [--decompress]", args[0]);
        eprintln!("  --decompress: Optional flag to run decompression test");
        std::process::exit(1);
    }
    
    let base_dir = &args[1];
    let finetune_dir = &args[2];
    
    // Check if decompression test is requested
    let run_decompress = args.len() == 4 && args[3] == "--decompress";
    
    info!("Starting optimized BitX compression benchmark");
    info!("Base directory: {}", base_dir);
    info!("Finetune directory: {}", finetune_dir);
    
    let result = benchmark_optimized_bitx_compression(base_dir, finetune_dir, run_decompress)?;
    
    info!("Benchmark results:");
    info!("Total tensors: {}", result.total_tensors);
    info!("Processed tensors: {}", result.processed_tensors);
    info!("Skipped tensors: {}", result.skipped_tensors);
    info!("Original size: {:.2} MB", result.total_original_size_mb);
    info!("Compressed size: {:.2} MB", result.total_compressed_size_mb);
    info!("Compression ratio: {:.4} ({:.2}%)", 
          result.compression_ratio, 
          result.compression_ratio * 100.0);
    info!("Compression time: {:.2} seconds", result.compression_time_seconds);
    info!("Using {} threads for compression", threads);
    info!("Compression throughput: {:.2} MB/s", result.throughput_mb_per_sec);
    
    // Print decompression results if available
    if let (Some(decomp_time), Some(decomp_throughput)) = (result.decompression_time_seconds, result.decompression_throughput_mb_per_sec) {
        info!("Decompression time: {:.2} seconds", decomp_time);
        info!("Decompression throughput: {:.2} MB/s", decomp_throughput);
    }
    
    Ok(())
}
