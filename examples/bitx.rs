use clap::Parser;
use std::fs;
use std::io::{Write};
use std::time::Instant;
use rayon::prelude::*;
use zstd::stream::Encoder;

/// XOR + optional ZSTD compress on two input files
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// First input file (positional)
    file1: String,

    /// Second input file (positional)
    file2: String,

    /// Output file for raw XOR result (optional)
    #[arg(long)]
    xor_output: Option<String>,

    /// Whether to compress the result using zstd
    #[arg(long, default_value_t = false)]
    compress: bool,

    /// Output file for compressed exponent bytes
    #[arg(long)]
    compressed_exp: Option<String>,

    /// Output file for compressed sign_mantissa bytes
    #[arg(long)]
    compressed_sm: Option<String>,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let read_start = Instant::now();
    let data1 = fs::read(&args.file1)?;
    let data2 = fs::read(&args.file2)?;
    let min_len = std::cmp::min(data1.len(), data2.len()) & !1;
    if min_len < 2 {
        eprintln!("Files too small");
        std::process::exit(1);
    }
    let chunk_count = min_len / 2;
    let read_time = read_start.elapsed().as_secs_f64();

    // ==== XOR Phase ====
    let xor_start = Instant::now();
    let results: Vec<(u8, u8)> = (0..chunk_count).into_par_iter().map(|i| {
        let idx = i * 2;
        let v1 = u16::from_le_bytes([data1[idx], data1[idx + 1]]);
        let v2 = u16::from_le_bytes([data2[idx], data2[idx + 1]]);
        let xor = v1 ^ v2;

        let sign = ((xor >> 15) & 0x1) as u8;
        let exponent = ((xor >> 7) & 0xFF) as u8;
        let mantissa = (xor & 0x7F) as u8;
        let sign_mantissa = (sign << 7) | mantissa;

        (exponent, sign_mantissa)
    }).collect();
    let xor_time = xor_start.elapsed().as_secs_f64();

    // unzip
    let unzip_start = Instant::now();
    let mut exp_output = vec![0u8; chunk_count];
    let mut sm_output = vec![0u8; chunk_count];
    let chunk_size = 4096;
    exp_output
        .par_chunks_mut(chunk_size)
        .zip(sm_output.par_chunks_mut(chunk_size))
        .zip(results.par_chunks(chunk_size))
        .for_each(|((exp_chunk, sm_chunk), chunk_data)| {
            for ((e, s), (exp, sm)) in exp_chunk.iter_mut().zip(sm_chunk.iter_mut()).zip(chunk_data.iter()) {
                *e = *exp;
                *s = *sm;
            }
        });
    let unzip_time = unzip_start.elapsed().as_secs_f64();

    if let Some(ref xor_path) = args.xor_output {
        let mut file = fs::File::create(xor_path)?;
        file.write_all(&exp_output)?;
        file.write_all(&sm_output)?;
        println!("✅ XOR raw output written to {}", xor_path);
    }

    let (compressed_exp, compressed_sm, zstd_time) = if args.compress {
        let zstd_start = Instant::now();
        let (c1, c2): (Vec<u8>, Vec<u8>) = rayon::join(
            || compress_data(&exp_output, 3, 48),
            || compress_data(&sm_output, 3, 48),
        );
        let zstd_time = zstd_start.elapsed().as_secs_f64();
        (Some(c1), Some(c2), zstd_time)
    } else {
        (None, None, 0.0)
    };

    if let Some(ref path) = args.compressed_exp {
        if let Some(ref data) = compressed_exp {
            fs::write(path, data)?;
            println!("✅ Compressed exponent written to {}", path);
        }
    }
    if let Some(ref path) = args.compressed_sm {
        if let Some(ref data) = compressed_sm {
            fs::write(path, data)?;
            println!("✅ Compressed sign_mantissa written to {}", path);
        }
    }

    let total_mb = min_len as f64 / (1024.0 * 1024.0);
    println!("Read time: {:.4} s ({:.2} MB/s)", read_time, total_mb / read_time);
    println!("XOR time: {:.4} s ({:.2} MB/s)", xor_time, total_mb / xor_time);
    println!("Unzip time: {:.4} s", unzip_time);
    if args.compress {
        println!("ZSTD time: {:.4} s ({:.2} MB/s)", zstd_time, total_mb / zstd_time);
    }
    println!(
        "Total time: {:.4} s",
        read_time + xor_time + unzip_time + zstd_time
    );

    Ok(())
}

fn compress_data(data: &[u8], level: i32, threads: u32) -> Vec<u8> {
    let mut output = Vec::new();
    let mut encoder = Encoder::new(&mut output, level).expect("Zstd encoder failed");
    encoder.multithread(threads).expect("Failed to set multithread");
    encoder.write_all(data).expect("Zstd write failed");
    encoder.finish().expect("Zstd finish failed");
    output
} 