use rayon::prelude::*;
use zstd::stream::{Encoder, Decoder};
use std::io::{Write, Read};
use std::time::Instant;
use log::{debug};

pub fn bitx_compress(data1: &[u8], data2: &[u8], threads: usize) -> (Vec<u8>, Vec<u8>) {
    let min_len = std::cmp::min(data1.len(), data2.len()) & !1;
    assert!(min_len >= 2, "Input files too small");
    let chunk_count = min_len / 2;

    // Step 1: XOR and collect (exponent, sign_mantissa)
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
    let _xor_time = xor_start.elapsed();

    // Step 2: unzip into separate buffers
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
    let _unzip_time = unzip_start.elapsed();

    // Step 3: compress in parallel using join
    let zstd_start = Instant::now();
    let (compressed_exp, compressed_sm): (Vec<u8>, Vec<u8>) = rayon::join(
        || zstd_compress_data(&exp_output, 3, threads),
        || zstd_compress_data(&sm_output, 3, threads),
    );
    let _zstd_time = zstd_start.elapsed();
    debug!("XOR Time: {:.3}s", _xor_time.as_secs_f64());
    debug!("Unzip Time: {:.3}s", _unzip_time.as_secs_f64());
    debug!("ZSTD Total Time: {:.3}s", _zstd_time.as_secs_f64());

    (compressed_exp, compressed_sm)
}

pub fn bitx_decompress(data1: &[u8], compressed_exp: &[u8], compressed_sm: &[u8]) -> Vec<u8> {


    // decompress in parallel
    let (exp_output, sm_output): (Vec<u8>, Vec<u8>) = rayon::join(
        || zstd_decompress_data(compressed_exp),
        || zstd_decompress_data(compressed_sm),
    );
    
    // check if the decompressed lengths are the same
    assert_eq!(exp_output.len(), sm_output.len());
    let chunk_count = exp_output.len();

    // Combined reconstruct XOR and write output in one parallel step
    let mut output = vec![0u8; chunk_count * 2];
    
    output.par_chunks_mut(2).enumerate().for_each(|(i, chunk)| {
        // Reconstruct XOR result directly
        let exp = exp_output[i];
        let sm = sm_output[i];
        let sign = ((sm >> 7) & 0x1) as u16;
        let mantissa = (sm & 0x7F) as u16;
        let xor_result = (sign << 15) | ((exp as u16) << 7) | mantissa;
        
        // Apply XOR with base data and write to output
        let idx = i * 2;
        let orig1 = u16::from_le_bytes([data1[idx], data1[idx + 1]]);
        let recovered = xor_result ^ orig1;
        chunk[0] = (recovered & 0xFF) as u8;
        chunk[1] = (recovered >> 8) as u8;
    });
    


    output
}

/// ZSTD compression (supports multithreading)
pub fn zstd_compress_data(data: &[u8], level: i32, threads: usize) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() / 3); // Estimated compression ratio
    let mut encoder = Encoder::new(&mut output, level).expect("Zstd encoder failed");
    encoder.multithread(threads as u32).expect("Failed to set threads");
    encoder.write_all(data).expect("Write failed");
    encoder.finish().expect("Finish failed");
    output
}

/// ZSTD decompression
pub fn zstd_decompress_data(data: &[u8]) -> Vec<u8> {
    let mut decoder = Decoder::new(data).expect("Failed to create decoder");
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed).expect("Failed to decompress");
    decompressed
}

// test
#[cfg(test)]
mod tests {
    use super::*;
    use rayon::ThreadPool;
    use rand::Rng;
    use std::fs;


    #[test]
    fn test_bitx_compress() {
        let size = 3*1024 * 1024 * 1024; // 1GB
        let mut data1 = vec![0u8; size];
        let mut data2 = vec![0u8; size];
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.fill(&mut data1[..]);
        rng.fill(&mut data2[..]);
        

        // throughput
        let start: Instant = Instant::now();
        let (_compressed_exp, _compressed_sm) = bitx_compress(&data1, &data2,48);
        let end = Instant::now();
        let duration = end.duration_since(start);
        println!("Throughput: {} GB/s", 3.0 / duration.as_secs_f64());
    }
    #[test]
    fn test_bitx_decompress() {
        let size = 1024 * 1024 * 1024; // 1GB
        let mut data1 = vec![0u8; size];
        let mut data2 = vec![0u8; size];
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.fill(&mut data1[..]);
        rng.fill(&mut data2[..]);
        
        let (compressed_exp, compressed_sm) = bitx_compress(&data1, &data2,48);
        let decompressed = bitx_decompress(&data1, &compressed_exp, &compressed_sm);
        //check if decompressed is equal to data2, don't use assert_eq! because it will print the whole vectors
        for i in 0..size {
            if decompressed[i] != data2[i] {
                println!("Decompressed and data2 mismatch at index {}", i);
                break;
            }
        }
    }
    #[test]
    fn test_bitx_throughput() {
        //init log
        let _ = env_logger::builder().is_test(true).try_init();
        let size = 3*1024 * 1024 * 1024; // 1GB
        let mut data1 = vec![0u8; size];
        let mut data2 = vec![0u8; size];
        let mut rng = rand::thread_rng();
        rng.fill(&mut data1[..]);
        rng.fill(&mut data2[..]);

        let start = Instant::now();
        let (compressed_exp, compressed_sm) = bitx_compress(&data1, &data2, 48);
        let duration = start.elapsed();
        println!("Compression Throughput: {:.2} GB/s", size as f64 / 1e9 / duration.as_secs_f64());

        let start_decomp = Instant::now();
        let decompressed = bitx_decompress(&data1, &compressed_exp, &compressed_sm);
        let duration_decomp = start_decomp.elapsed();
        println!("Decompression Throughput: {:.2} GB/s", size as f64 / 1e9 / duration_decomp.as_secs_f64());

        assert_eq!(decompressed, data2);
    }   
    #[test]
    fn test_bitx_throughput_real_files() {
        let _ = env_logger::builder().is_test(true).try_init();

        let path1 = "input1.bin";
        let path2 = "input2.bin";

        let start_read = Instant::now();
        let data1 = fs::read(path1).expect("Failed to read file1");
        let data2 = fs::read(path2).expect("Failed to read file2");
        let read_time = start_read.elapsed().as_secs_f64();

        let size = std::cmp::min(data1.len(), data2.len());

        println!(
            "Read {} MB from each file in {:.2} s ({:.2} MB/s)",
            size / 1024 / 1024,
            read_time,
            size as f64 / 1024.0 / 1024.0 / read_time
        );

        let start = Instant::now();
        let (compressed_exp, compressed_sm) = bitx_compress(&data1, &data2, 48);
        let duration = start.elapsed();
        println!(
            "Compression Throughput: {:.2} GB/s",
            size as f64 / 1e9 / duration.as_secs_f64()
        );

        let start_decomp = Instant::now();
        let decompressed = bitx_decompress(&data1, &compressed_exp, &compressed_sm);
        let duration_decomp = start_decomp.elapsed();
        println!(
            "Decompression Throughput: {:.2} GB/s",
            size as f64 / 1e9 / duration_decomp.as_secs_f64()
        );

        assert_eq!(decompressed.len(), size);
        assert_eq!(&decompressed[..size], &data2[..size]);
    }

}