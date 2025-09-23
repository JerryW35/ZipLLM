fn main() {
    println!("Hello, world!");
}
//
// Copyright (c) 2023 Nathan Fiedler
//
use clap::{Arg, arg, command, value_parser};
use fastcdc::v2020::*;
use std::fs::File;
use std::io::{Read, Write, BufWriter};
use std::time::Instant;
use std::path::Path;

fn main() {
    let matches = command!("Example of using v2020 chunker.")
        .about("Finds the content-defined chunk boundaries of a file.")
        .arg(
            arg!(
                -s --size <SIZE> "The desired average size of the chunks."
            )
            .value_parser(value_parser!(u32)),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output file path for lines: '<hash> <size>'")
                .value_parser(value_parser!(String))
        )
        .arg(
            Arg::new("INPUT")
                .help("Sets the input file to use")
                .required(true)
                .index(1),
        )
        .get_matches();
    let size = matches.get_one::<u32>("size").unwrap_or(&131072);
    let avg_size = *size;
    let filename = matches.get_one::<String>("INPUT").unwrap();

    // Read the entire file into memory
    let mut file = File::open(filename).expect("cannot open file!");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("failed to read file into memory");

    let min_size = avg_size / 4;
    let max_size = avg_size * 4;

    // Start timing
    let start = Instant::now();

    // Use FastCDC on the in-memory buffer
    let chunker = FastCDC::new(&buffer[..], min_size, avg_size, max_size);
    let mut chunk_count = 0;
    let mut total_chunk_bytes = 0usize;
    let mut lines: Vec<String> = Vec::new();
    for entry in chunker {
        let start = entry.offset as usize;
        let end = start + entry.length as usize;
        let chunk = &buffer[start..end];
        let hash = blake3::hash(chunk);
        let hash_hex = hash.to_hex();
        lines.push(format!("{} {}", hash_hex, entry.length));
        chunk_count += 1;
        total_chunk_bytes += entry.length as usize;
    }

    // Write all lines at once at the end to reduce IO switching
    let in_path = Path::new(filename);
    let out_path = if let Some(p) = matches.get_one::<String>("output") {
        std::path::PathBuf::from(p)
    } else {
        in_path.with_extension("chunks.txt")
    };
    let mut writer = BufWriter::new(File::create(&out_path).expect("cannot create output txt"));
    for line in lines {
        writeln!(writer, "{}", line).expect("write failed");
    }

    let elapsed = start.elapsed();
    let elapsed_secs = elapsed.as_secs_f64();
    let file_size_mb = buffer.len() as f64 / (1024.0 * 1024.0);
    let throughput = if elapsed_secs > 0.0 {
        file_size_mb / elapsed_secs
    } else {
        0.0
    };

    println!(
        "Processed {:.2} MB in {:.3} seconds, throughput = {:.2} MB/s, {} chunks",
        file_size_mb, elapsed_secs, throughput, chunk_count
    );
    println!("Output written to: {}", out_path.display());
}