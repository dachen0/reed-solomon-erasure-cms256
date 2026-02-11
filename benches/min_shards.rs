use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::env;
use std::process::Command;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};
use reed_solomon_erasure::galois_8::ReedSolomon;

const ALIGN: usize = 32;
const DEFAULT_BLOCK_SIZE: usize = 4096;
const DEFAULT_ITERS: usize = 200;
const DEFAULT_VALUES: &[usize] = &[2, 4, 6, 8, 10, 12, 16, 20, 24, 32];
const DEFAULT_CASES: &[(usize, usize)] = &[(4, 4), (8, 4), (10, 4), (16, 8), (32, 16)];

struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    layout: Layout,
}

impl AlignedBuf {
    fn new(len: usize, align: usize) -> Self {
        let layout = Layout::from_size_align(len, align).expect("layout");
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            panic!("alloc failed");
        }
        Self { ptr, len, layout }
    }

    fn fill(&mut self, val: u8) {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }.fill(val);
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) };
    }
}

impl AsRef<[u8]> for AlignedBuf {
    fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl AsMut<[u8]> for AlignedBuf {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|val| val.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_values() -> Vec<usize> {
    if let Ok(values) = env::var("RSE_MIN_SHARDS_VALUES") {
        let mut parsed = Vec::new();
        for part in values.split(',') {
            if let Ok(v) = part.trim().parse::<usize>() {
                parsed.push(v);
            }
        }
        if !parsed.is_empty() {
            return parsed;
        }
    }
    DEFAULT_VALUES.to_vec()
}

fn create_shards(block_size: usize, data: usize, parity: usize) -> Vec<AlignedBuf> {
    let mut rng = SmallRng::from_entropy();
    let mut shards = Vec::with_capacity(data + parity);

    for _ in 0..data {
        let mut buf = AlignedBuf::new(block_size, ALIGN);
        rng.fill_bytes(buf.as_mut());
        shards.push(buf);
    }
    for _ in 0..parity {
        let mut buf = AlignedBuf::new(block_size, ALIGN);
        buf.fill(0);
        shards.push(buf);
    }

    shards
}

fn run_child() {
    let min_shards = env_usize("RSE_ISA_L_MIN_SHARDS", 0);
    let block_size = env_usize("RSE_MIN_SHARDS_BLOCK_SIZE", DEFAULT_BLOCK_SIZE);
    let iters = env_usize("RSE_MIN_SHARDS_ITERS", DEFAULT_ITERS);

    let mut total_bytes: u64 = 0;
    let start = Instant::now();

    for &(data, parity) in DEFAULT_CASES.iter() {
        let rs = ReedSolomon::new(data, parity).unwrap();
        let mut shards = create_shards(block_size, data, parity);

        for _ in 0..iters {
            for shard in &mut shards[data..] {
                shard.fill(0);
            }
            rs.encode(&mut shards).unwrap();
        }

        total_bytes += (block_size * data * iters) as u64;
    }

    let secs = start.elapsed().as_secs_f64();
    let mbps = (total_bytes as f64 / (1024.0 * 1024.0)) / secs;

    println!("min_shards={} mbps={:.2}", min_shards, mbps);
}

fn run_parent() {
    if !cfg!(feature = "isa-l") {
        eprintln!("min_shards benchmark requires --features isa-l");
        return;
    }

    let mut best_value = 0usize;
    let mut best_mbps = 0.0f64;

    let values = parse_values();

    for value in values {
        let mut cmd = Command::new(env::current_exe().expect("current exe"));
        cmd.arg("--child")
            .env("RSE_ISA_L_MIN_SHARDS", value.to_string());

        if let Ok(min_work) = env::var("RSE_ISA_L_MIN_WORK") {
            cmd.env("RSE_ISA_L_MIN_WORK", min_work);
        }
        if let Ok(block_size) = env::var("RSE_MIN_SHARDS_BLOCK_SIZE") {
            cmd.env("RSE_MIN_SHARDS_BLOCK_SIZE", block_size);
        }
        if let Ok(iters) = env::var("RSE_MIN_SHARDS_ITERS") {
            cmd.env("RSE_MIN_SHARDS_ITERS", iters);
        }

        let output = cmd.output().expect("run child");
        let stdout = String::from_utf8_lossy(&output.stdout);
        print!("{}", stdout);

        for line in stdout.lines() {
            if let Some(pos) = line.find("mbps=") {
                if let Ok(mbps) = line[pos + 5..].trim().parse::<f64>() {
                    if mbps > best_mbps {
                        best_mbps = mbps;
                        best_value = value;
                    }
                }
            }
        }
    }

    eprintln!("best min_shards={} (mbps={:.2})", best_value, best_mbps);
}

fn main() {
    if env::args().any(|arg| arg == "--child") {
        run_child();
    } else {
        run_parent();
    }
}
