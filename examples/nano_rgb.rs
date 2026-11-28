//! CLI, profiler, and benchmark for `nano-v1`.
//! Usage:
//!   nano_rgb <corpus.rgb> <output.f32>
//!   nano_rgb <corpus.rgb> --bench
//!   nano_rgb <corpus.rgb> --profile
//!   nano_rgb <corpus.rgb> --bench-batch <threads>

use std::{
    alloc::{GlobalAlloc, Layout, System},
    fs,
    hint::black_box,
    io::Write,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};
use visers_nano::{
    NanoEncoder, NanoSpinPool, NanoStageTimings, NANO_BAKED_BYTES_LEN, NANO_EMBEDDING_DIM,
    RGB_BYTES,
};

struct CountedAllocator;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountedAllocator = CountedAllocator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        return Err(
            "usage: nano_rgb <corpus.rgb> <output.f32|--bench|--profile|--bench-batch N>".into(),
        );
    }

    let model = Arc::new(NanoEncoder::baked()?);
    let mut ws = model.workspace();
    let dim = model.embedding_dim();
    let bytes = fs::read(&args[1])?;
    if bytes.is_empty() || bytes.len() % RGB_BYTES != 0 {
        return Err("invalid 128x128 RGB corpus length".into());
    }

    let mut out = vec![0.0f32; dim];
    if args[2] == "--bench" {
        let images: Vec<_> = bytes.chunks_exact(RGB_BYTES).collect();
        for _ in 0..2 {
            for image in &images {
                model.encode_rgb(image, &mut ws, &mut out)?;
                black_box(&out);
            }
        }
        let mut samples = Vec::with_capacity(images.len() * 5);
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        for _ in 0..5 {
            for image in &images {
                let t0 = Instant::now();
                model.encode_rgb(image, &mut ws, &mut out)?;
                black_box(&out);
                samples.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let inference_allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        samples.sort_by(f64::total_cmp);
        let pct = |p: usize| samples[(samples.len() - 1) * p / 100];
        println!(
            "{{\"samples\":{},\"images\":{},\"dim\":{},\"baked_bytes\":{},\"p50_ms\":{:.6},\"p90_ms\":{:.6},\"p95_ms\":{:.6},\"p99_ms\":{:.6},\"mean_ms\":{:.6},\"images_per_second\":{:.3},\"inference_allocations\":{},\"backend\":\"{}\"}}",
            samples.len(),
            images.len(),
            NANO_EMBEDDING_DIM,
            NANO_BAKED_BYTES_LEN,
            pct(50),
            pct(90),
            pct(95),
            pct(99),
            mean,
            1000.0 / mean,
            inference_allocations,
            model.backend(),
        );
        return Ok(());
    }

    if args[2] == "--profile" {
        let images: Vec<_> = bytes.chunks_exact(RGB_BYTES).collect();
        for image in &images {
            model.encode_rgb(image, &mut ws, &mut out)?;
        }
        let mut prof = NanoStageTimings::default();
        let rounds = 5;
        for _ in 0..rounds {
            for image in &images {
                model.encode_rgb_profiled(image, &mut ws, &mut out, &mut prof)?;
                black_box(&out);
            }
        }
        let n = (images.len() * rounds) as f64;
        let sum = prof.fused_stem_b0_us + prof.conv3x3_us + prof.dw_us + prof.pw_us + prof.head_us;
        println!(
            "{{\"samples\":{},\"fused_stem_b0_us\":{:.2},\"conv3x3_us\":{:.2},\"dw_us\":{:.2},\"pw_us\":{:.2},\"head_us\":{:.2},\"sum_us\":{:.2}}}",
            images.len() * rounds,
            prof.fused_stem_b0_us / n,
            prof.conv3x3_us / n,
            prof.dw_us / n,
            prof.pw_us / n,
            prof.head_us / n,
            sum / n,
        );
        return Ok(());
    }

    if args[2] == "--bench-batch" {
        let threads: usize = args.get(3).ok_or("missing thread count")?.parse()?;
        let images: Vec<_> = bytes.chunks_exact(RGB_BYTES).collect();
        let pool = NanoSpinPool::new(Arc::clone(&model), threads);
        let mut batch_out = vec![0.0f32; images.len() * dim];
        let mut seq_out = vec![0.0f32; images.len() * dim];
        for (i, &im) in images.iter().enumerate() {
            model.encode_rgb(im, &mut ws, &mut seq_out[i * dim..(i + 1) * dim])?;
        }

        // Warmup
        for _ in 0..4 {
            pool.encode_batch(&images, &mut ws, &mut batch_out)?;
            black_box(&batch_out);
        }
        assert_eq!(
            batch_out, seq_out,
            "NanoSpinPool output must be bit-identical to single-thread"
        );

        let rounds = 20usize;
        let mut round_ms = Vec::with_capacity(rounds);
        let before = ALLOCATIONS.load(Ordering::Relaxed);
        for _ in 0..rounds {
            let t0 = Instant::now();
            pool.encode_batch(&images, &mut ws, &mut batch_out)?;
            black_box(&batch_out);
            round_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let inference_allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
        let total_images = images.len() * rounds;
        let mean_batch_ms = round_ms.iter().sum::<f64>() / rounds as f64;
        round_ms.sort_by(f64::total_cmp);
        let p50_batch_ms = round_ms[(rounds - 1) / 2];
        let per_image_p50_ms = p50_batch_ms / images.len() as f64;
        let ips = 1000.0 * images.len() as f64 / mean_batch_ms;
        println!(
            "{{\"threads\":{},\"batch_images\":{},\"total_images\":{},\"p50_batch_ms\":{:.4},\"per_image_p50_ms\":{:.6},\"images_per_second\":{:.1},\"inference_allocations\":{},\"bit_exact\":true}}",
            pool.threads(),
            images.len(),
            total_images,
            p50_batch_ms,
            per_image_p50_ms,
            ips,
            inference_allocations,
        );
        return Ok(());
    }

    let mut file = fs::File::create(&args[2])?;
    for image in bytes.chunks_exact(RGB_BYTES) {
        model.encode_rgb(image, &mut ws, &mut out)?;
        for &v in &out {
            file.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}
