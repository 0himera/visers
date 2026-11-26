//! Unit, parity, zero-alloc, and lock-free spin-pool tests for `nano-v1`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use visers_nano::{
    encode_single_rgb, NanoEncoder, NanoSpinPool, NANO_BAKED_AVAILABLE, NANO_EMBEDDING_DIM,
    RGB_BYTES,
};

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.try_with(|t| t.get()).unwrap_or(false) {
            let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING.try_with(|t| t.get()).unwrap_or(false) {
            let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn pseudo_rgb(seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_add(1);
    let mut out = vec![0u8; RGB_BYTES];
    for b in &mut out {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (state >> 24) as u8;
    }
    out
}

#[test]
fn nano_baked_simd_matches_scalar_and_normalized() {
    const { assert!(NANO_BAKED_AVAILABLE, "nano_v1.bin must be baked") };
    let model = NanoEncoder::baked().expect("baked nano model");
    assert_eq!(model.embedding_dim(), NANO_EMBEDDING_DIM);
    let mut ws = model.workspace();

    for seed in 0..4u32 {
        let rgb = pseudo_rgb(seed);
        let mut out_simd = vec![0.0f32; NANO_EMBEDDING_DIM];
        let mut out_scalar = vec![0.0f32; NANO_EMBEDDING_DIM];
        model.encode_rgb(&rgb, &mut ws, &mut out_simd).unwrap();
        model.encode_rgb_scalar(&rgb, &mut ws, &mut out_scalar).unwrap();

        let norm: f32 = out_simd.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");

        let max_abs = out_simd
            .iter()
            .zip(&out_scalar)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs < 2e-5, "SIMD vs scalar drift too high: {max_abs}");
    }
}

#[test]
fn nano_zero_alloc_and_input_validation() {
    let model = NanoEncoder::baked().expect("baked nano model");
    let mut ws = model.workspace();
    let mut out = vec![0.0f32; NANO_EMBEDDING_DIM];
    let rgb = pseudo_rgb(42);

    model.encode_rgb(&rgb, &mut ws, &mut out).unwrap();
    let conv = encode_single_rgb(&rgb).unwrap();
    assert_eq!(out, conv);

    ALLOCS.with(|c| c.set(0));
    TRACKING.with(|t| t.set(true));
    model.encode_rgb(&rgb, &mut ws, &mut out).unwrap();
    let conv2 = encode_single_rgb(&rgb).unwrap();
    TRACKING.with(|t| t.set(false));
    assert_eq!(ALLOCS.with(|c| c.get()), 0, "expected 0 inference allocations");
    assert_eq!(out, conv2);

    assert!(model.encode_rgb(&rgb[..100], &mut ws, &mut out).is_err());
    assert!(model.encode_rgb(&rgb, &mut ws, &mut out[..1]).is_err());
}

#[test]
fn nano_spin_pool_bit_exact_and_zero_alloc() {
    let model = Arc::new(NanoEncoder::baked().expect("baked nano model"));
    let mut ws = model.workspace();
    let dim = model.embedding_dim();

    let imgs: Vec<Vec<u8>> = (0..16u32).map(pseudo_rgb).collect();
    let refs: Vec<&[u8]> = imgs.iter().map(|v| v.as_slice()).collect();

    let mut expected = vec![0.0f32; imgs.len() * dim];
    for (i, &im) in refs.iter().enumerate() {
        model
            .encode_rgb(im, &mut ws, &mut expected[i * dim..(i + 1) * dim])
            .unwrap();
    }

    let pool = NanoSpinPool::new(Arc::clone(&model), 4);
    let mut actual = vec![0.0f32; imgs.len() * dim];
    pool.encode_batch(&refs, &mut ws, &mut actual).unwrap();
    assert_eq!(actual, expected);

    ALLOCS.with(|c| c.set(0));
    TRACKING.with(|t| t.set(true));
    pool.encode_batch(&refs, &mut ws, &mut actual).unwrap();
    TRACKING.with(|t| t.set(false));
    assert_eq!(ALLOCS.with(|c| c.get()), 0, "expected 0 batch dispatch allocations");
    assert_eq!(actual, expected);
}
