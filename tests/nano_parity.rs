//! Validation, parity, zero-allocation, and spin-pool tests for `nano-v1`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::{Arc, Barrier};
use std::thread;
use visers_nano::{
    encode_single_rgb, NanoEncoder, NanoSpinPool, NanoWorkspace, NANO_BAKED_AVAILABLE,
    NANO_EMBEDDING_DIM, RGB_BYTES,
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
        model
            .encode_rgb_scalar(&rgb, &mut ws, &mut out_scalar)
            .unwrap();

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
fn nano_custom_rgb_normalization_matches_scalar() {
    let mut bytes = include_bytes!("../weights/nano_v1.bin").to_vec();
    let normalization = [0.31f32, 0.42, 0.53, 0.21, 0.32, 0.43];
    for (slot, value) in bytes[20..44].chunks_exact_mut(4).zip(normalization) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    let model = NanoEncoder::from_bytes(&bytes).unwrap();
    let mut workspace = model.workspace();
    let rgb = pseudo_rgb(27);
    let mut actual = vec![0.0; model.embedding_dim()];
    let mut reference = actual.clone();
    model.encode_rgb(&rgb, &mut workspace, &mut actual).unwrap();
    model
        .encode_rgb_scalar(&rgb, &mut workspace, &mut reference)
        .unwrap();
    assert!(actual
        .iter()
        .zip(reference)
        .all(|(a, b)| (a - b).abs() < 2e-5));
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
    assert_eq!(
        ALLOCS.with(|c| c.get()),
        0,
        "expected 0 inference allocations"
    );
    assert_eq!(out, conv2);

    assert!(model.encode_rgb(&rgb[..100], &mut ws, &mut out).is_err());
    assert!(model.encode_rgb(&rgb, &mut ws, &mut out[..1]).is_err());
}

#[test]
fn nano_rejects_incompatible_vsn1_graphs() {
    let weights = include_bytes!("../weights/nano_v1.bin");
    let num_ops = u32::from_le_bytes(weights[8..12].try_into().unwrap());
    let mut offset = 44;
    let offsets: Vec<_> = (0..num_ops)
        .map(|_| {
            let start = offset;
            let co = u16::from_le_bytes(weights[start + 2..start + 4].try_into().unwrap());
            let size = u32::from_le_bytes(weights[start + 12..start + 16].try_into().unwrap());
            offset += 16 + usize::from(co) * 8 + size as usize;
            start
        })
        .collect();

    let cases = [
        ("channel chain", offsets[2], 32, "invalid VSN1 op shape"),
        (
            "depthwise channels",
            offsets[5] + 2,
            48,
            "invalid VSN1 op shape",
        ),
        (
            "pointwise stride",
            offsets[2] + 5,
            2,
            "unsupported VSN1 convolution geometry",
        ),
        (
            "activation size",
            offsets[2] + 2,
            32,
            "VSN1 activation exceeds workspace",
        ),
        (
            "residual size",
            offsets[2] + 9,
            1,
            "invalid VSN1 residual save",
        ),
        (
            "residual without save",
            offsets[4] + 10,
            1,
            "invalid VSN1 residual add",
        ),
        (
            "residual shape mismatch",
            offsets[9] + 10,
            1,
            "invalid VSN1 residual add",
        ),
        (
            "head input",
            12,
            240,
            "VSN1 head input does not match final activation",
        ),
    ];

    for (name, offset, value, error) in cases {
        let mut invalid = weights.to_vec();
        invalid[offset] = value;
        assert_eq!(
            NanoEncoder::from_bytes(&invalid).err(),
            Some(error),
            "{name}"
        );
    }
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
    let mut wrong_workspace = NanoWorkspace::new(0);
    assert_eq!(
        pool.encode_batch(&refs, &mut wrong_workspace, &mut actual),
        Err("nano-v1 output/workspace mismatch")
    );
    assert!(actual.iter().all(|&value| value == 0.0));
    pool.encode_batch(&refs, &mut ws, &mut actual).unwrap();
    assert_eq!(actual, expected);

    ALLOCS.with(|c| c.set(0));
    TRACKING.with(|t| t.set(true));
    pool.encode_batch(&refs, &mut ws, &mut actual).unwrap();
    TRACKING.with(|t| t.set(false));
    assert_eq!(
        ALLOCS.with(|c| c.get()),
        0,
        "expected 0 batch dispatch allocations"
    );
    assert_eq!(actual, expected);
}

#[test]
fn nano_spin_pool_supports_concurrent_callers() {
    let model = Arc::new(NanoEncoder::baked().unwrap());
    let pool = NanoSpinPool::new(Arc::clone(&model), 3);
    let start = Barrier::new(3);
    let dim = model.embedding_dim();

    thread::scope(|scope| {
        for caller in 0..3 {
            let (model, pool, start) = (&model, &pool, &start);
            scope.spawn(move || {
                let mut ws = model.workspace();
                let images: Vec<_> = (0..3 + caller)
                    .map(|idx| pseudo_rgb(caller * 10 + idx))
                    .collect();
                let refs: Vec<_> = images.iter().map(Vec::as_slice).collect();
                let mut expected = vec![0.0; images.len() * dim];
                for (&rgb, out) in refs.iter().zip(expected.chunks_exact_mut(dim)) {
                    model.encode_rgb(rgb, &mut ws, out).unwrap();
                }

                let mut actual = vec![0.0; expected.len()];
                start.wait();
                for _ in 0..4 {
                    actual.fill(0.0);
                    pool.encode_batch(&refs, &mut ws, &mut actual).unwrap();
                    assert_eq!(actual, expected);
                }
            });
        }
    });
}
