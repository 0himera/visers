//! `visers-nano/build.rs` — Compile-time `.rodata` baking and Matryoshka head slicing.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=VISERS_NANO_WEIGHTS");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let default_weights = manifest_dir.join("weights/nano_v1.bin");
    println!("cargo:rerun-if-changed={}", default_weights.display());

    let nano_dim: usize = if env::var("CARGO_FEATURE_DIM_256").is_ok() {
        256
    } else if env::var("CARGO_FEATURE_DIM_128").is_ok() {
        128
    } else if env::var("CARGO_FEATURE_DIM_64").is_ok() {
        64
    } else {
        256
    };

    let selected = if let Ok(explicit) = env::var("VISERS_NANO_WEIGHTS") {
        let p = PathBuf::from(&explicit);
        assert!(p.is_file(), "Explicit VISERS_NANO_WEIGHTS does not exist: {explicit}");
        println!("cargo:rerun-if-changed={}", p.display());
        Some(p)
    } else if default_weights.is_file() {
        Some(default_weights)
    } else {
        None
    };

    let nano_rs = out_dir.join("nano_model.rs");
    let Some(weights_path) = selected else {
        fs::write(
            nano_rs,
            format!(
                "pub const NANO_EMBEDDING_DIM: usize = {nano_dim};\n\
                 pub const NANO_BAKED_AVAILABLE: bool = false;\n\
                 #[repr(align(64))]\n\
                 pub struct AlignedNanoBytes<const N: usize>(pub [u8; N]);\n\
                 pub static NANO_BAKED_BYTES: AlignedNanoBytes<0> = AlignedNanoBytes([]);\n"
            ),
        )
        .unwrap();
        return;
    };

    let data = fs::read(&weights_path).expect("Failed to read weights/nano_v1.bin");
    assert!(data.len() >= 44, "VSN1 file too small");
    assert_eq!(&data[0..4], b"VSN1", "Invalid VSN1 magic");
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    assert_eq!(version, 1, "Unsupported VSN1 version");
    let num_ops = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let head_in = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let head_out = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
    assert!(nano_dim <= head_out, "Requested dim {nano_dim} > head_out {head_out}");

    let mut off = 44usize;
    for _ in 0..num_ops {
        assert!(off + 16 <= data.len(), "Truncated VSN1 op descriptor");
        let co = u16::from_le_bytes(data[off + 2..off + 4].try_into().unwrap()) as usize;
        let weight_len = u32::from_le_bytes(data[off + 12..off + 16].try_into().unwrap()) as usize;
        off += 16 + co * 4 + co * 4 + weight_len;
        assert!(off <= data.len(), "Truncated VSN1 op payload");
    }

    let expected_head_bytes = head_out * 4 + head_out * 4 + head_out * head_in;
    assert_eq!(off + expected_head_bytes, data.len(), "VSN1 head length mismatch");

    let mut sliced = Vec::with_capacity(off + nano_dim * 4 + nano_dim * 4 + nano_dim * head_in);
    sliced.extend_from_slice(&data[0..off]);
    sliced[16..20].copy_from_slice(&(nano_dim as u32).to_le_bytes());

    let scales_start = off;
    let biases_start = off + head_out * 4;
    let weights_start = biases_start + head_out * 4;
    sliced.extend_from_slice(&data[scales_start..scales_start + nano_dim * 4]);
    sliced.extend_from_slice(&data[biases_start..biases_start + nano_dim * 4]);
    sliced.extend_from_slice(&data[weights_start..weights_start + nano_dim * head_in]);

    let baked_bin = out_dir.join("nano_baked.bin");
    fs::write(&baked_bin, &sliced).unwrap();
    let total_len = sliced.len();
    fs::write(
        nano_rs,
        format!(
            "pub const NANO_EMBEDDING_DIM: usize = {nano_dim};\n\
             pub const NANO_BAKED_AVAILABLE: bool = true;\n\
             #[repr(align(64))]\n\
             pub struct AlignedNanoBytes<const N: usize>(pub [u8; N]);\n\
             pub static NANO_BAKED_BYTES: AlignedNanoBytes<{total_len}> = AlignedNanoBytes(*include_bytes!(concat!(env!(\"OUT_DIR\"), \"/nano_baked.bin\")));\n"
        ),
    )
    .unwrap();
}
