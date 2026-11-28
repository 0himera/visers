#![cfg(feature = "image-files")]

use std::{fs, io::Cursor, num::NonZeroUsize, path::Path, process::Command, time::Duration};

use image::{DynamicImage, ImageFormat, Rgb, RgbImage, Rgba, RgbaImage};
use visers_nano::{
    image_io::{encode_files, DecodeMode},
    NanoEncoder, RGB_SIDE,
};

fn frame() -> RgbImage {
    let colors = [[13, 99, 33], [220, 21, 90], [28, 44, 187], [253, 189, 21]];
    RgbImage::from_fn(128, 128, |x, y| {
        Rgb(colors[(x / 19 + 2 * (y / 27)) as usize % 4])
    })
}

fn compress(img: &DynamicImage, format: ImageFormat) -> Vec<u8> {
    if format == ImageFormat::Jpeg {
        let img = img.to_rgb8();
        let width = img.width() as usize;
        let frame = turbojpeg::Image {
            pixels: img.as_raw().as_slice(),
            width,
            height: img.height() as usize,
            pitch: width * 3,
            format: turbojpeg::PixelFormat::RGB,
        };
        return turbojpeg::compress(frame, 90, turbojpeg::Subsamp::Sub2x2)
            .unwrap()
            .to_vec();
    }
    let mut cursor = Cursor::new(Vec::new());
    img.write_to(&mut cursor, format).unwrap();
    cursor.into_inner()
}

fn expected(model: &NanoEncoder, img: &RgbImage) -> Vec<f32> {
    let mut result = vec![0.0; model.embedding_dim()];
    model
        .encode_rgb(img, &mut model.workspace(), &mut result)
        .unwrap();
    result
}

#[test]
fn lossless_formats_and_rectangular_alpha_match_raw_rgb() {
    let model = NanoEncoder::baked().unwrap();
    let mut pipeline = model.image_pipeline(DecodeMode::Fast);
    let rgb = frame();
    let reference = expected(&model, &rgb);
    let mut out = vec![f32::INFINITY; model.embedding_dim() + 1];
    let img = DynamicImage::ImageRgb8(rgb.clone());
    for format in [
        ImageFormat::Png,
        ImageFormat::WebP,
        ImageFormat::Bmp,
        ImageFormat::Gif,
        ImageFormat::Pnm,
    ] {
        let t = pipeline
            .encode_bytes(&compress(&img, format), &mut out)
            .unwrap();
        assert_eq!(t.read, Duration::ZERO);
        assert_eq!(&out[..model.embedding_dim()], reference, "{format:?}");
        assert_eq!(out[model.embedding_dim()], f32::INFINITY);
    }
    let rgba = RgbaImage::from_fn(256, 128, |x, y| {
        let [r, g, b] = rgb.get_pixel(x / 2, y).0;
        if x % 2 == 0 {
            Rgba([r, g, b, 0])
        } else {
            Rgba([0, 0, 0, 255])
        }
    });
    let encoded = compress(&rgba.into(), ImageFormat::Png);
    let averaged = RgbImage::from_fn(128, 128, |x, y| {
        Rgb(rgb.get_pixel(x, y).0.map(|v| v.div_ceil(2)))
    });
    for mode in [DecodeMode::Full, DecodeMode::Fast] {
        model
            .image_pipeline(mode)
            .encode_bytes(&encoded, &mut out)
            .unwrap();
        assert_eq!(&out[..model.embedding_dim()], expected(&model, &averaged));
    }
}

fn jpeg_with_orientation(bytes: &[u8], tag: u8) -> Vec<u8> {
    // APP1, little-endian TIFF, one IFD0 orientation entry of type SHORT.
    let mut exif = b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec();
    exif.extend_from_slice(&[tag, 0, 0, 0, 0, 0, 0, 0]);
    let mut data = vec![0xff, 0xd8, 0xff, 0xe1];
    data.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
    data.extend_from_slice(&exif);
    data.extend_from_slice(&bytes[2..]);
    data
}

#[test]
fn all_jpeg_orientations_match_physical_transforms() {
    let model = NanoEncoder::baked().unwrap();
    let mut full = model.image_pipeline(DecodeMode::Full);
    let mut fast = model.image_pipeline(DecodeMode::Fast);
    let bytes = compress(&frame().into(), ImageFormat::Jpeg);
    let decoded = turbojpeg::decompress(&bytes, turbojpeg::PixelFormat::RGB).unwrap();
    let image = RgbImage::from_raw(128, 128, decoded.pixels).unwrap();
    let edge = RGB_SIDE as u32 - 1;
    for tag in 1..=8 {
        let oriented = RgbImage::from_fn(128, 128, |x, y| {
            let (sx, sy) = match tag {
                1 => (x, y),
                2 => (edge - x, y),
                3 => (edge - x, edge - y),
                4 => (x, edge - y),
                5 => (y, x),
                6 => (y, edge - x),
                7 => (edge - y, edge - x),
                _ => (edge - y, x),
            };
            *image.get_pixel(sx, sy)
        });
        let mut out = vec![0.0; model.embedding_dim()];
        let tagged = jpeg_with_orientation(&bytes, tag);
        for pipeline in [&mut full, &mut fast] {
            pipeline.encode_bytes(&tagged, &mut out).unwrap();
            assert_eq!(out, expected(&model, &oriented), "EXIF {tag}");
        }
    }
}

#[test]
fn animated_webp_orientation_and_gray16_match_training_conversion() {
    let model = NanoEncoder::baked().unwrap();
    // The first 8x4 lossless frame has four quadrants and orientation 6; frame two is magenta.
    let webp = include_bytes!("fixtures/oriented-animation.webp");
    let rotated_colors = [[23, 49, 185], [125, 43, 19], [227, 186, 47], [18, 135, 61]];
    let first = RgbImage::from_fn(128, 128, |x, y| {
        Rgb(rotated_colors[(x >= 64) as usize + 2 * (y >= 64) as usize])
    });
    let gray = image::ImageBuffer::from_fn(128, 128, |x, y| image::Luma([(3 * x + y) as u16]));
    let clipped = RgbImage::from_fn(128, 128, |x, y| Rgb([(3 * x + y).min(255) as u8; 3]));
    let png = compress(&DynamicImage::ImageLuma16(gray), ImageFormat::Png);
    let mut out = vec![0.0; model.embedding_dim()];
    for mode in [DecodeMode::Fast, DecodeMode::Full] {
        let mut pipeline = model.image_pipeline(mode);
        for (bytes, rgb) in [(webp.as_slice(), &first), (png.as_slice(), &clipped)] {
            pipeline.encode_bytes(bytes, &mut out).unwrap();
            assert_eq!(out, expected(&model, rgb));
        }
    }
}

#[test]
fn decoder_recovers_from_errors_and_resets_jpeg_scale() {
    let model = NanoEncoder::baked().unwrap();
    let mut pipeline = model.image_pipeline(DecodeMode::Fast);
    let mut out = vec![0.0; model.embedding_dim()];
    let rgb = frame();
    let large = DynamicImage::ImageRgb8(RgbImage::from_fn(1024, 640, |x, y| {
        *rgb.get_pixel(x % 128, y % 128)
    }));
    let large = compress(&large, ImageFormat::Jpeg);
    let small = compress(&rgb.into(), ImageFormat::Jpeg);
    let mut reference = vec![0.0; model.embedding_dim()];
    pipeline.encode_bytes(&small, &mut reference).unwrap();
    pipeline.encode_bytes(&large, &mut out).unwrap();
    let before = out.clone();
    let truncated_png = &b"\x89PNG\r\n\x1a\n"[..];
    let truncated_webp = &b"RIFF\x0c\x00\x00\x00WEBPVP8 "[..];
    for bytes in [
        &b"bad"[..],
        &small[..7],
        &large[..2],
        truncated_png,
        truncated_webp,
    ] {
        assert!(pipeline.encode_bytes(bytes, &mut out).is_err());
        assert_eq!(out, before);
    }
    assert!(pipeline.encode_bytes(&small, &mut out[..2]).is_err());
    pipeline.encode_bytes(&small, &mut out).unwrap();
    assert_eq!(out, reference);
}

#[test]
fn ordered_batch_cli_and_file_errors() {
    let temporary = tempfile::tempdir().unwrap();
    let dir = temporary.path();
    let a = dir.join("first.data");
    let b = dir.join("second.jpg");
    let absent = dir.join("missing.webp");
    let output = dir.join("embeddings.f32");
    let rgb_a = frame();
    let rgb_b = image::imageops::rotate90(&rgb_a);
    fs::write(&a, compress(&rgb_a.clone().into(), ImageFormat::Png)).unwrap();
    fs::write(&b, compress(&rgb_b.clone().into(), ImageFormat::WebP)).unwrap();

    let model = NanoEncoder::baked().unwrap();
    let (ref_a, ref_b) = (expected(&model, &rgb_a), expected(&model, &rgb_b));
    let jobs = NonZeroUsize::new(2).unwrap();
    assert!(encode_files::<&Path>(&model, &[], DecodeMode::Fast, jobs).is_empty());
    let results = encode_files(&model, &[&b, &absent, &a], DecodeMode::Fast, jobs);
    assert_eq!(results[0].as_ref().unwrap().embedding, ref_b);
    assert!(results[1].is_err());
    assert_eq!(results[2].as_ref().unwrap().embedding, ref_a);

    let cli = || Command::new(env!("CARGO_BIN_EXE_nano_image"));
    let run = cli()
        .args(["--jobs", "2", "--profile", "-o"])
        .arg(&output)
        .arg(&b)
        .arg(&a)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stderr).contains("inference_ms"));
    let binary: Vec<_> = ref_b
        .iter()
        .chain(&ref_a)
        .flat_map(|x| x.to_le_bytes())
        .collect();
    assert_eq!(fs::read(&output).unwrap(), binary);
    let fail = cli()
        .arg("-o")
        .arg(&output)
        .arg(&a)
        .arg(&absent)
        .output()
        .unwrap();
    assert!(!fail.status.success());
    assert_eq!(fs::read(&output).unwrap(), binary);
    let same = cli().arg("-o").arg(&a).arg(&a).output().unwrap();
    assert!(!same.status.success());
    let json = cli().arg("--full").arg(&a).output().unwrap();
    assert!(json.status.success());
    let json = String::from_utf8(json.stdout).unwrap();
    let row: Vec<f32> = json
        .trim()
        .strip_prefix('[')
        .unwrap()
        .strip_suffix(']')
        .unwrap()
        .split(',')
        .map(|n| n.parse().unwrap())
        .collect();
    assert_eq!(row, ref_a);
}

#[test]
fn baked_image_api_orders_mixed_inputs_and_reuses_modes() {
    use visers_nano::{
        encode, encode_into, encode_single, encode_with_options, ImageInput, DIMENSIONS,
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("image");
    let png = compress(&frame().into(), ImageFormat::Png);
    fs::write(&path, &png).unwrap();
    let rgb = frame();
    let large = RgbImage::from_fn(769, 555, |x, y| *rgb.get_pixel(x % 128, y % 128));
    let jpeg = compress(&large.into(), ImageFormat::Jpeg);
    let model = NanoEncoder::baked().unwrap();
    let mut reference = [0.0; DIMENSIONS];
    let mut full = model.image_pipeline(DecodeMode::Full);
    full.encode_bytes(&jpeg, &mut reference).unwrap();
    let fast = encode_single(&jpeg).unwrap();
    assert_ne!(reference, fast);
    let image = encode_single(&path).unwrap();
    let inputs = [
        ImageInput::Path(&path),
        ImageInput::Bytes(&jpeg),
        ImageInput::Bytes(&png),
    ];
    for jobs in [1, 3] {
        let rows = encode_with_options(&inputs, DecodeMode::Full, NonZeroUsize::new(jobs).unwrap())
            .unwrap();
        assert_eq!(rows, [image, reference, image]);
        assert_eq!(encode_single(&jpeg).unwrap(), fast);
    }
    assert_eq!(encode(inputs).unwrap(), [image, fast, image]);
    assert_eq!(encode(inputs.as_slice()).unwrap(), [image, fast, image]);
    let paths = [path.as_path(), path.as_path()];
    assert_eq!(encode(paths.as_slice()).unwrap(), [image, image]);
    let mut buffer = vec![f32::INFINITY; DIMENSIONS + 1];
    assert!(encode_into(&jpeg, &mut buffer[..2]).is_err());
    encode_into(&jpeg, &mut buffer).unwrap();
    assert_eq!(&buffer[..DIMENSIONS], &fast);
    assert!(buffer[DIMENSIONS].is_infinite());
    assert!(encode(std::iter::empty::<&str>()).unwrap().is_empty());
    let missing = dir.path().join("missing");
    let err = encode([
        inputs[0],
        ImageInput::Path(&missing),
        ImageInput::Bytes(b"broken"),
    ])
    .unwrap_err();
    assert_eq!(err.index, 1);
    assert!(matches!(
        err.source,
        visers_nano::image_io::ImageError::Io(_)
    ));
}
