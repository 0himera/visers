use crate::{RGB_BYTES, RGB_SIDE};
use image::{
    codecs::webp::WebPDecoder, imageops, metadata::Orientation, ImageBuffer, ImageDecoder, Rgb,
};
use std::io::Cursor;

/// Look for APP1 in the JPEG header. Stop at the start of the first entropy-coded scan.
pub(super) fn jpeg(bytes: &[u8]) -> Option<Orientation> {
    let mut bytes = bytes.strip_prefix(&[0xff, 0xd8])?;
    loop {
        bytes = bytes.strip_prefix(&[0xff])?;
        while let Some(rest) = bytes.strip_prefix(&[0xff]) {
            bytes = rest;
        }
        let (&marker, rest) = bytes.split_first()?;
        match marker {
            0xda | 0xd9 => return None,
            0x01 | 0xd0..=0xd7 => {
                bytes = rest;
                continue;
            }
            _ => {}
        }
        let (length, payload) = rest.split_at_checked(2)?;
        let length = u16::from_be_bytes([length[0], length[1]]);
        let (segment, remaining) = payload.split_at_checked(usize::from(length.checked_sub(2)?))?;
        if marker == 0xe1 {
            if let Some(tag) = segment
                .strip_prefix(b"Exif\0\0")
                .and_then(Orientation::from_exif_chunk)
            {
                return Some(tag);
            }
        }
        bytes = remaining;
    }
}

pub(super) fn webp(bytes: &[u8]) -> Option<Orientation> {
    // Only the extended container can carry EXIF. Most files need no metadata reader.
    if bytes.get(12..16)? != b"VP8X" || bytes.get(20)? & 0x08 == 0 {
        return None;
    }
    let data = WebPDecoder::new(Cursor::new(bytes))
        .ok()?
        .exif_metadata()
        .ok()??;
    Orientation::from_exif_chunk(data.strip_prefix(b"Exif\0\0").unwrap_or(&data))
}

/// Transform the final square in fast mode, so a rotated photo needs only thumbnail-sized scratch.
pub(super) fn apply_to_thumbnail<'a>(
    rgb: &'a [u8],
    orientation: Orientation,
    scratch: &'a mut Vec<u8>,
) -> image::ImageResult<&'a [u8]> {
    if orientation == Orientation::NoTransforms {
        return Ok(rgb);
    }
    let side = RGB_SIDE as u32;
    let src =
        ImageBuffer::<Rgb<u8>, _>::from_raw(side, side, rgb).expect("RGB thumbnail dimensions");
    scratch.resize(RGB_BYTES, 0);
    let mut dst = ImageBuffer::from_raw(side, side, scratch.as_mut_slice())
        .expect("RGB thumbnail dimensions");
    match orientation {
        Orientation::Rotate90 | Orientation::Rotate90FlipH => {
            imageops::rotate90_in(&src, &mut dst)?
        }
        Orientation::Rotate180 => imageops::rotate180_in(&src, &mut dst)?,
        Orientation::Rotate270 | Orientation::Rotate270FlipH => {
            imageops::rotate270_in(&src, &mut dst)?
        }
        Orientation::FlipHorizontal => imageops::flip_horizontal_in(&src, &mut dst)?,
        Orientation::FlipVertical => imageops::flip_vertical_in(&src, &mut dst)?,
        Orientation::NoTransforms => unreachable!(),
    }
    if matches!(
        orientation,
        Orientation::Rotate90FlipH | Orientation::Rotate270FlipH
    ) {
        imageops::flip_horizontal_in_place(&mut dst);
    }
    Ok(scratch)
}
