use image::{ColorType, DynamicImage, Limits};
use libwebp_sys as ffi;

use super::{decode::rgb_image, DecodeMode, ImageError};
use crate::RGB_SIDE;

// Animated WebPs go through the animation-aware reader to get the composited first frame.
pub(super) fn decode_still(
    bytes: &[u8],
    mode: DecodeMode,
    pixels: &mut Vec<u8>,
) -> Result<Option<DynamicImage>, ImageError> {
    let mut config = ffi::WebPDecoderConfig::new()
        .map_err(|_| ImageError::WebP("decoder ABI mismatch".into()))?;
    // SAFETY: libwebp reads the supplied slice and initializes the feature struct.
    let status = unsafe { ffi::WebPGetFeatures(bytes.as_ptr(), bytes.len(), &mut config.input) };
    check(status)?;
    let width = config.input.width as u32;
    let height = config.input.height as u32;
    Limits::default().reserve_buffer(width, height, ColorType::Rgba8)?;
    if config.input.has_animation != 0 {
        return Ok(None);
    }

    let side = RGB_SIDE as u32;
    let scaled =
        mode == DecodeMode::Fast && config.input.has_alpha == 0 && width.min(height) >= 2 * side;
    let (width, height) = if scaled {
        (side, side)
    } else {
        (width, height)
    };
    config.options.use_scaling = i32::from(scaled);
    config.options.scaled_width = width as i32;
    config.options.scaled_height = height as i32;
    pixels.resize(width as usize * height as usize * 3, 0);
    config.output.colorspace = ffi::WEBP_CSP_MODE::MODE_RGB;
    config.output.is_external_memory = 1;
    config.output.u.RGBA = ffi::WebPRGBABuffer {
        rgba: pixels.as_mut_ptr(),
        stride: (width * 3) as i32,
        size: pixels.len(),
    };
    // SAFETY: the external RGB buffer covers every row and neither slice moves during the call.
    let status = unsafe {
        let status = ffi::WebPDecode(bytes.as_ptr(), bytes.len(), &mut config);
        ffi::WebPFreeDecBuffer(&mut config.output);
        status
    };
    check(status)?;
    Ok(Some(rgb_image(width, height, pixels)))
}

fn check(status: ffi::VP8StatusCode) -> Result<(), ImageError> {
    if status == ffi::VP8StatusCode::VP8_STATUS_OK {
        Ok(())
    } else {
        Err(ImageError::WebP(format!("{status:?}")))
    }
}
