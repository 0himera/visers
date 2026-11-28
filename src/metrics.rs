fn check_dimensions(a: &[f32], b: &[f32]) -> Result<(), &'static str> {
    if a.len() != b.len() {
        return Err("embedding dimensions must match");
    }
    if a.is_empty() {
        return Err("embeddings must have at least one dimension");
    }
    Ok(())
}

/// Cosine similarity of two finite, nonzero vectors of the same length.
/// Accumulates in f64 and accepts unnormalized vectors as well as Nano embeddings.
pub fn similarity(a: &[f32], b: &[f32]) -> Result<f64, &'static str> {
    check_dimensions(a, b)?;
    let (mut dot, mut a_sq, mut b_sq) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        a_sq += x * x;
        b_sq += y * y;
    }
    if !a_sq.is_finite() || !b_sq.is_finite() {
        return Err("embedding values must be finite");
    }
    if a_sq == 0.0 || b_sq == 0.0 {
        return Err("cosine similarity requires nonzero vectors");
    }
    Ok((dot / (a_sq * b_sq).sqrt()).clamp(-1.0, 1.0))
}

/// Euclidean (L2) distance between two finite vectors of the same nonzero length.
pub fn distance(a: &[f32], b: &[f32]) -> Result<f64, &'static str> {
    check_dimensions(a, b)?;
    let squared: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum();
    if !squared.is_finite() {
        return Err("embedding values must be finite");
    }
    Ok(squared.sqrt())
}

#[cfg(test)]
mod tests {
    use super::{distance, similarity};

    #[test]
    fn metrics_match_general_vectors_and_check_their_domain() {
        assert!((similarity(&[3., 4.], &[4., 3.]).unwrap() - 0.96).abs() < 1e-12);
        assert_eq!(distance(&[0., 0.], &[3., 4.]), Ok(5.));
        assert_eq!(similarity(&[f32::MAX], &[-f32::MAX]), Ok(-1.));
        assert_eq!(
            distance(&[f32::MAX], &[-f32::MAX]),
            Ok(2. * f64::from(f32::MAX))
        );
        assert!(similarity(&[0.], &[1.]).is_err());
        for (a, b) in [
            (&[][..], &[][..]),
            (&[1.][..], &[1., 2.][..]),
            (&[f32::NAN][..], &[1.][..]),
            (&[1.][..], &[f32::INFINITY][..]),
        ] {
            assert!(similarity(a, b).is_err());
            assert!(distance(a, b).is_err());
        }
    }
}
