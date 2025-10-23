//! SIMD kernels for vector similarity computations
//!
//! Pure computational kernels optimized with SIMD (8-wide f32 lanes).
//! These operate on raw slices and are independent of Arrow storage.

use wide::*;

/// Compute dot product using SIMD
#[inline(always)]
pub fn dot_product(vec1: &[f32], vec2: &[f32]) -> f32 {
    vec1.chunks_exact(8)
        .zip(vec2.chunks_exact(8))
        .map(|(v1, v2)| f32x8::from(v1) * f32x8::from(v2))
        .fold(f32x8::splat(0.0), |acc, prod| acc + prod)
        .reduce_add()
        + vec1
            .chunks_exact(8)
            .remainder()
            .iter()
            .zip(vec2.chunks_exact(8).remainder())
            .map(|(a, b)| a * b)
            .sum::<f32>()
}

/// Compute cosine similarity with pre-computed inverse norms
#[inline(always)]
pub fn cosine_similarity(
    vec1: &[f32],
    vec2: &[f32],
    vec1_inv_norm: f32,
    vec2_inv_norm: f32,
) -> f32 {
    dot_product(vec1, vec2) * vec1_inv_norm * vec2_inv_norm
}

/// Compute squared Euclidean distance using SIMD
#[inline(always)]
pub fn euclidean_distance_squared(vec1: &[f32], vec2: &[f32]) -> f32 {
    vec1.chunks_exact(8)
        .zip(vec2.chunks_exact(8))
        .map(|(v1, v2)| {
            let diff = f32x8::from(v1) - f32x8::from(v2);
            diff * diff
        })
        .fold(f32x8::splat(0.0), |acc, squared| acc + squared)
        .reduce_add()
        + vec1
            .chunks_exact(8)
            .remainder()
            .iter()
            .zip(vec2.chunks_exact(8).remainder())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f32>()
}

/// Compute inverse norm for a vector (1 / ||v||)
#[inline]
pub fn inverse_norm(vec: &[f32]) -> f32 {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm != 0.0 { 1.0 / norm } else { 0.0 }
}
