use crate::vec::{Cmp, Metric, TakeType};
use wide::*;

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

#[inline(always)]
pub fn cosine_similarity(
    vec1: &[f32],
    vec2: &[f32],
    vec1_inv_norm: f32,
    vec2_inv_norm: f32,
) -> f32 {
    dot_product(vec1, vec2) * vec1_inv_norm * vec2_inv_norm
}

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

// Calculate scores for all queries for a chunk of 8 vectors
#[inline(always)]
pub fn calculate_scores_chunk(
    queries: &[Vec<f32>],
    query_inv_norms: &[f32],
    vectors: &[Vec<f32>],
    inv_norms: &[f32],
    metric: &Metric,
) -> Vec<f32x8> {
    queries
        .iter()
        .zip(query_inv_norms.iter())
        .map(|(query, &query_inv_norm)| {
            let mut scores = [0.0f32; 8];
            for i in 0..8 {
                scores[i] = match metric {
                    Metric::Cosine => {
                        cosine_similarity(query, &vectors[i], query_inv_norm, inv_norms[i])
                    }
                    Metric::Euclidean => euclidean_distance_squared(query, &vectors[i]),
                    Metric::DotProduct => dot_product(query, &vectors[i]),
                };
            }
            f32x8::from(&scores[..])
        })
        .collect()
}

fn filter_simd(scores: f32x8, threshold: f32, cmp: &Cmp) -> f32x8 {
    let threshold_simd = f32x8::splat(threshold);
    match cmp {
        Cmp::Lt => scores.cmp_lt(threshold_simd),
        Cmp::Gt => scores.cmp_gt(threshold_simd),
        Cmp::Leq => scores.cmp_le(threshold_simd),
        Cmp::Geq => scores.cmp_ge(threshold_simd),
        Cmp::Eq => scores.cmp_eq(threshold_simd),
    }
}

// Top-K collector with integrated filtering
pub struct TopKCollector<'a> {
    buffer: Vec<(usize, f32)>,
    k: usize,
    take_type: &'a TakeType,
    filter: Option<&'a (f32, Cmp)>,
    is_sorted: bool,
    threshold: f32,
    effective_threshold: Option<f32>,
    effective_cmp: Option<Cmp>,
}

impl<'a> TopKCollector<'a> {
    pub fn new(k: usize, take_type: &'a TakeType, filter: Option<&'a (f32, Cmp)>) -> Self {
        let threshold = match take_type {
            TakeType::Min => f32::INFINITY,
            TakeType::Max => f32::NEG_INFINITY,
        };

        // Calculate effective threshold and comparison once at init
        let (effective_threshold, effective_cmp) = match filter {
            Some((filter_threshold, filter_cmp)) => {
                // Determine which threshold is more restrictive
                let combined_threshold = match (take_type, filter_cmp) {
                    (TakeType::Min, Cmp::Lt) | (TakeType::Min, Cmp::Leq) => {
                        // For min operations, use the smaller threshold
                        Some(filter_threshold.min(threshold))
                    }
                    (TakeType::Max, Cmp::Gt) | (TakeType::Max, Cmp::Geq) => {
                        // For max operations, use the larger threshold
                        Some(filter_threshold.max(threshold))
                    }
                    _ => Some(*filter_threshold),
                };
                (combined_threshold, Some(filter_cmp.clone()))
            }
            None => (None, None),
        };

        Self {
            buffer: Vec::with_capacity(k.min(1024)),
            k,
            take_type,
            filter,
            is_sorted: true,
            threshold,
            effective_threshold,
            effective_cmp,
        }
    }

    fn get_effective_threshold(&self) -> Option<(f32, Cmp)> {
        match (&self.effective_threshold, &self.effective_cmp) {
            (Some(threshold), Some(cmp)) => Some((*threshold, cmp.clone())),
            _ => {
                if self.buffer.len() == self.k {
                    match &self.take_type {
                        TakeType::Min => Some((self.threshold, Cmp::Lt)),
                        TakeType::Max => Some((self.threshold, Cmp::Gt)),
                    }
                } else {
                    None
                }
            }
        }
    }

    fn update_effective_threshold(&mut self) {
        if self.buffer.len() == self.k {
            if let Some(ref mut eff_threshold) = self.effective_threshold {
                // Update the effective threshold based on current topk threshold and filter
                match (&self.take_type, &self.effective_cmp) {
                    (TakeType::Min, Some(Cmp::Lt)) | (TakeType::Min, Some(Cmp::Leq)) => {
                        *eff_threshold = eff_threshold.min(self.threshold);
                    }
                    (TakeType::Max, Some(Cmp::Gt)) | (TakeType::Max, Some(Cmp::Geq)) => {
                        *eff_threshold = eff_threshold.max(self.threshold);
                    }
                    _ => {} // For other comparisons, keep filter threshold
                }
            } else {
                // No filter, just use topk threshold
                self.effective_threshold = Some(self.threshold);
                self.effective_cmp = Some(match &self.take_type {
                    TakeType::Min => Cmp::Lt,
                    TakeType::Max => Cmp::Gt,
                });
            }
        }
    }

    pub fn push_chunk(&mut self, chunk_idx: usize, scores: f32x8) {
        if self.k == 0 {
            return;
        }

        // Apply effective threshold
        let mask = match self.get_effective_threshold() {
            Some((threshold, cmp)) => filter_simd(scores, threshold, &cmp),
            None => f32x8::splat(1.0),
        };

        let mask_arr = mask.to_array();
        let scores_arr = scores.to_array();

        for i in 0..8 {
            if mask_arr[i] != 0.0 {
                self.push_single(chunk_idx * 8 + i, scores_arr[i]);
            }
        }
    }

    // Like push_chunk, but also applies an optional row mask for the 8-lane block
    pub fn push_chunk_masked(&mut self, chunk_idx: usize, scores: f32x8, rowmask: Option<[bool; 8]>) {
        if self.k == 0 {
            return;
        }

        let threshold_mask = match self.get_effective_threshold() {
            Some((threshold, cmp)) => filter_simd(scores, threshold, &cmp),
            None => f32x8::splat(1.0),
        };

        let tmask = threshold_mask.to_array();
        let smask = rowmask.unwrap_or([true; 8]);
        let scores_arr = scores.to_array();

        for i in 0..8 {
            if smask[i] && tmask[i] != 0.0 {
                self.push_single(chunk_idx * 8 + i, scores_arr[i]);
            }
        }
    }

    pub fn push_scalars(&mut self, scores: Vec<(usize, f32)>) {
        if self.k == 0 {
            return;
        }

        // Filter based on criteria
        let filtered: Vec<(usize, f32)> = match &self.filter {
            Some((threshold, cmp)) => scores
                .into_iter()
                .filter(|&(_, score)| match cmp {
                    Cmp::Lt => score < *threshold,
                    Cmp::Gt => score > *threshold,
                    Cmp::Leq => score <= *threshold,
                    Cmp::Geq => score >= *threshold,
                    Cmp::Eq => score == *threshold,
                })
                .collect(),
            None => scores,
        };

        // Push filtered scores
        filtered.into_iter().for_each(|(idx, score)| {
            self.push_single(idx, score);
        });
    }

    fn push_single(&mut self, idx: usize, score: f32) {
        match self.buffer.len() == self.k {
            true => {
                // Buffer full - check if we should insert
                let should_insert = match &self.take_type {
                    TakeType::Min => score < self.threshold,
                    TakeType::Max => score > self.threshold,
                };

                if should_insert {
                    let pos = self.find_insert_position(score);
                    self.buffer.insert(pos, (idx, score));
                    self.buffer.pop();
                    self.threshold = self.buffer[self.k - 1].1;
                    self.update_effective_threshold();
                }
            }
            false => {
                // Buffer not full - always insert
                self.buffer.push((idx, score));
                self.is_sorted = false;

                if self.buffer.len() == self.k {
                    self.sort();
                    self.threshold = self.buffer[self.k - 1].1;
                    self.update_effective_threshold();
                }
            }
        }
    }

    fn find_insert_position(&self, score: f32) -> usize {
        self.buffer
            .binary_search_by(|probe| match &self.take_type {
                TakeType::Min => probe.1.partial_cmp(&score).unwrap(),
                TakeType::Max => score.partial_cmp(&probe.1).unwrap(),
            })
            .unwrap_or_else(|e| e)
    }

    fn sort(&mut self) {
        if !self.is_sorted {
            let cmp_fn = match &self.take_type {
                TakeType::Min => {
                    |a: &(usize, f32), b: &(usize, f32)| a.1.partial_cmp(&b.1).unwrap()
                }
                TakeType::Max => {
                    |a: &(usize, f32), b: &(usize, f32)| b.1.partial_cmp(&a.1).unwrap()
                }
            };
            self.buffer.sort_unstable_by(cmp_fn);
            self.is_sorted = true;
        }
    }

    pub fn into_sorted_vec(mut self) -> Vec<(usize, f32)> {
        self.sort();
        self.buffer
    }
}
