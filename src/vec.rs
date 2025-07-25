#![allow(unused)]
use wide::*;

pub enum ComparisonType {
    Min,
    Max,
}
#[derive(Debug)]
#[repr(align(64))]
pub struct RowAlignedVecs {
    vectors: Vec<Vec<f32>>,
    dim: usize,
    inv_norms: Vec<f32>,
    n_vecs: usize,
}

impl RowAlignedVecs {
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dim,
            inv_norms: Vec::new(),
            n_vecs: 0,
        }
    }

    pub fn add_vector(&mut self, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dim {
            return Err(format!(
                "Input vector length {} does not match expected dimension {}",
                vector.len(),
                self.dim
            ));
        }
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        self.vectors.push(vector.clone());
        self.inv_norms.push(1.0 / norm);
        self.n_vecs += 1;
        Ok(())
    }

    pub fn add_vectors(&mut self, vectors: Vec<Vec<f32>>) -> Result<(), String> {
        vectors.iter().try_for_each(|x| self.add_vector(x.to_vec()))
    }

    pub fn dot_product(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        assert!(
            vec1.len() == vec2.len(),
            "Mismatched input lengths, vec1: {} != vec2 {}",
            vec1.len(),
            vec2.len()
        );

        let v1_chunks = vec1.chunks_exact(8);
        let v2_chunks = vec2.chunks_exact(8);

        let v1_rest = v1_chunks.remainder();
        let v2_rest = v2_chunks.remainder();

        let mut res = v1_chunks
            .map(f32x8::from)
            .zip(v2_chunks.map(f32x8::from))
            .fold(f32x8::splat(0.0), |x, (a, b)| a.mul_add(b, x))
            .reduce_add();

        v1_rest.iter().zip(v2_rest).for_each(|(a, b)| {
            res += a * b;
        });

        res
    }

    pub fn search_vec_cosine(&self, query: &[f32]) -> impl Iterator<Item = (usize, f32)> {
        if query.len() != self.dim {
            panic!(
                "Query vector length {} does not match expected dimension {}",
                query.len(),
                self.dim
            );
        }

        let inv_query_norm = 1.0 / query.iter().map(|x| x * x).sum::<f32>().sqrt();

        (0..self.n_vecs)
            .map(move |idx| {
                self.dot_product(&self.vectors[idx], query) * inv_query_norm * self.inv_norms[idx]
            })
            .enumerate()
    }

    pub fn euclidean_distance_squared(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        assert!(
            vec1.len() == vec2.len(),
            "Mismatched input lengths, vec1: {} != vec2 {}",
            vec1.len(),
            vec2.len()
        );

        let v1_chunks = vec1.chunks_exact(8);
        let v2_chunks = vec2.chunks_exact(8);

        let v1_rest = v1_chunks.remainder();
        let v2_rest = v2_chunks.remainder();

        let mut res = v1_chunks
            .map(f32x8::from)
            .zip(v2_chunks.map(f32x8::from))
            .fold(f32x8::splat(0.0), |acc, (a, b)| {
                let diff = a - b;
                diff.mul_add(diff, acc)
            })
            .reduce_add();

        for (a, b) in v1_rest.iter().zip(v2_rest) {
            let diff = a - b;
            res += diff * diff;
        }

        res
    }

    pub fn search_vec_euclidean(&self, query: &[f32]) -> impl Iterator<Item = (usize, f32)> {
        if query.len() != self.dim {
            panic!(
                "Query vector length {} does not match expected dimension {}",
                query.len(),
                self.dim
            );
        }

        (0..self.n_vecs)
            .map(move |idx| self.euclidean_distance_squared(&self.vectors[idx], query))
            .enumerate()
    }
}

pub trait TopKIterator {
    fn take_max(self, k: usize) -> Vec<(usize, f32)>;
    fn take_min(self, k: usize) -> Vec<(usize, f32)>;
}

impl<T> TopKIterator for T
where
    T: Iterator<Item = (usize, f32)>,
{
    fn take_max(self, k: usize) -> Vec<(usize, f32)> {
        top_k(k, self, ComparisonType::Max)
    }

    fn take_min(self, k: usize) -> Vec<(usize, f32)> {
        top_k(k, self, ComparisonType::Min)
    }
}

pub fn top_k(
    k: usize,
    iter: impl Iterator<Item = (usize, f32)>,
    cmp: ComparisonType,
) -> Vec<(usize, f32)> {
    if k == 0 {
        return Vec::new();
    }

    let mut iter = iter;
    let mut buffer: Vec<(usize, f32)> = Vec::with_capacity(k);

    for _ in 0..k {
        if let Some(item) = iter.next() {
            buffer.push(item);
        } else {
            // Sort and return if we don't have enough elements
            match cmp {
                ComparisonType::Min => buffer
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)),
                ComparisonType::Max => buffer
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)),
            }
            return buffer;
        }
    }

    // Sort the initial buffer according to comparison type
    match cmp {
        ComparisonType::Min => {
            buffer.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        }
        ComparisonType::Max => {
            buffer.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        }
    }

    // Process remaining elements
    for (idx, val) in iter {
        let should_insert = match cmp {
            ComparisonType::Min => val < buffer[k - 1].1,
            ComparisonType::Max => val > buffer[k - 1].1,
        };

        if should_insert {
            // Find insertion position using binary search
            let pos = buffer
                .binary_search_by(|probe| match cmp {
                    ComparisonType::Min => probe
                        .1
                        .partial_cmp(&val)
                        .unwrap_or(std::cmp::Ordering::Equal),
                    ComparisonType::Max => val
                        .partial_cmp(&probe.1)
                        .unwrap_or(std::cmp::Ordering::Equal),
                })
                .unwrap_or_else(|e| e);

            // Insert and remove the last element
            buffer.insert(pos, (idx, val));
            buffer.pop();
        }
    }

    buffer
}
