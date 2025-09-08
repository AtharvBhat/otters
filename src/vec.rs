//! Vector store and query planning primitives.
//!
//! Provides an in-memory row-major `VecStore` and a builder-style `VecQueryPlan`
//! supporting cosine, dot product, and squared euclidean similarity with optional
//! score filtering and row masking. Batch queries are treated as a single search
//! over multiple inputs and return one merged result set.
use crate::vec_compute::TopKCollector;
pub use crate::vec_compute::{cosine_similarity, dot_product, euclidean_distance_squared};
use bitvec::prelude::BitVec;
use wide::f32x8;

#[derive(Debug, Clone, Copy)]
pub enum Metric {
    Cosine,
    Euclidean,
    DotProduct,
}

#[derive(Debug, Clone, Copy)]
pub enum TakeType {
    Min,
    Max,
}

#[derive(Debug, Clone)]
pub enum Cmp {
    Lt,
    Gt,
    Lte,
    Gte,
    Eq, // Idk why you would ever use this
}

/// Rich result type for vector (and metadata) queries.
/// Provides a stable, extensible surface instead of raw (index, score) tuples.
/// Additional metadata fields (e.g. chunk id, original vector reference, column slices)
/// can be added later without breaking tuple-based callers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub index: usize,
    pub score: f32,
}

impl std::fmt::Display for SearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{} score={:.6}", self.index, self.score)
    }
}

impl From<(usize, f32)> for SearchResult {
    fn from(t: (usize, f32)) -> Self {
        SearchResult {
            index: t.0,
            score: t.1,
        }
    }
}

#[derive(Debug)]
pub struct VecQueryPlan<'a> {
    query_vectors: Option<Vec<Vec<f32>>>,
    query_vectors_inv_norms: Option<Vec<f32>>,
    search_metric: Option<Metric>,
    filter_criteria: Option<(f32, Cmp)>,
    take_type: Option<TakeType>,
    take_count: Option<usize>,
    vector_store: Option<&'a VecStore>,
    error: Option<String>,
    row_mask: Option<BitVec>,
}

impl<'a> VecQueryPlan<'a> {
    pub fn new() -> Self {
        Self {
            query_vectors: None,
            query_vectors_inv_norms: None,
            search_metric: None,
            filter_criteria: None,
            take_type: None,
            take_count: None,
            vector_store: None,
            error: None,
            row_mask: None,
        }
    }

    #[inline]
    fn map_ok(mut self, f: impl FnOnce(&mut Self)) -> Self {
        if self.error.is_none() {
            f(&mut self);
        }
        self
    }

    #[inline]
    fn infer_default_take_type(metric: &Metric) -> TakeType {
        match metric {
            Metric::Euclidean => TakeType::Min,
            Metric::Cosine | Metric::DotProduct => TakeType::Max,
        }
    }

    /// Generic helper powering all public `take*` variants. Determines count
    /// and (optionally) explicit `TakeType`. If `take_type` is None we infer one
    /// from the already-selected metric (if present).
    fn take_with_options(mut self, count: usize, take_type: Option<TakeType>) -> Self {
        if self.error.is_some() {
            return self;
        }
        self.take_count = Some(count);
        if let Some(tt) = take_type {
            self.take_type = Some(tt);
        } else if self.take_type.is_none()
            && let Some(metric) = self.search_metric
        {
            self.take_type = Some(Self::infer_default_take_type(&metric));
        }
        self
    }

    pub fn with_vector_store(self, store: &'a VecStore) -> Self {
        self.map_ok(|s| s.vector_store = Some(store))
    }

    pub fn with_query_vectors(self, queries: impl Into<QueryBatch>) -> Self {
        self.map_ok(|s| {
            let query_batch = queries.into();
            let inv_norms: Vec<f32> = query_batch
                .queries
                .iter()
                .map(|vec| {
                    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm != 0.0 { 1.0 / norm } else { 0.0 }
                })
                .collect();
            s.query_vectors = Some(query_batch.queries);
            s.query_vectors_inv_norms = Some(inv_norms);
        })
    }

    pub fn with_metric(self, metric: Metric) -> Self {
        self.map_ok(|s| s.search_metric = Some(metric))
    }

    pub fn with_row_mask(self, mask: BitVec) -> Self {
        self.map_ok(|s| s.row_mask = Some(mask))
    }

    pub fn filter(self, score: f32, cmp: Cmp) -> Self {
        self.map_ok(|s| s.filter_criteria = Some((score, cmp)))
    }

    pub fn take(self, count: usize) -> Self {
        self.take_with_options(count, None)
    }

    pub fn take_min(self, count: usize) -> Self {
        self.take_with_options(count, Some(TakeType::Min))
    }

    pub fn take_max(self, count: usize) -> Self {
        self.take_with_options(count, Some(TakeType::Max))
    }

    fn validate(&self) -> Result<(), String> {
        if let Some(ref error) = self.error {
            return Err(error.clone());
        }
        if self.query_vectors.is_none() || self.query_vectors_inv_norms.is_none() {
            return Err("Query vectors or their norms are not set".to_string());
        }
        if self.search_metric.is_none() {
            return Err("Search metric is not set".to_string());
        }
        if self.vector_store.is_none() {
            return Err("Vector store is not set".to_string());
        }

        // Validate query vectors dimensions and content
        let query_vectors = self.query_vectors.as_ref().unwrap();
        let vector_store = self.vector_store.as_ref().unwrap();

        if query_vectors.is_empty() {
            return Err("No queries provided".to_string());
        }

        for query in query_vectors {
            if query.len() != vector_store.dim {
                return Err(format!(
                    "Query vector length {} does not match expected dimension {}",
                    query.len(),
                    vector_store.dim
                ));
            }
        }

        Ok(())
    }

    pub fn collect(self) -> Result<Vec<SearchResult>, String> {
        self.validate()?;

        let vector_store = self.vector_store.unwrap();
        let query_vectors = self.query_vectors.as_ref().unwrap();
        let query_vectors_inv_norms = self.query_vectors_inv_norms.as_ref().unwrap();
        let search_metric = self.search_metric.as_ref().unwrap();
        let take_count = self.take_count.unwrap_or(vector_store.n_vecs);
        let take_type = self.take_type.as_ref().unwrap_or(&TakeType::Max);
        let _num_queries = query_vectors.len();

        // Single global collector aggregating across all queries
        let mut collector =
            TopKCollector::new(take_count, take_type, self.filter_criteria.as_ref());

        // Process chunks: 8 rows per block
        let full_chunks = vector_store.n_vecs / 8;
        vector_store
            .inv_norms
            .chunks_exact(8)
            .take(full_chunks)
            .enumerate()
            .for_each(|(chunk_idx, inv_chunk)| {
                let base_row = chunk_idx * 8;
                // Compute block row mask once per chunk
                let bm: Option<[bool; 8]> = self.row_mask.as_ref().map(|rm| {
                    let mut m = [true; 8];
                    for (i, mi) in m.iter_mut().enumerate() {
                        *mi = rm.get(base_row + i).map(|br| *br).unwrap_or(true);
                    }
                    m
                });

                // Reuse preallocated scratch for scores across queries
                let mut scratch = [0.0f32; 8];

                // Compute and push scores per query without allocating a Vec
                query_vectors
                    .iter()
                    .zip(query_vectors_inv_norms.iter())
                    .for_each(|(query, &query_inv_norm)| {
                        for i in 0..8 {
                            if let Some(mask) = bm.as_ref()
                                && !mask[i]
                            {
                                continue;
                            }
                            let row = base_row + i;
                            let start = row * vector_store.dim;
                            let v = &vector_store.vectors[start..start + vector_store.dim];
                            scratch[i] = match search_metric {
                                Metric::Cosine => {
                                    cosine_similarity(query, v, query_inv_norm, inv_chunk[i])
                                }
                                Metric::Euclidean => euclidean_distance_squared(query, v),
                                Metric::DotProduct => dot_product(query, v),
                            };
                        }
                        let scores = f32x8::from(scratch);
                        collector.push_chunk_masked(chunk_idx, scores, bm);
                    });
            });

        // Process remainder
        let remainder_start = full_chunks * 8;
        if remainder_start < vector_store.n_vecs {
            let remainder_inv_norms = &vector_store.inv_norms[remainder_start..];

            query_vectors
                .iter()
                .zip(query_vectors_inv_norms.iter())
                .for_each(|(query, &query_inv_norm)| {
                    let scores = remainder_inv_norms
                        .iter()
                        .enumerate()
                        .map(|(i, &inv_norm)| {
                            let row = remainder_start + i;
                            let start = row * vector_store.dim;
                            let vec = &vector_store.vectors[start..start + vector_store.dim];
                            let s = match search_metric {
                                Metric::Cosine => {
                                    cosine_similarity(query, vec, query_inv_norm, inv_norm)
                                }
                                Metric::Euclidean => euclidean_distance_squared(query, vec),
                                Metric::DotProduct => dot_product(query, vec),
                            };
                            (row, s)
                        })
                        .filter(|(idx, _)| {
                            self.row_mask
                                .as_ref()
                                .map(|rm| rm.get(*idx).map(|b| *b).unwrap_or(true))
                                .unwrap_or(true)
                        })
                        .collect();
                    collector.push_scalars(scores);
                });
        }

        // Return results
        Ok(collector
            .into_sorted_vec()
            .into_iter()
            .map(SearchResult::from)
            .collect::<Vec<_>>())
    }
}

impl<'a> Default for VecQueryPlan<'a> {
    fn default() -> Self {
        Self::new()
    }
}
// Convenience trait to accept both single and batch queries
pub struct QueryBatch {
    queries: Vec<Vec<f32>>,
}

impl From<Vec<f32>> for QueryBatch {
    fn from(query: Vec<f32>) -> Self {
        QueryBatch {
            queries: vec![query],
        }
    }
}

impl From<Vec<Vec<f32>>> for QueryBatch {
    fn from(queries: Vec<Vec<f32>>) -> Self {
        QueryBatch { queries }
    }
}

#[derive(Debug)]
pub struct VecStore {
    vectors: Vec<f32>,
    dim: usize,
    inv_norms: Vec<f32>,
    n_vecs: usize,
}

impl VecStore {
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
        self.vectors.extend_from_slice(&vector);
        let inv = if norm != 0.0 { 1.0 / norm } else { 0.0 };
        self.inv_norms.push(inv);
        self.n_vecs += 1;
        Ok(())
    }

    pub fn add_vectors(&mut self, vectors: Vec<Vec<f32>>) -> Result<(), String> {
        vectors.iter().try_for_each(|x| self.add_vector(x.to_vec()))
    }

    pub fn len(&self) -> usize {
        self.n_vecs
    }

    pub fn is_empty(&self) -> bool {
        self.n_vecs == 0
    }

    pub fn query(&self, queries: impl Into<QueryBatch>, metric: Metric) -> VecQueryPlan<'_> {
        let query_batch = queries.into();

        let inv_norms: Vec<f32> = query_batch
            .queries
            .iter()
            .map(|vec| {
                let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm != 0.0 { 1.0 / norm } else { 0.0 }
            })
            .collect();

        VecQueryPlan {
            query_vectors: Some(query_batch.queries),
            query_vectors_inv_norms: Some(inv_norms),
            search_metric: Some(metric),
            filter_criteria: None,
            take_type: None,
            take_count: None,

            vector_store: Some(self),
            error: None,
            row_mask: None,
        }
    }
}
