#![allow(unused)]
use crate::vec_compute::{TopKCollector, calculate_scores_chunk};
pub use crate::vec_compute::{cosine_similarity, dot_product, euclidean_distance_squared};
use wide::*;
use bitvec::prelude::BitVec;

#[derive(Debug)]
pub enum Metric {
    Cosine,
    Euclidean,
    DotProduct,
}

#[derive(Debug)]
pub enum TakeType {
    Min,
    Max,
}

#[derive(Debug)]
pub enum TakeScope {
    Local,
    Global,
}

#[derive(Debug, Clone)]
pub enum Cmp {
    Lt,
    Gt,
    Leq,
    Geq,
    Eq, // Idk why you would ever use this
}

#[derive(Debug)]
pub struct VecQueryPlan<'a> {
    query_vectors: Option<Vec<Vec<f32>>>,
    query_vectors_inv_norms: Option<Vec<f32>>,
    search_metric: Option<Metric>,
    filter_criteria: Option<(f32, Cmp)>,
    take_type: Option<TakeType>,
    take_count: Option<usize>,
    take_scope: TakeScope,
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
            take_scope: TakeScope::Local,
            vector_store: None,
            error: None,
            row_mask: None,
        }
    }

    pub fn with_vector_store(mut self, store: &'a VecStore) -> Self {
        if self.error.is_some() {
            return self;
        }
        self.vector_store = Some(store);
        self
    }

    pub fn with_query_vectors(mut self, queries: impl Into<QueryBatch>) -> Self {
        if self.error.is_some() {
            return self;
        }
        let query_batch = queries.into();

        let inv_norms: Vec<f32> = query_batch
            .queries
            .iter()
            .map(|vec| {
                let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 / norm
            })
            .collect();
        self.query_vectors = Some(query_batch.queries);
        self.query_vectors_inv_norms = Some(inv_norms);
        self
    }

    pub fn with_metric(mut self, metric: Metric) -> Self {
        if self.error.is_some() {
            return self;
        }
        self.search_metric = Some(metric);
        self
    }

    pub fn with_row_mask(mut self, mask: BitVec) -> Self {
        if self.error.is_some() {
            return self;
        }
        self.row_mask = Some(mask);
        self
    }

    pub fn filter(mut self, score: f32, cmp: Cmp) -> Self {
        if self.error.is_some() {
            return self;
        }
        self.filter_criteria = Some((score, cmp));
        self
    }

    pub fn take(mut self, count: usize) -> Self {
        if self.error.is_some() {
            return self;
        }
        self.take_count = Some(count);
        match self.search_metric {
            Some(Metric::Cosine) => self.take_type = Some(TakeType::Max),
            Some(Metric::Euclidean) => self.take_type = Some(TakeType::Min),
            Some(Metric::DotProduct) => self.take_type = Some(TakeType::Max),
            None => {}
        }
        self
    }

    pub fn take_global(mut self, count: usize) -> Self {
        if self.error.is_some() {
            return self;
        }
        self.take_count = Some(count);
        self.take_scope = TakeScope::Global;
        match self.search_metric {
            Some(Metric::Cosine) => self.take_type = Some(TakeType::Max),
            Some(Metric::Euclidean) => self.take_type = Some(TakeType::Min),
            Some(Metric::DotProduct) => self.take_type = Some(TakeType::Max),
            None => {}
        }
        self
    }

    pub fn take_min(mut self, count: usize) -> Self {
        if self.error.is_some() {
            return self;
        }
        self.take_count = Some(count);
        self.take_type = Some(TakeType::Min);
        self.take_scope = TakeScope::Local;
        self
    }

    pub fn take_min_global(mut self, count: usize) -> Self {
        if self.error.is_some() {
            return self;
        }
        self.take_count = Some(count);
        self.take_type = Some(TakeType::Min);
        self.take_scope = TakeScope::Global;
        self
    }

    pub fn take_max(mut self, count: usize) -> Self {
        if self.error.is_some() {
            return self;
        }
        self.take_count = Some(count);
        self.take_type = Some(TakeType::Max);
        self.take_scope = TakeScope::Local;
        self
    }

    pub fn take_max_global(mut self, count: usize) -> Self {
        if self.error.is_some() {
            return self;
        }
        self.take_count = Some(count);
        self.take_type = Some(TakeType::Max);
        self.take_scope = TakeScope::Global;
        self
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

    pub fn collect(self) -> Result<Vec<Vec<(usize, f32)>>, String> {
        self.validate()?;

        let vector_store = self.vector_store.unwrap();
        let query_vectors = self.query_vectors.as_ref().unwrap();
        let query_vectors_inv_norms = self.query_vectors_inv_norms.as_ref().unwrap();
        let search_metric = self.search_metric.as_ref().unwrap();
        let take_count = self.take_count.unwrap_or(vector_store.n_vecs);
        let take_type = self.take_type.as_ref().unwrap_or(&TakeType::Max);
        let num_queries = query_vectors.len();

        // Initialize collectors
        let mut collectors: Vec<TopKCollector> = match self.take_scope {
            TakeScope::Global => vec![TopKCollector::new(
                take_count,
                take_type,
                self.filter_criteria.as_ref(),
            )],
            TakeScope::Local => (0..num_queries)
                .map(|_| TopKCollector::new(take_count, take_type, self.filter_criteria.as_ref()))
                .collect(),
        };

        // Process chunks
        let vec_chunks = vector_store.vectors.chunks_exact(8);
        let inv_chunks = vector_store.inv_norms.chunks_exact(8);

        vec_chunks
            .zip(inv_chunks)
            .enumerate()
            .for_each(|(chunk_idx, (vec_chunk, inv_chunk))| {
                // Calculate scores for all queries for this chunk
                let scores_per_query = calculate_scores_chunk(
                    query_vectors,
                    query_vectors_inv_norms,
                    vec_chunk,
                    inv_chunk,
                    search_metric,
                );

                // Push to appropriate collectors
                match self.take_scope {
                    TakeScope::Global => {
                        scores_per_query.into_iter().for_each(|scores| {
                            // Prepare optional row mask for this 8-lane block
                            let block_mask = self.row_mask.as_ref().map(|rm| {
                                let mut m = [true; 8];
                                let base = chunk_idx * 8;
                                for i in 0..8 { m[i] = rm.get(base + i).map(|br| *br).unwrap_or(true); }
                                m
                            });
                            collectors[0].push_chunk_masked(chunk_idx, scores, block_mask);
                        });
                    }
                    TakeScope::Local => {
                        scores_per_query
                            .into_iter()
                            .enumerate()
                            .for_each(|(q_idx, scores)| {
                                let block_mask = self.row_mask.as_ref().map(|rm| {
                                    let mut m = [true; 8];
                                    let base = chunk_idx * 8;
                                    for i in 0..8 { m[i] = rm.get(base + i).map(|br| *br).unwrap_or(true); }
                                    m
                                });
                                collectors[q_idx].push_chunk_masked(chunk_idx, scores, block_mask);
                            });
                    }
                }
            });

        // Process remainder
        let remainder_start = (vector_store.n_vecs / 8) * 8;
        if remainder_start < vector_store.n_vecs {
            let remainder_vecs = &vector_store.vectors[remainder_start..];
            let remainder_inv_norms = &vector_store.inv_norms[remainder_start..];

            query_vectors
                .iter()
                .zip(query_vectors_inv_norms.iter())
                .enumerate()
                .for_each(|(q_idx, (query, &query_inv_norm))| {
                    // Collect remainder scores
                    let remainder_scores: Vec<(usize, f32)> = remainder_vecs
                        .iter()
                        .zip(remainder_inv_norms.iter())
                        .enumerate()
                        .map(|(i, (vec, &inv_norm))| {
                            let score = match search_metric {
                                Metric::Cosine => {
                                    cosine_similarity(query, vec, query_inv_norm, inv_norm)
                                }
                                Metric::Euclidean => euclidean_distance_squared(query, vec),
                                Metric::DotProduct => dot_product(query, vec),
                            };
                            (remainder_start + i, score)
                        })
                        .filter(|(idx, _)| {
                            if let Some(rm) = &self.row_mask { rm.get(*idx).map(|br| *br).unwrap_or(true) } else { true }
                        })
                        .collect();

                    // Push to appropriate collector
                    match self.take_scope {
                        TakeScope::Global => collectors[0].push_scalars(remainder_scores),
                        TakeScope::Local => collectors[q_idx].push_scalars(remainder_scores),
                    }
                });
        }

        // Return results
        Ok(collectors
            .into_iter()
            .map(|c| c.into_sorted_vec())
            .collect())
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
#[repr(align(64))]
pub struct VecStore {
    vectors: Vec<Vec<f32>>,
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
        self.vectors.push(vector.clone());
        self.inv_norms.push(1.0 / norm);
        self.n_vecs += 1;
        Ok(())
    }

    pub fn add_vectors(&mut self, vectors: Vec<Vec<f32>>) -> Result<(), String> {
        vectors.iter().try_for_each(|x| self.add_vector(x.to_vec()))
    }

    // Move vectors into the store without cloning individual vectors
    pub fn add_vectors_owned(&mut self, vectors: Vec<Vec<f32>>) -> Result<(), String> {
        for vector in vectors.into_iter() {
            if vector.len() != self.dim {
                return Err(format!(
                    "Input vector length {} does not match expected dimension {}",
                    vector.len(),
                    self.dim
                ));
            }
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            self.inv_norms.push(1.0 / norm);
            self.n_vecs += 1;
            self.vectors.push(vector);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.n_vecs
    }

    pub fn query(&self, queries: impl Into<QueryBatch>, metric: Metric) -> VecQueryPlan {
        let query_batch = queries.into();

        let inv_norms: Vec<f32> = query_batch
            .queries
            .iter()
            .map(|vec| {
                let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 / norm
            })
            .collect();

        VecQueryPlan {
            query_vectors: Some(query_batch.queries),
            query_vectors_inv_norms: Some(inv_norms),
            search_metric: Some(metric),
            filter_criteria: None,
            take_type: None,
            take_count: None,
            take_scope: TakeScope::Local,
            vector_store: Some(self),
            error: None,
            row_mask: None,
        }
    }
}
