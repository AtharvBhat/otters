#![allow(unused)]
use wide::*;

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

#[derive(Debug)]
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
        }
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

    fn filter_scores<'b>(&self, mut scores: &'b mut Vec<(usize, f32)>) -> &'b Vec<(usize, f32)> {
        if let Some((threshold, cmp)) = &self.filter_criteria {
            scores.retain(|&(_, score)| match cmp {
                Cmp::Lt => score < *threshold,
                Cmp::Gt => score > *threshold,
                Cmp::Leq => score <= *threshold,
                Cmp::Geq => score >= *threshold,
                Cmp::Eq => score == *threshold,
            });
        }
        scores
    }

    fn filter_and_merge_results(
        &self,
        results: &mut Vec<Vec<(usize, f32)>>,
        local_results: &mut Vec<Vec<(usize, f32)>>,
    ) {
        results.iter_mut().enumerate().for_each(|(i, result)| {
            let filtered_res = match self.filter_criteria {
                None => &local_results[i],
                Some(_) => self.filter_scores(&mut local_results[i]),
            };
            filtered_res.iter().for_each(|res| {
                update_top_k(
                    result,
                    *res,
                    self.take_count.unwrap(),
                    self.take_type.as_ref().unwrap(),
                );
            });
        });
    }

    fn merge_results_to_global(&self, results: &Vec<Vec<(usize, f32)>>) -> Vec<(usize, f32)> {
        let mut merged_results: Vec<(usize, f32)> = Vec::new();
        results.iter().for_each(|ress| {
            ress.iter().for_each(|res| {
                update_top_k(
                    &mut merged_results,
                    *res,
                    self.take_count.unwrap(),
                    self.take_type.as_ref().unwrap(),
                );
            });
        });
        merged_results
    }

    pub fn collect(self) -> Result<Vec<Vec<(usize, f32)>>, String> {
        self.validate()?;

        let search_vectors: &[Vec<f32>] = self.vector_store.unwrap().vectors.as_ref();
        let search_vectors_inv_norms: &[f32] = self.vector_store.unwrap().inv_norms.as_ref();
        let query_vectors = self.query_vectors.as_ref().unwrap();
        let query_vectors_inv_norms = self.query_vectors_inv_norms.as_ref().unwrap();
        let search_metric = self.search_metric.as_ref().unwrap();

        let mut results = match &self.take_count {
            None => vec![Vec::new(); query_vectors.len()],
            Some(count) => {
                vec![Vec::with_capacity(*count); query_vectors.len()]
            }
        };

        search_vectors
            .iter()
            .enumerate()
            .for_each(|(search_idx, search_vec)| {
                let search_inv_norm = search_vectors_inv_norms[search_idx];
                let mut search_res = vec![Vec::new(); query_vectors.len()];
                query_vectors
                    .iter()
                    .enumerate()
                    .for_each(|(query_idx, query_vec)| {
                        let score = match search_metric {
                            Metric::Cosine => cosine_similarity(
                                query_vec,
                                search_vec,
                                query_vectors_inv_norms[query_idx],
                                search_inv_norm,
                            ),
                            Metric::Euclidean => euclidean_distance_squared(query_vec, search_vec),
                            Metric::DotProduct => dot_product(query_vec, search_vec),
                        };
                        search_res[query_idx].push((search_idx, score));
                    });
                self.filter_and_merge_results(&mut results, &mut search_res);
            });

        match self.take_scope {
            TakeScope::Global => {
                let merged_results = self.merge_results_to_global(&results);
                results = vec![merged_results];
            }
            _ => {}
        }
        Ok(results)
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
        }
    }
}

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

pub fn update_top_k(
    current: &mut Vec<(usize, f32)>,
    new_entry: (usize, f32),
    k: usize,
    cmp: &TakeType,
) {
    if k == 0 {
        return;
    }

    // If we haven't reached k items yet, just add and sort
    if current.len() < k {
        current.push(new_entry);

        // Sort when we reach capacity or at the end
        match cmp {
            TakeType::Min => current.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()),
            TakeType::Max => current.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()),
        }
        return;
    }

    // We have k items - check if new entry should be inserted
    let should_insert = match cmp {
        TakeType::Min => new_entry.1 < current[k - 1].1,
        TakeType::Max => new_entry.1 > current[k - 1].1,
    };

    if should_insert {
        // Binary search for insertion position
        let pos = current
            .binary_search_by(|probe| match cmp {
                TakeType::Min => probe.1.partial_cmp(&new_entry.1).unwrap(),
                TakeType::Max => new_entry.1.partial_cmp(&probe.1).unwrap(),
            })
            .unwrap_or_else(|e| e);

        current.insert(pos, new_entry);
        current.pop(); // Remove the worst element
    }
}
