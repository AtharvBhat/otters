#![allow(unused)]

pub struct VecSearchResult<I>
where
    I: Iterator<Item = (usize, f32)>,
{
    iter: I,
}

impl<I> VecSearchResult<I>
where
    I: Iterator<Item = (usize, f32)>,
{
    pub fn new(iter: I) -> Self {
        Self { iter }
    }

    pub fn take_min(self, k: usize) -> Vec<(usize, f32)> {
        take_min(k, self.iter)
    }

    pub fn take_max(self, k: usize) -> Vec<(usize, f32)> {
        take_max(k, self.iter)
    }
}

#[derive(Debug, Clone)]
pub struct RowAlignedVecs {
    vectors: Vec<Vec<f32>>,
    dim: usize,
    inv_norms: Vec<f32>,
    n_vecs: usize,
}

#[derive(Debug, Clone)]
pub struct ColumnAlignedVecs {
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

    pub fn search_vec_cosine<'a>(
        &'a self,
        vec_r: &'a [f32],
    ) -> VecSearchResult<impl Iterator<Item = (usize, f32)> + 'a> {
        let vec_r_inv_norm = 1.0 / vec_r.iter().map(|x| x * x).sum::<f32>().sqrt();

        let iter = self.vectors.iter().enumerate().map(move |(i, vec_l)| {
            let dot_product = vec_l
                .iter()
                .zip(vec_r.iter())
                .map(|(&x, &y)| x * y)
                .sum::<f32>();

            let similarity = dot_product * self.inv_norms[i] * vec_r_inv_norm;
            (i, similarity)
        });

        VecSearchResult::new(iter)
    }

    pub fn search_vec_euclidean<'a>(
        &'a self,
        vec_r: &'a [f32],
    ) -> VecSearchResult<impl Iterator<Item = (usize, f32)> + 'a> {
        let iter = self.vectors.iter().enumerate().map(move |(i, vec_l)| {
            let distance = vec_l
                .iter()
                .zip(vec_r)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>();
            (i, distance)
        });
        VecSearchResult::new(iter)
    }

    pub fn take_min(
        &self,
        k: usize,
        iter: impl Iterator<Item = (usize, f32)>,
    ) -> Vec<(usize, f32)> {
        take_min(k, iter)
    }

    pub fn take_max(
        &self,
        k: usize,
        iter: impl Iterator<Item = (usize, f32)>,
    ) -> Vec<(usize, f32)> {
        take_max(k, iter)
    }
}

impl ColumnAlignedVecs {
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: vec![Vec::new(); dim],
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
        let inv_norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        for (i, &value) in vector.clone().iter().enumerate() {
            self.vectors[i].push(value);
        }
        self.inv_norms.push(1.0 / inv_norm);
        self.n_vecs += 1;
        Ok(())
    }

    pub fn add_vectors(&mut self, vectors: Vec<Vec<f32>>) -> Result<(), String> {
        vectors.iter().try_for_each(|x| self.add_vector(x.to_vec()))
    }

    pub fn search_vec_cosine<'a>(
        &'a self,
        vec_r: &'a [f32],
    ) -> VecSearchResult<impl Iterator<Item = (usize, f32)> + 'a> {
        let vec_r_inv_norm = 1.0 / vec_r.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut similarities = vec![0.0f32; self.n_vecs];

        self.vectors
            .iter()
            .zip(vec_r.iter())
            .for_each(|(dim_values, &query_val)| {
                similarities
                    .iter_mut()
                    .zip(dim_values.iter())
                    .for_each(|(sim, &stored_val)| {
                        *sim += stored_val * query_val;
                    });
            });

        let iter = similarities
            .into_iter()
            .enumerate()
            .zip(&self.inv_norms)
            .map(move |((i, sim), &inv_norm)| (i, sim * inv_norm * vec_r_inv_norm));
        VecSearchResult::new(iter)
    }

    pub fn search_vec_euclidean<'a>(
        &'a self,
        vec_r: &'a [f32],
    ) -> VecSearchResult<impl Iterator<Item = (usize, f32)> + 'a> {
        let mut dist = vec![0.0f32; self.n_vecs];

        // Use iterator pattern for clean, functional style
        self.vectors
            .iter()
            .zip(vec_r.iter())
            .for_each(|(dim_values, &query_val)| {
                dist.iter_mut()
                    .zip(dim_values.iter())
                    .for_each(|(distance, &stored_val)| {
                        *distance += (stored_val - query_val).powi(2);
                    });
            });

        VecSearchResult::new(dist.into_iter().enumerate().map(|(i, d)| (i, d)))
    }

    pub fn take_min(
        &self,
        k: usize,
        iter: impl Iterator<Item = (usize, f32)>,
    ) -> Vec<(usize, f32)> {
        take_min(k, iter)
    }

    pub fn take_max(
        &self,
        k: usize,
        iter: impl Iterator<Item = (usize, f32)>,
    ) -> Vec<(usize, f32)> {
        take_max(k, iter)
    }
}

pub fn take_min(k: usize, iter: impl Iterator<Item = (usize, f32)>) -> Vec<(usize, f32)> {
    if k == 0 {
        return Vec::new();
    }

    let mut iter = iter;
    let mut buffer: Vec<(usize, f32)> = Vec::with_capacity(k);

    for _ in 0..k {
        if let Some(item) = iter.next() {
            buffer.push(item);
        } else {
            buffer.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            return buffer;
        }
    }

    // Sort the initial buffer
    buffer.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Process remaining elements
    for (idx, val) in iter {
        if val < buffer[k - 1].1 {
            // Find insertion position using binary search
            let pos = buffer
                .binary_search_by(|probe| {
                    probe
                        .1
                        .partial_cmp(&val)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|e| e);

            // Insert and remove the last element
            buffer.insert(pos, (idx, val));
            buffer.pop();
        }
    }

    buffer
}

pub fn take_max(k: usize, iter: impl Iterator<Item = (usize, f32)>) -> Vec<(usize, f32)> {
    if k == 0 {
        return Vec::new();
    }

    let mut iter = iter;
    let mut buffer: Vec<(usize, f32)> = Vec::with_capacity(k);

    for _ in 0..k {
        if let Some(item) = iter.next() {
            buffer.push(item);
        } else {
            buffer.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return buffer;
        }
    }

    buffer.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, val) in iter {
        if val > buffer[k - 1].1 {
            let pos = buffer
                .binary_search_by(|probe| {
                    val.partial_cmp(&probe.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|e| e);

            buffer.insert(pos, (idx, val));
            buffer.pop();
        }
    }

    buffer
}
