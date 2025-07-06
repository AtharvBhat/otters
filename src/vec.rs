use std::iter::zip;

#[derive(Debug, Clone)]
pub struct RowAlignedVecs {
    vectors: Vec<Vec<f32>>,
    dim: usize,
    norms: Vec<f32>,
    n_vecs: usize,
}

pub enum ComparisonMethods {
    Cosine,
    Euclidean,
}

impl RowAlignedVecs {
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dim,
            norms: Vec::new(),
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
        self.vectors.push(vector);
        self.norms.push(norm);
        self.n_vecs += 1;
        Ok(())
    }

    pub fn add_vectors(&mut self, vectors: Vec<Vec<f32>>) -> Result<(), String> {
        vectors.iter().try_for_each(|x| self.add_vector(x.to_vec()))
    }

    // compare cosine similarity of any two generic row aligned vector
    pub fn cosine_generic(self, vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();
        zip(vec1, vec2)
            .map(|(x, y)| x / norm1 * y / norm2)
            .sum::<f32>()
    }

    // compare cosine similarity of a generic vector with stored vectors
    pub fn cosine(&self, vec_l: &Vec<f32>, vec_l_norm: &f32, vec_r: &Vec<f32>) -> f32 {
        let norm_r: f32 = vec_r.iter().map(|x| x * x).sum::<f32>().sqrt();
        zip(vec_l, vec_r)
            .map(|(x, y)| x / vec_l_norm * y / norm_r)
            .sum()
    }

    pub fn euclidean(&self, vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
        zip(vec1, vec2)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    pub fn compare(
        &self,
        compare_idx: usize,
        vec_r: &Vec<f32>,
        method: ComparisonMethods,
    ) -> Result<f32, String> {
        if compare_idx >= self.n_vecs {
            return Err(format!(
                "Comparison Index {} out of range {}",
                compare_idx, self.n_vecs
            ));
        }
        let result = match method {
            ComparisonMethods::Cosine => {
                self.cosine(&self.vectors[compare_idx], &self.norms[compare_idx], vec_r)
            }
            ComparisonMethods::Euclidean => self.euclidean(&self.vectors[compare_idx], vec_r),
        };
        Ok(result)
    }

    pub fn compare_all(&self, vec_r: &Vec<f32>, method: ComparisonMethods) -> Vec<f32> {
        match method {
            ComparisonMethods::Cosine => zip(&self.vectors, &self.norms)
                .map(|(x, n)| self.cosine(x, n, vec_r))
                .collect(),
            ComparisonMethods::Euclidean => self
                .vectors
                .iter()
                .map(|x| self.euclidean(x, vec_r))
                .collect(),
        }
    }
}
