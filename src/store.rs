//! Unified Arrow-based store for vectors and metadata
//!
//! Combines vector search with metadata filtering using Apache Arrow's
//! RecordBatch for true columnar storage. All data (embeddings, metadata)
//! are stored together in a single table.

use crate::col::{Column, ColumnBuilder};
use arrow::array::ArrayRef;
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

/// Unified store for vectors and metadata using Arrow RecordBatch
#[derive(Debug, Clone)]
pub struct ArrowStore {
    /// RecordBatch containing all columns (vectors + metadata)
    batch: RecordBatch,

    /// Name of the vector/embedding column
    vector_column: String,

    /// Dimension of vectors
    dim: i32,

    /// Pre-computed inverse norms for cosine similarity
    inv_norms: Vec<f32>,
}

/// Builder for constructing an ArrowStore
pub struct ArrowStoreBuilder {
    dim: i32,
    vector_column: String,
    vectors: Option<Column>,
    inv_norms: Option<Vec<f32>>,
    metadata: Vec<(String, Column)>,
}

impl ArrowStoreBuilder {
    /// Create a new builder for vectors of given dimension
    pub fn new(dim: i32) -> Self {
        Self {
            dim,
            vector_column: "embeddings".to_string(),
            vectors: None,
            inv_norms: None,
            metadata: Vec::new(),
        }
    }

    /// Set custom name for vector column (default: "embeddings")
    pub fn with_vector_column_name(mut self, name: impl Into<String>) -> Self {
        self.vector_column = name.into();
        self
    }

    /// Add vectors from a Vec<Vec<f32>>
    pub fn with_vectors(mut self, vectors: Vec<Vec<f32>>) -> Result<Self, String> {
        if vectors.is_empty() {
            return Err("Cannot add empty vector list".to_string());
        }

        // Validate dimensions
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != self.dim as usize {
                return Err(format!(
                    "Vector at index {} has dimension {}, expected {}",
                    i,
                    vec.len(),
                    self.dim
                ));
            }
        }

        // Build column
        let mut builder = ColumnBuilder::new_vector(&self.vector_column, self.dim);
        let mut inv_norms = Vec::with_capacity(vectors.len());

        for vec in &vectors {
            builder
                .append_vector(Some(vec.as_slice()))
                .map_err(|e| e.to_string())?;

            // Compute inverse norm
            let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            inv_norms.push(if norm != 0.0 { 1.0 / norm } else { 0.0 });
        }

        self.vectors = Some(builder.collect());
        self.inv_norms = Some(inv_norms);
        Ok(self)
    }

    /// Add a metadata column
    pub fn with_metadata_column(mut self, name: impl Into<String>, column: Column) -> Self {
        self.metadata.push((name.into(), column));
        self
    }

    /// Add multiple metadata columns
    pub fn with_metadata_columns(
        mut self,
        columns: impl IntoIterator<Item = (String, Column)>,
    ) -> Self {
        self.metadata.extend(columns);
        self
    }

    /// Build the final store
    pub fn build(self) -> Result<ArrowStore, String> {
        let vectors = self
            .vectors
            .ok_or_else(|| "Vectors not provided".to_string())?;

        let inv_norms = self
            .inv_norms
            .ok_or_else(|| "Inverse norms not computed".to_string())?;

        let len = vectors.len();

        // Validate all metadata columns have same length
        for (name, col) in &self.metadata {
            if col.len() != len {
                return Err(format!(
                    "Metadata column '{}' has length {}, expected {}",
                    name,
                    col.len(),
                    len
                ));
            }
        }

        // Build schema and arrays for RecordBatch
        let mut fields = Vec::with_capacity(1 + self.metadata.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(1 + self.metadata.len());

        // Add vector column first
        fields.push(vectors.field().clone());
        arrays.push(vectors.array().clone());

        // Add metadata columns
        for (name, col) in self.metadata {
            // Ensure field name matches the key
            let mut field = col.field().clone();
            if field.name() != &name {
                field = Field::new(name, field.data_type().clone(), field.is_nullable());
            }
            fields.push(field);
            arrays.push(col.array().clone());
        }

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| format!("Failed to create RecordBatch: {e}"))?;

        Ok(ArrowStore {
            batch,
            vector_column: self.vector_column,
            dim: self.dim,
            inv_norms,
        })
    }
}

impl ArrowStore {
    /// Create a new builder for given dimension
    pub fn builder(dim: i32) -> ArrowStoreBuilder {
        ArrowStoreBuilder::new(dim)
    }

    /// Get number of rows
    pub fn len(&self) -> usize {
        self.batch.num_rows()
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.batch.num_rows() == 0
    }

    /// Get vector dimension
    pub fn dim(&self) -> i32 {
        self.dim
    }

    /// Get the underlying RecordBatch
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Get schema
    pub fn schema(&self) -> Arc<Schema> {
        self.batch.schema().clone()
    }

    /// Get the vectors column
    pub fn vectors(&self) -> Column {
        let idx = self.batch.schema().index_of(&self.vector_column).unwrap();
        let array = self.batch.column(idx).clone();
        Column::from_arrow(&self.vector_column, array)
    }

    /// Get a column by name
    pub fn column(&self, name: &str) -> Option<Column> {
        let idx = self.batch.schema().index_of(name).ok()?;
        let array = self.batch.column(idx).clone();
        Some(Column::from_arrow(name, array))
    }

    /// Get all column names
    pub fn column_names(&self) -> Vec<String> {
        self.batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    /// Get metadata column names (all except vectors)
    pub fn metadata_columns(&self) -> Vec<String> {
        self.batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|name| name != &self.vector_column)
            .collect()
    }

    /// Get pre-computed inverse norms
    pub fn inv_norms(&self) -> &[f32] {
        &self.inv_norms
    }

    /// Get the name of the vector column
    pub fn vector_column_name(&self) -> &str {
        &self.vector_column
    }
}
