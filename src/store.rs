//! Unified Arrow-based store for vectors and metadata
//!
//! Combines vector search with metadata filtering using Apache Arrow's
//! RecordBatch for true columnar storage. All data (embeddings, metadata)
//! are stored together in a single table.

use crate::col::{Column, ColumnBuilder};
use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, Int64Array};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use std::fmt;
use std::sync::{Arc, Mutex};

const DEFAULT_VECTOR_COL: &str = "embeddings";
const DEFAULT_INV_NORM_COL: &str = "inv_norms";
const DEFAULT_ROW_ID_COL: &str = "row_id";

/// Unified store for vectors and metadata using Arrow RecordBatch
#[derive(Debug, Clone)]
pub struct OttersStore {
    /// RecordBatch containing all columns (vectors + metadata)
    batch: RecordBatch,

    /// Dimension of vectors
    dim: i32,

    /// Column names tracked for quick access
    vector_column: String,
    inv_norm_column: String,
    row_id_column: String,

    /// Cached column indices for fast lookup
    vector_index: usize,
    inv_norm_index: usize,
    row_id_index: usize,
    /// Stats from the most recent query execution
    last_query_stats: Arc<Mutex<Option<RecordBatch>>>,
}

/// Builder for constructing an OttersStore
pub struct OttersStoreBuilder {
    dim: i32,
    vector_column: String,
    inv_norm_column: String,
    row_id_column: String,
    vectors: Option<Column>,
    inv_norms: Option<Column>,
    row_ids: Option<Column>,
    metadata: Vec<(String, Column)>,
    error: Option<String>,
}

impl OttersStoreBuilder {
    /// Create a new builder for vectors of given dimension
    pub fn new(dim: i32) -> Self {
        Self {
            dim,
            vector_column: DEFAULT_VECTOR_COL.to_string(),
            inv_norm_column: DEFAULT_INV_NORM_COL.to_string(),
            row_id_column: DEFAULT_ROW_ID_COL.to_string(),
            vectors: None,
            inv_norms: None,
            row_ids: None,
            metadata: Vec::new(),
            error: None,
        }
    }

    /// Set custom name for vector column (default: "embeddings")
    pub fn with_vector_column_name(mut self, name: impl Into<String>) -> Self {
        self.vector_column = name.into();
        self
    }

    /// Set custom name for inverse-norm column (default: "inv_norms")
    pub fn with_inv_norm_column_name(mut self, name: impl Into<String>) -> Self {
        self.inv_norm_column = name.into();
        self
    }

    /// Set custom name for row-id column (default: "row_id")
    pub fn with_row_id_column_name(mut self, name: impl Into<String>) -> Self {
        self.row_id_column = name.into();
        self
    }

    /// Add vectors from a Vec<Vec<f32>>
    pub fn with_vectors(mut self, vectors: Vec<Vec<f32>>) -> Self {
        if self.error.is_some() {
            return self;
        }

        if vectors.is_empty() {
            self.error = Some("Cannot add empty vector list".to_string());
            return self;
        }

        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != self.dim as usize {
                self.error = Some(format!(
                    "Vector at index {i} has dimension {}, expected {}",
                    vec.len(),
                    self.dim
                ));
                return self;
            }
        }

        let mut vec_builder = ColumnBuilder::new_vector(&self.vector_column, self.dim);
        let mut inv_builder = ColumnBuilder::new_float32(&self.inv_norm_column);
        let mut row_builder = ColumnBuilder::new_int64(&self.row_id_column);

        for (row_id, vec) in vectors.iter().enumerate() {
            if let Err(e) = vec_builder.append(Some(vec.as_slice())) {
                self.error = Some(e.to_string());
                return self;
            }

            let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv = if norm != 0.0 { 1.0 / norm } else { 0.0 };
            if let Err(e) = inv_builder.append(Some(inv)) {
                self.error = Some(e.to_string());
                return self;
            }
            if let Err(e) = row_builder.append(Some(row_id as i64)) {
                self.error = Some(e.to_string());
                return self;
            }
        }

        self.vectors = Some(vec_builder.collect());
        self.inv_norms = Some(inv_builder.collect());
        self.row_ids = Some(row_builder.collect());
        self
    }

    fn is_reserved_name(&self, name: &str) -> bool {
        name == self.vector_column || name == self.inv_norm_column || name == self.row_id_column
    }

    /// Add a metadata column
    pub fn with_metadata_column(mut self, name: impl Into<String>, column: Column) -> Self {
        if self.error.is_some() {
            return self;
        }

        let name = name.into();
        if self.is_reserved_name(&name) {
            self.error = Some(format!(
                "Metadata column name '{name}' conflicts with reserved store columns"
            ));
            return self;
        }
        self.metadata.push((name, column));
        self
    }

    /// Add multiple metadata columns
    pub fn with_metadata_columns(
        mut self,
        columns: impl IntoIterator<Item = (String, Column)>,
    ) -> Self {
        if self.error.is_some() {
            return self;
        }

        for (name, column) in columns {
            if self.is_reserved_name(&name) {
                self.error = Some(format!(
                    "Metadata column name '{name}' conflicts with reserved store columns"
                ));
                return self;
            }
            self.metadata.push((name, column));
        }
        self
    }

    /// Build the final store
    pub fn build(self) -> Result<OttersStore, String> {
        if let Some(err) = self.error {
            return Err(err);
        }

        let vectors = self
            .vectors
            .ok_or_else(|| "Vectors not provided".to_string())?;

        let inv_norms = self
            .inv_norms
            .ok_or_else(|| "Inverse norms not computed".to_string())?;

        let row_ids = self
            .row_ids
            .ok_or_else(|| "Row ids not generated".to_string())?;

        let len = vectors.len();

        if inv_norms.len() != len {
            return Err(format!(
                "Inverse norms length {} does not match vectors length {}",
                inv_norms.len(),
                len
            ));
        }

        if row_ids.len() != len {
            return Err(format!(
                "Row ids length {} does not match vectors length {}",
                row_ids.len(),
                len
            ));
        }

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
        let mut fields = Vec::with_capacity(3 + self.metadata.len());
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(3 + self.metadata.len());

        // Add row id column first for stable indexing
        fields.push(row_ids.field().clone());
        arrays.push(row_ids.array().clone());

        // Add vector column
        fields.push(vectors.field().clone());
        arrays.push(vectors.array().clone());

        // Add inverse norms
        fields.push(inv_norms.field().clone());
        arrays.push(inv_norms.array().clone());

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
        let batch = RecordBatch::try_new(schema.clone(), arrays)
            .map_err(|e| format!("Failed to create RecordBatch: {e}"))?;

        let vector_index = schema
            .index_of(&self.vector_column)
            .map_err(|e| format!("Vector column '{}' missing: {e}", self.vector_column))?;
        let inv_norm_index = schema.index_of(&self.inv_norm_column).map_err(|e| {
            format!(
                "Inverse norm column '{}' missing: {e}",
                self.inv_norm_column
            )
        })?;
        let row_id_index = schema
            .index_of(&self.row_id_column)
            .map_err(|e| format!("Row id column '{}' missing: {e}", self.row_id_column))?;

        Ok(OttersStore {
            batch,
            dim: self.dim,
            vector_column: self.vector_column,
            inv_norm_column: self.inv_norm_column,
            row_id_column: self.row_id_column,
            vector_index,
            inv_norm_index,
            row_id_index,
            last_query_stats: Arc::new(Mutex::new(None)),
        })
    }
}

impl OttersStore {
    /// Create a new builder for given dimension
    pub fn builder(dim: i32) -> OttersStoreBuilder {
        OttersStoreBuilder::new(dim)
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

    /// Get the vectors column as a convenience wrapper
    pub fn vectors(&self) -> Column {
        let array = self.batch.column(self.vector_index).clone();
        Column::from_arrow(&self.vector_column, array)
    }

    /// Borrow the vectors as a FixedSizeListArray for zero-copy compute
    pub fn vectors_array(&self) -> &FixedSizeListArray {
        self.batch
            .column(self.vector_index)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("vector column must be FixedSizeListArray")
    }

    /// Borrow the inverse norms column as a Float32Array
    pub fn inv_norms_array(&self) -> &Float32Array {
        self.batch
            .column(self.inv_norm_index)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("inv norm column must be Float32Array")
    }

    /// Borrow the row id column as an Int64Array
    pub fn row_ids_array(&self) -> &Int64Array {
        self.batch
            .column(self.row_id_index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("row id column must be Int64Array")
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

    /// Get metadata column names (exclude vector, inv norms, and row id)
    pub fn metadata_columns(&self) -> Vec<String> {
        self.batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|name| {
                name != &self.vector_column
                    && name != &self.inv_norm_column
                    && name != &self.row_id_column
            })
            .collect()
    }

    /// Get the name of the vector column
    pub fn vector_column_name(&self) -> &str {
        &self.vector_column
    }

    /// Get the name of the inverse norm column
    pub fn inv_norm_column_name(&self) -> &str {
        &self.inv_norm_column
    }

    /// Get the name of the row id column
    pub fn row_id_column_name(&self) -> &str {
        &self.row_id_column
    }

    /// Retrieve the stats for the most recent query, if available.
    pub fn get_last_query_stats(&self) -> Option<RecordBatch> {
        self.last_query_stats
            .lock()
            .ok()
            .and_then(|stats| stats.clone())
    }

    /// Update the stored stats for the last query.
    pub(crate) fn set_last_query_stats(&self, stats: RecordBatch) {
        if let Ok(mut guard) = self.last_query_stats.lock() {
            *guard = Some(stats);
        }
    }
}

impl fmt::Display for OttersStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match pretty_format_batches(&[self.batch.clone()]) {
            Ok(formatted) => write!(f, "{formatted}"),
            Err(err) => write!(f, "Failed to format OttersStore: {err}"),
        }
    }
}
