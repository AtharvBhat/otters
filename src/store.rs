//! Simplified Arrow-based store with a builder-style workflow.
//!
//! `OttersStore` starts in an unbuilt state with a single [`RecordBatch`].
//! Configuration helpers like [`with_embedding_column`](Self::with_embedding_column)
//! never return errors; they record validation issues internally so callers can
//! chain them freely. [`build`](Self::build) is the single fallible step: it
//! validates accumulated errors, materializes helper columns (inverse norms and
//! row ids), and ensures the embedding column is a fixed-size Float32 vector.

use crate::col::{Column, OttersColumn};
use crate::query::OttersQuery;
use crate::record::OttersRecord;
use crate::vec_compute::inverse_norm;
use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float32Builder, Float64Array,
    GenericListArray, Int64Builder, LargeListArray, ListArray, OffsetSizeTrait, StringBuilder,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_csv::reader::Format;
use arrow_csv::ReaderBuilder;
use arrow_select::concat::concat_batches;
use glob::glob;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fmt;
use std::fs::File;
use std::io::Seek;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const DEFAULT_HEAD_ROWS: usize = 5;
const ROW_ID_COLUMN: &str = "row_id";
const UNBUILT_ERROR: &str =
    "Store In an unbuilt state, Call build() method of the state to build it";

/// Runtime state of the store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreState {
    Unbuilt,
    Built,
}

/// Arrow-backed store with builder-style configuration.
#[derive(Clone)]
pub struct OttersStore {
    batch: RecordBatch,
    state: StoreState,
    embedding_column: Option<String>,
    inv_norm_column: Option<String>,
    errors: Vec<String>,
    last_query_stats: Arc<Mutex<Option<OttersRecord>>>,
}

impl fmt::Debug for OttersStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OttersStore")
            .field("state", &self.state)
            .field("rows", &self.batch.num_rows())
            .field("columns", &self.batch.schema().fields().len())
            .field("embedding_column", &self.embedding_column)
            .field("inv_norm_column", &self.inv_norm_column)
            .field("errors", &self.errors)
            .finish()
    }
}

impl OttersStore {
    fn new_internal(batch: RecordBatch) -> Self {
        Self {
            batch,
            state: StoreState::Unbuilt,
            embedding_column: None,
            inv_norm_column: None,
            errors: Vec::new(),
            last_query_stats: Arc::new(Mutex::new(None)),
        }
    }

    fn empty_batch() -> RecordBatch {
        RecordBatch::new_empty(Arc::new(Schema::new(Vec::<Field>::new())))
    }

    fn empty() -> Self {
        Self::new_internal(Self::empty_batch())
    }

    fn from_errors(errors: Vec<String>) -> Self {
        let mut store = Self::empty();
        store.errors = errors;
        store
    }

    fn add_error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    fn take_errors(&mut self) -> Option<String> {
        if self.errors.is_empty() {
            None
        } else {
            Some(self.errors.drain(..).collect::<Vec<_>>().join("\n"))
        }
    }

    fn pretty_schema(&self) -> String {
        format!("{}", self.schema())
    }

    /// Load one or more Parquet files that match `pattern`.
    pub fn from_parquet(pattern: impl AsRef<str>) -> Self {
        let paths = match collect_paths(pattern.as_ref()) {
            Ok(paths) => paths,
            Err(err) => return Self::from_errors(vec![err]),
        };

        if paths.is_empty() {
            return Self::from_errors(vec![format!(
                "No parquet files matched pattern '{}'",
                pattern.as_ref()
            )]);
        }

        let mut batches = Vec::new();
        let mut errors = Vec::new();

        for path in paths {
            match read_parquet_batches(&path) {
                Ok(mut chunk) => batches.append(&mut chunk),
                Err(err) => errors.push(err),
            }
        }

        if !errors.is_empty() {
            return Self::from_errors(errors);
        }

        Self::from_recordbatches(batches)
    }

    /// Load one or more CSV files that match `pattern` (header row expected).
    pub fn from_csv(pattern: impl AsRef<str>) -> Self {
        let paths = match collect_paths(pattern.as_ref()) {
            Ok(paths) => paths,
            Err(err) => return Self::from_errors(vec![err]),
        };

        if paths.is_empty() {
            return Self::from_errors(vec![format!(
                "No CSV files matched pattern '{}'",
                pattern.as_ref()
            )]);
        }

        let mut batches = Vec::new();
        let mut errors = Vec::new();

        for path in paths {
            match read_csv_batches(&path) {
                Ok(mut chunk) => batches.append(&mut chunk),
                Err(err) => errors.push(err),
            }
        }

        if !errors.is_empty() {
            return Self::from_errors(errors);
        }

        Self::from_recordbatches(batches)
    }

    /// Create a store from a single [`RecordBatch`].
    pub fn from_recordbatch(batch: RecordBatch) -> Self {
        Self::new_internal(batch)
    }

    /// Create a store from multiple [`RecordBatch`] instances.
    pub fn from_recordbatches(batches: Vec<RecordBatch>) -> Self {
        if batches.is_empty() {
            return Self::from_errors(vec!["Cannot build store from empty batch list".to_string()]);
        }

        match concat_record_batches(batches) {
            Ok(batch) => Self::new_internal(batch),
            Err(err) => Self::from_errors(vec![err]),
        }
    }

    /// Build a store directly from columnar inputs.
    pub fn new<S, D>(schema: S, data: D) -> Self
    where
        S: IntoIterator,
        S::Item: Into<String>,
        D: IntoIterator<Item = OttersColumn>,
    {
        let names: Vec<String> = schema.into_iter().map(Into::into).collect();
        let columns: Vec<OttersColumn> = data.into_iter().collect();

        if names.len() != columns.len() {
            return Self::from_errors(vec![format!(
                "Schema column count {} does not match data column count {}",
                names.len(),
                columns.len()
            )]);
        }

        if columns.is_empty() {
            return Self::from_errors(vec!["Cannot create store without columns".to_string()]);
        }

        let expected_rows = columns[0].len();
        if columns.iter().any(|col| col.len() != expected_rows) {
            return Self::from_errors(vec![
                "All columns must have the same number of rows".to_string()
            ]);
        }

        let arrays: Vec<ArrayRef> = columns.iter().map(|c| c.array().clone()).collect();
        let fields: Vec<Field> = names
            .iter()
            .zip(arrays.iter())
            .map(|(name, array)| Field::new(name.clone(), array.data_type().clone(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));

        match RecordBatch::try_new(schema, arrays) {
            Ok(batch) => Self::new_internal(batch),
            Err(err) => Self::from_errors(vec![format!("Failed to create batch: {err}")]),
        }
    }

    /// Current runtime state of the store.
    pub fn state(&self) -> StoreState {
        self.state
    }

    /// Returns `true` once [`OttersStore::build`] has succeeded.
    pub fn is_ready(&self) -> bool {
        matches!(self.state, StoreState::Built)
    }

    /// Number of rows present in the underlying batch.
    pub fn len(&self) -> usize {
        self.batch.num_rows()
    }

    /// Returns `true` when the store has zero rows.
    pub fn is_empty(&self) -> bool {
        self.batch.num_rows() == 0
    }

    /// Return a preview of the first few rows (default: 5).
    pub fn head(&self) -> OttersRecord {
        self.head_n(DEFAULT_HEAD_ROWS)
    }

    /// Return a preview of the first `rows` rows.
    pub fn head_n(&self, rows: usize) -> OttersRecord {
        let preview_rows = rows.min(self.batch.num_rows());
        let preview = if preview_rows == self.batch.num_rows() {
            self.batch.clone()
        } else {
            self.batch.slice(0, preview_rows)
        };
        OttersRecord::from(preview)
    }

    /// Render the schema as a printable [`OttersRecord`].
    pub fn schema(&self) -> OttersRecord {
        let mut name_builder = StringBuilder::new();
        let mut type_builder = StringBuilder::new();

        for field in self.batch.schema().fields() {
            name_builder.append_value(field.name());
            type_builder.append_value(format!("{:?}", field.data_type()));
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("column", DataType::Utf8, false),
            Field::new("data_type", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(name_builder.finish()) as ArrayRef,
                Arc::new(type_builder.finish()) as ArrayRef,
            ],
        )
        .expect("failed to build schema record");

        OttersRecord::from(batch)
    }

    /// List of column names in the current batch.
    pub fn column_names(&self) -> Vec<String> {
        self.batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }

    /// Fetch a column by name. Errors include the formatted schema when missing.
    pub fn column(&self, name: &str) -> Result<OttersColumn, String> {
        let schema = self.batch.schema();
        let (index, field) = schema.column_with_name(name).ok_or_else(|| {
            format!(
                "Column '{name}' not found. Available schema:\n{}",
                self.pretty_schema()
            )
        })?;
        let array = self.batch.column(index).clone();
        Ok(OttersColumn::from_field(field.clone(), array))
    }

    /// Metadata column names (excluding embeddings, inverse norms, and row id).
    pub fn metadata_columns(&self) -> Vec<String> {
        self.column_names()
            .into_iter()
            .filter(|name| {
                name != ROW_ID_COLUMN
                    && self.embedding_column.as_deref() != Some(name.as_str())
                    && self.inv_norm_column.as_deref() != Some(name.as_str())
            })
            .collect()
    }

    /// Embedding column selected via [`with_embedding_column`](Self::with_embedding_column).
    pub fn embedding_column(&self) -> Option<&str> {
        self.embedding_column.as_deref()
    }

    /// Inverse norm column name generated during [`build`](Self::build).
    pub fn inv_norm_column(&self) -> Option<&str> {
        self.inv_norm_column.as_deref()
    }

    /// Embedding column name (panics if not configured).
    pub fn embedding_column_name(&self) -> &str {
        self.embedding_column
            .as_deref()
            .expect("Embedding column not configured; call with_embedding_column() first")
    }

    /// Inverse norm column name (panics if store not built).
    pub fn inv_norm_column_name(&self) -> &str {
        self.inv_norm_column
            .as_deref()
            .expect("Inverse norm column not available; build the store first")
    }

    /// Name of the generated row id column.
    pub fn row_id_column_name(&self) -> &str {
        ROW_ID_COLUMN
    }

    /// Borrow the underlying schema.
    pub fn arrow_schema(&self) -> Arc<Schema> {
        self.batch.schema()
    }

    /// Set the embedding column for the store. Errors are recorded internally.
    pub fn with_embedding_column(mut self, column: impl Into<String>) -> Self {
        if self.is_ready() {
            self.add_error("Store already built; embedding column cannot be changed");
            return self;
        }

        let name = column.into();
        if name.is_empty() {
            self.add_error("Embedding column name cannot be empty");
            return self;
        }

        let schema = self.batch.schema();
        if schema.column_with_name(&name).is_none() {
            self.add_error(format!(
                "Embedding column '{name}' not found. Schema:\n{}",
                self.pretty_schema()
            ));
            return self;
        }

        self.embedding_column = Some(name.clone());
        self.inv_norm_column = Some(format!("{name}_inv_norms"));
        self
    }

    /// Finalize the store, computing helper columns for querying.
    pub fn build(mut self) -> Result<Self, String> {
        if self.state == StoreState::Built {
            return Err("Store already built".to_string());
        }

        if self.embedding_column.is_none() {
            self.add_error("Embedding column not set; call with_embedding_column() before build()");
        }

        if let Some(err) = self.take_errors() {
            return Err(err);
        }

        let schema = self.batch.schema();
        let mut fields: Vec<Field> = schema.fields().iter().map(|f| (**f).clone()).collect();
        let mut arrays: Vec<ArrayRef> = self.batch.columns().to_vec();

        let embedding_name = self.embedding_column.clone().expect("checked earlier");
        let inv_norm_name = self
            .inv_norm_column
            .clone()
            .unwrap_or_else(|| format!("{embedding_name}_inv_norms"));

        let (index, field) = match schema.column_with_name(&embedding_name) {
            Some(pair) => pair,
            None => {
                self.add_error(format!(
                    "Embedding column '{embedding_name}' missing during build. Schema:\n{}",
                    self.pretty_schema()
                ));
                return Err(self.take_errors().unwrap());
            }
        };

        match convert_embedding_array(field, &arrays[index]) {
            Ok((new_field, new_array)) => {
                arrays[index] = new_array;
                fields[index] = new_field;
            }
            Err(err) => {
                self.add_error(err);
                return Err(self.take_errors().unwrap());
            }
        }

        if fields.iter().any(|f| f.name() == &inv_norm_name) {
            return Err(format!(
                "Column '{inv_norm_name}' already exists. Adjust embedding column name."
            ));
        }

        let list_array = arrays[index]
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                format!("Embedding column '{embedding_name}' must be FixedSizeList<Float32>")
            })?;

        let inv_array = build_inv_norm_array(list_array, &embedding_name)?;
        arrays.push(inv_array);
        fields.push(Field::new(inv_norm_name.clone(), DataType::Float32, true));

        if fields.iter().any(|f| f.name() == ROW_ID_COLUMN) {
            self.add_error(format!(
                "Column '{ROW_ID_COLUMN}' already exists. Rename it before building."
            ));
        } else {
            let mut row_builder = Int64Builder::with_capacity(self.batch.num_rows());
            for idx in 0..self.batch.num_rows() {
                row_builder.append_value(idx as i64);
            }
            arrays.push(Arc::new(row_builder.finish()));
            fields.push(Field::new(ROW_ID_COLUMN, DataType::Int64, false));
        }

        if let Some(err) = self.take_errors() {
            return Err(err);
        }

        let new_schema = Arc::new(Schema::new(fields));
        self.batch = RecordBatch::try_new(new_schema, arrays)
            .map_err(|e| format!("Failed to finalize store: {e}"))?;
        self.inv_norm_column = Some(inv_norm_name);
        self.state = StoreState::Built;
        Ok(self)
    }

    /// Dimension of the primary embedding column.
    pub fn dim(&self) -> i32 {
        let schema = self.batch.schema();
        let name = self.embedding_column_name().to_string();
        let field = schema
            .field_with_name(&name)
            .unwrap_or_else(|_| panic!("Embedding column '{name}' missing"));
        match field.data_type() {
            DataType::FixedSizeList(_, value_length) => *value_length,
            other => panic!("Embedding column '{name}' has incompatible type {other:?}"),
        }
    }

    /// Access the finalized record batch.
    pub fn batch(&self) -> Option<&RecordBatch> {
        if self.is_ready() {
            Some(&self.batch)
        } else {
            None
        }
    }

    /// Clone the finalized record batch.
    pub fn to_recordbatch(&self) -> Option<RecordBatch> {
        self.batch().cloned()
    }

    /// Last query statistics, if recorded.
    pub fn get_last_query_stats(&self) -> Option<OttersRecord> {
        self.last_query_stats
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().cloned())
    }

    /// Update last query statistics.
    pub fn set_last_query_stats(&self, stats: OttersRecord) {
        if let Ok(mut guard) = self.last_query_stats.lock() {
            *guard = Some(stats);
        }
    }

    /// Start a new query plan (errors are deferred until `collect()`).
    pub fn query(&self) -> OttersQuery<'_> {
        let mut plan = OttersQuery::new(self);
        if !self.is_ready() {
            plan.set_error(UNBUILT_ERROR.to_string());
        }
        plan
    }

    pub fn ensure_ready(&self) -> Result<(), String> {
        if self.is_ready() {
            Ok(())
        } else {
            Err(UNBUILT_ERROR.to_string())
        }
    }
}

impl fmt::Display for OttersStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "OttersStore<{:?}>: rows={}, columns={}",
            self.state,
            self.batch.num_rows(),
            self.batch.schema().fields().len()
        )?;
        writeln!(f, "Embedding column: {:?}", self.embedding_column)?;
        if self.is_ready() {
            writeln!(f, "Inverse norm column: {:?}", self.inv_norm_column)?;
            writeln!(f, "Includes generated row id column '{ROW_ID_COLUMN}'")?;
        } else {
            writeln!(
                f,
                "Call build() after selecting embedding columns to enable querying."
            )?;
            if !self.errors.is_empty() {
                writeln!(f, "Pending errors:")?;
                for err in &self.errors {
                    writeln!(f, "- {err}")?;
                }
            }
        }
        Ok(())
    }
}

fn concat_record_batches(mut batches: Vec<RecordBatch>) -> Result<RecordBatch, String> {
    if batches.len() == 1 {
        return Ok(batches.remove(0));
    }

    let schema = batches[0].schema();
    concat_batches(&schema, &batches).map_err(|e| format!("Failed to concatenate batches: {e}"))
}

fn collect_paths(pattern: &str) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    for entry in glob(pattern).map_err(|e| format!("Invalid glob pattern '{pattern}': {e}"))? {
        match entry {
            Ok(path) if path.is_file() => paths.push(path),
            Ok(_) => {}
            Err(err) => return Err(format!("Failed to read glob entry: {err}")),
        }
    }
    Ok(paths)
}

fn read_parquet_batches(path: &Path) -> Result<Vec<RecordBatch>, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open parquet file '{}': {e}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        format!(
            "Failed to create parquet reader for '{}': {e}",
            path.display()
        )
    })?;
    let reader = builder.build().map_err(|e| {
        format!(
            "Failed to build parquet reader for '{}': {e}",
            path.display()
        )
    })?;

    reader.collect::<Result<Vec<_>, _>>().map_err(|e| {
        format!(
            "Failed to read parquet batches from '{}': {e}",
            path.display()
        )
    })
}

fn read_csv_batches(path: &Path) -> Result<Vec<RecordBatch>, String> {
    let mut file = File::open(path)
        .map_err(|e| format!("Failed to open CSV file '{}': {e}", path.display()))?;
    let format = Format::default().with_header(true);
    let (schema, _) = format
        .infer_schema(&mut file, None)
        .map_err(|e| format!("Failed to infer CSV schema for '{}': {e}", path.display()))?;
    file.rewind()
        .map_err(|e| format!("Failed to rewind CSV file '{}': {e}", path.display()))?;

    let reader = ReaderBuilder::new(Arc::new(schema))
        .with_format(format)
        .build(file)
        .map_err(|e| format!("Failed to create CSV reader for '{}': {e}", path.display()))?;

    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to read CSV batches from '{}': {e}", path.display()))
}

fn convert_embedding_array(field: &Field, array: &ArrayRef) -> Result<(Field, ArrayRef), String> {
    match array.data_type() {
        DataType::FixedSizeList(inner, dim) => match inner.data_type() {
            DataType::Float32 => Ok(convert_existing_fixed_size(field, array.clone())),
            DataType::Float64 => convert_fixed_size_float64(field, array, *dim),
            other => Err(format!(
                "Embedding column '{}' must contain Float32 or Float64 values, found {other:?}",
                field.name()
            )),
        },
        DataType::List(_) => {
            let list = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| "Failed to downcast embedding column to ListArray".to_string())?;
            convert_generic_list(field, list)
        }
        DataType::LargeList(_) => {
            let list = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| {
                    "Failed to downcast embedding column to LargeListArray".to_string()
                })?;
            convert_generic_list(field, list)
        }
        other => Err(format!(
            "Embedding column '{}' must be a list of Float32/Float64 values, found {other:?}",
            field.name()
        )),
    }
}

fn convert_existing_fixed_size(field: &Field, array: ArrayRef) -> (Field, ArrayRef) {
    let mut new_field = Field::new(
        field.name().clone(),
        array.data_type().clone(),
        field.is_nullable(),
    );
    if !field.metadata().is_empty() {
        new_field = new_field.with_metadata(field.metadata().clone());
    }
    (new_field, array)
}

fn convert_fixed_size_float64(
    field: &Field,
    array: &ArrayRef,
    dim: i32,
) -> Result<(Field, ArrayRef), String> {
    let list = array
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| "Failed to downcast embedding column to FixedSizeListArray".to_string())?;

    let mut builder = Column::new_vector(field.name(), dim);
    let mut buffer = Vec::with_capacity(dim as usize);

    for row in 0..list.len() {
        if list.is_null(row) {
            builder = builder.append([None::<&[f32]>]);
            continue;
        }

        let values = list.value(row);
        let values = values
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| "Embedding column must contain Float64 values".to_string())?;
        buffer.clear();
        copy_numeric_values(values, dim as usize, &mut buffer, field.name())?;
        builder = builder.append([Some(buffer.as_slice())]);
    }

    let column = builder
        .collect()
        .map_err(|e| format!("Failed to rebuild embedding column '{}': {e}", field.name()))?;
    let array = column.array().clone();
    let mut new_field = Field::new(
        field.name().clone(),
        array.data_type().clone(),
        field.is_nullable(),
    );
    if !field.metadata().is_empty() {
        new_field = new_field.with_metadata(field.metadata().clone());
    }
    Ok((new_field, array))
}

fn convert_generic_list<O: OffsetSizeTrait>(
    field: &Field,
    list: &GenericListArray<O>,
) -> Result<(Field, ArrayRef), String> {
    let mut rows: Vec<Option<Vec<f32>>> = Vec::with_capacity(list.len());
    let mut dim: Option<i32> = None;

    for row in 0..list.len() {
        if list.is_null(row) {
            rows.push(None);
            continue;
        }

        let values = list.value(row);
        let mut row_values = Vec::new();

        if let Some(arr) = values.as_any().downcast_ref::<Float32Array>() {
            if arr.null_count() > 0 {
                return Err(format!(
                    "Embedding column '{}' contains null values; clean the data before loading",
                    field.name()
                ));
            }
            row_values.extend_from_slice(arr.values());
        } else if let Some(arr64) = values.as_any().downcast_ref::<Float64Array>() {
            if arr64.null_count() > 0 {
                return Err(format!(
                    "Embedding column '{}' contains null values; clean the data before loading",
                    field.name()
                ));
            }
            row_values.extend(arr64.values().iter().map(|v| *v as f32));
        } else {
            return Err(format!(
                "Embedding column '{}' must contain Float32/Float64 values",
                field.name()
            ));
        }

        let row_dim = row_values.len() as i32;
        if row_dim == 0 {
            return Err(format!(
                "Embedding column '{}' contains empty vectors",
                field.name()
            ));
        }

        if let Some(existing) = dim {
            if existing != row_dim {
                return Err(format!(
                    "Embedding column '{}' has inconsistent vector dimensions: {existing} vs {row_dim}",
                    field.name()
                ));
            }
        } else {
            dim = Some(row_dim);
        }

        rows.push(Some(row_values));
    }

    let dim = dim.ok_or_else(|| {
        format!(
            "Embedding column '{}' contains only null values; unable to infer dimension",
            field.name()
        )
    })?;

    let mut builder = Column::new_vector(field.name(), dim);
    for row in rows {
        match row {
            Some(values) => builder = builder.append([Some(values.as_slice())]),
            None => builder = builder.append([None::<&[f32]>]),
        }
    }

    let column = builder
        .collect()
        .map_err(|e| format!("Failed to rebuild embedding column '{}': {e}", field.name()))?;
    let array = column.array().clone();
    let mut new_field = Field::new(
        field.name().clone(),
        array.data_type().clone(),
        field.is_nullable(),
    );
    if !field.metadata().is_empty() {
        new_field = new_field.with_metadata(field.metadata().clone());
    }
    Ok((new_field, array))
}

fn copy_numeric_values(
    values: &Float64Array,
    expected_len: usize,
    buffer: &mut Vec<f32>,
    column: &str,
) -> Result<(), String> {
    if values.null_count() > 0 {
        return Err(format!(
            "Embedding column '{column}' contains null values; please clean the data before loading"
        ));
    }
    if values.len() != expected_len {
        return Err(format!(
            "Embedding column '{column}' contains inconsistent lengths ({})",
            values.len()
        ));
    }
    buffer.clear();
    buffer.extend(values.values().iter().map(|v| *v as f32));
    Ok(())
}

fn build_inv_norm_array(list: &FixedSizeListArray, name: &str) -> Result<ArrayRef, String> {
    let mut builder = Float32Builder::with_capacity(list.len());

    for row in 0..list.len() {
        if list.is_null(row) {
            builder.append_null();
            continue;
        }

        let values = list.value(row);
        let values = values
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| "Embedding vectors must contain Float32 values".to_string())?;
        if values.null_count() > 0 {
            return Err(format!(
                "Embedding column '{name}' contains null vector entries; clean the data before building"
            ));
        }
        let inv = inverse_norm(values.values());
        builder.append_value(inv);
    }

    Ok(Arc::new(builder.finish()))
}
