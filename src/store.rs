//! Unified Arrow-based store for vectors and metadata with lazy build stages.
//!
//! OttersStore can be created from Arrow data (Parquet, CSV, RecordBatches, or
//! in-memory columns). The store starts in a **draft** state where the schema
//! can be inspected, column names can be tweaked, and the embedding column can
//! be selected. Queries are only available once `build()` has been called,
//! which finalizes the store by computing row ids, inverse norms, and caching
//! column indexes.

use crate::col::{Column, OttersColumn};
use crate::record::OttersRecord;
use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, Int64Array, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_array::Array;
use arrow_csv::ReaderBuilder;
use arrow_csv::reader::Format;
use arrow_select::concat::concat_batches;
use glob::glob;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const DEFAULT_VECTOR_COL: &str = "embeddings";
const DEFAULT_INV_NORM_COL: &str = "inv_norms";
const DEFAULT_ROW_ID_COL: &str = "row_id";

/// Helper selection type for choosing and optionally renaming the embedding column.
pub enum EmbeddingSelection {
    Source(String),
    SourceAs { source: String, alias: String },
}

impl From<String> for EmbeddingSelection {
    fn from(value: String) -> Self {
        EmbeddingSelection::Source(value)
    }
}

impl<'a> From<&'a str> for EmbeddingSelection {
    fn from(value: &'a str) -> Self {
        EmbeddingSelection::Source(value.to_string())
    }
}

impl<'a> From<&'a String> for EmbeddingSelection {
    fn from(value: &'a String) -> Self {
        EmbeddingSelection::Source(value.clone())
    }
}

impl<S: Into<String>, T: Into<String>> From<(S, T)> for EmbeddingSelection {
    fn from(value: (S, T)) -> Self {
        EmbeddingSelection::SourceAs {
            source: value.0.into(),
            alias: value.1.into(),
        }
    }
}

/// Unified store for vectors and metadata using Arrow RecordBatch.
pub struct OttersStore {
    state: StoreState,
    vector_column_name: String,
    inv_norm_column_name: String,
    row_id_column_name: String,
    health: StoreHealth,
}

impl Clone for OttersStore {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            vector_column_name: self.vector_column_name.clone(),
            inv_norm_column_name: self.inv_norm_column_name.clone(),
            row_id_column_name: self.row_id_column_name.clone(),
            health: self.health.clone(),
        }
    }
}

impl fmt::Debug for OttersStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OttersStore")
            .field("state", &self.state)
            .field("vector_column_name", &self.vector_column_name)
            .field("inv_norm_column_name", &self.inv_norm_column_name)
            .field("row_id_column_name", &self.row_id_column_name)
            .field("health", &self.health)
            .finish()
    }
}

#[derive(Clone, Debug)]
enum StoreState {
    Draft(DraftState),
    Ready(ReadyState),
}

#[derive(Clone, Debug, Default)]
enum StoreHealth {
    #[default]
    Ready,
    Failed(String),
}

impl StoreHealth {
    fn fail(&mut self, msg: impl Into<String>) {
        if matches!(self, StoreHealth::Ready) {
            *self = StoreHealth::Failed(msg.into());
        }
    }

    fn is_failed(&self) -> bool {
        matches!(self, StoreHealth::Failed(_))
    }

    fn ensure_ok(&self) -> Result<(), String> {
        match self {
            StoreHealth::Ready => Ok(()),
            StoreHealth::Failed(msg) => Err(msg.clone()),
        }
    }

    fn error_message(&self) -> Option<&String> {
        match self {
            StoreHealth::Ready => None,
            StoreHealth::Failed(msg) => Some(msg),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct DraftState {
    batches: Vec<RecordBatch>,
    selected_embedding_column: Option<String>,
}

#[derive(Clone, Debug)]
struct ReadyState {
    batch: OttersRecord,
    dim: i32,
    vector_index: usize,
    inv_norm_index: usize,
    row_id_index: usize,
    last_query_stats: Arc<Mutex<Option<OttersRecord>>>,
}

impl DraftState {
    fn new(mut batches: Vec<RecordBatch>) -> Result<Self, String> {
        if batches.is_empty() {
            return Err("No record batches available".to_string());
        }

        let schema = batches[0].schema();
        for batch in &batches[1..] {
            if batch.schema().fields().len() != schema.fields().len() || batch.schema() != schema {
                return Err("Input batches have mismatched schemas".to_string());
            }
        }

        // Normalize batches to share the same schema instance for easier equality checks.
        for batch in &mut batches[1..] {
            if !Arc::ptr_eq(&batch.schema(), &schema) {
                *batch = RecordBatch::try_new(schema.clone(), batch.columns().to_vec())
                    .map_err(|e| format!("Failed to normalize batch schema: {e}"))?;
            }
        }

        Ok(Self {
            batches,
            selected_embedding_column: None,
        })
    }

    fn schema(&self) -> Arc<Schema> {
        self.batches
            .first()
            .map(|batch| batch.schema())
            .unwrap_or_else(|| Arc::new(Schema::new(Vec::<Field>::new())))
    }

    fn len(&self) -> usize {
        self.batches.iter().map(|batch| batch.num_rows()).sum()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn column(&self, name: &str) -> Option<OttersColumn> {
        let batch = self.batches.first()?;
        let schema = batch.schema();
        let (index, field) = schema.column_with_name(name)?;
        let array = batch.column(index).clone();
        Some(OttersColumn::from_field(field.clone(), array))
    }

    fn apply_renames(&mut self, renames: &HashMap<String, String>) -> Result<(), String> {
        if renames.is_empty() {
            return Ok(());
        }

        let schema = self.schema();
        for from in renames.keys() {
            if schema.column_with_name(from).is_none() {
                return Err(format!("Cannot rename missing column '{from}'"));
            }
        }

        self.batches = self
            .batches
            .iter()
            .map(|batch| rename_batch(batch, renames))
            .collect::<Result<Vec<_>, _>>()?;

        if let Some(selected) = &mut self.selected_embedding_column {
            if let Some(new_name) = renames.get(selected) {
                *selected = new_name.clone();
            }
        }

        Ok(())
    }
}

impl ReadyState {
    fn schema(&self) -> Arc<Schema> {
        self.batch.schema().clone()
    }

    fn len(&self) -> usize {
        self.batch.num_rows()
    }
}

impl OttersStore {
    fn empty() -> Self {
        Self {
            state: StoreState::Draft(DraftState::default()),
            vector_column_name: DEFAULT_VECTOR_COL.to_string(),
            inv_norm_column_name: DEFAULT_INV_NORM_COL.to_string(),
            row_id_column_name: DEFAULT_ROW_ID_COL.to_string(),
            health: StoreHealth::default(),
        }
    }

    fn fail(&mut self, err: impl Into<String>) {
        self.health.fail(err);
    }

    fn is_failed(&self) -> bool {
        self.health.is_failed()
    }

    fn ensure_ok(&self) -> Result<(), String> {
        self.health.ensure_ok()
    }

    fn error_message(&self) -> Option<&String> {
        self.health.error_message()
    }

    fn assign_draft_batches(&mut self, batches: Vec<RecordBatch>) -> Result<(), String> {
        let draft = DraftState::new(batches)?;
        self.state = StoreState::Draft(draft);
        Ok(())
    }

    /// Load one or more Parquet files into an `OttersStore` in draft form.
    pub fn from_parquet(pattern: impl AsRef<str>) -> Self {
        let mut store = Self::empty();

        match load_batches_with(
            pattern.as_ref(),
            |p| format!("No parquet files matched pattern '{p}'"),
            read_parquet_batches,
        ) {
            Ok(batches) => {
                if let Err(err) = store.assign_draft_batches(batches) {
                    store.fail(err);
                }
            }
            Err(err) => store.fail(err),
        }

        store
    }

    /// Load one or more CSV files into an `OttersStore` in draft form.
    ///
    /// Schema inference follows Arrow's default CSV reader rules (header row enabled).
    pub fn from_csv(pattern: impl AsRef<str>) -> Self {
        let mut store = Self::empty();

        match load_batches_with(
            pattern.as_ref(),
            |p| format!("No CSV files matched pattern '{p}'"),
            read_csv_batches,
        ) {
            Ok(batches) => {
                if let Err(err) = store.assign_draft_batches(batches) {
                    store.fail(err);
                }
            }
            Err(err) => store.fail(err),
        }

        store
    }

    /// Create a draft store from an existing [`RecordBatch`].
    pub fn from_recordbatch(batch: RecordBatch) -> Self {
        Self::from_recordbatches(vec![batch])
    }

    /// Create a draft store from multiple [`RecordBatch`] instances.
    pub fn from_recordbatches(batches: Vec<RecordBatch>) -> Self {
        let mut store = Self::empty();
        if let Err(err) = store.assign_draft_batches(batches) {
            store.fail(err);
        }
        store
    }

    /// Build a store from in-memory columns.
    ///
    /// Example:
    /// ```
    /// # use otters::col::Column;
    /// # use otters::store::OttersStore;
    /// let ages = Column::new_int32("age").collect().unwrap();
    /// let names = Column::new_string("name").collect().unwrap();
    /// let embeddings = Column::new_vector("embedding", 3).collect().unwrap();
    /// let store = OttersStore::new(
    ///     ["age", "name", "embedding"],
    ///     [ages, names, embeddings],
    /// )
    /// .with_embedding_column("embedding")
    /// .build()
    /// .unwrap();
    /// ```
    pub fn new<S, D>(schema: S, data: D) -> Self
    where
        S: IntoIterator,
        S::Item: Into<String>,
        D: IntoIterator<Item = OttersColumn>,
    {
        let names: Vec<String> = schema.into_iter().map(Into::into).collect();
        let mut columns: Vec<OttersColumn> = data.into_iter().collect();
        let mut store = Self::empty();

        if names.len() != columns.len() {
            store.fail(format!(
                "Schema column count {} does not match data column count {}",
                names.len(),
                columns.len()
            ));
            return store;
        }

        if names.is_empty() {
            store.fail("Cannot build store with empty schema".to_string());
            return store;
        }

        let len = columns.first().map(|col| col.len()).unwrap_or(0);
        for (idx, column) in columns.iter().enumerate() {
            if column.len() != len {
                store.fail(format!(
                    "Column '{}' length {} does not match expected length {}",
                    names[idx],
                    column.len(),
                    len
                ));
            }
        }
        if store.is_failed() {
            return store;
        }

        for (name, column) in names.iter().zip(columns.iter_mut()) {
            if column.name() != name {
                let field = Field::new(
                    name.clone(),
                    column.dtype().clone(),
                    column.field().is_nullable(),
                );
                *column = OttersColumn::from_field(field, column.array().clone());
            }
        }

        let arrays: Vec<ArrayRef> = columns.into_iter().map(|col| col.array().clone()).collect();
        let fields: Vec<Field> = names
            .iter()
            .zip(arrays.iter())
            .map(|(name, array)| Field::new(name.clone(), array.data_type().clone(), true))
            .collect();
        let schema = Arc::new(Schema::new(fields));

        match RecordBatch::try_new(schema.clone(), arrays) {
            Ok(batch) => {
                if let Err(err) = store.assign_draft_batches(vec![batch]) {
                    store.fail(err);
                }
            }
            Err(err) => store.fail(format!("Failed to create RecordBatch: {err}")),
        }

        store
    }

    /// Select the embedding column, optionally renaming it in the finalized store.
    ///
    /// Examples:
    /// - `store.with_embedding_column("embedding");`
    /// - `store.with_embedding_column(("embedding", "vec"));`
    pub fn with_embedding_column(mut self, selection: impl Into<EmbeddingSelection>) -> Self {
        if !self.is_failed() {
            if let Err(err) = self.set_embedding_column(selection.into()) {
                self.fail(err);
            }
        }
        self
    }

    /// Configure the generated inverse norm column name in the finalized store.
    pub fn with_inv_norm_column_name(mut self, name: impl Into<String>) -> Self {
        if !self.is_failed() {
            if let Err(err) = self.set_inv_norm_column_name(name.into()) {
                self.fail(err);
            }
        }
        self
    }

    /// Configure the generated row id column name in the finalized store.
    pub fn with_row_id_column_name(mut self, name: impl Into<String>) -> Self {
        if !self.is_failed() {
            if let Err(err) = self.set_row_id_column_name(name.into()) {
                self.fail(err);
            }
        }
        self
    }

    /// Finalize the draft store, computing cached state required for querying.
    pub fn build(mut self) -> Result<Self, String> {
        self.build_mut()?;
        Ok(self)
    }

    /// Finalize the draft store in-place.
    pub fn build_mut(&mut self) -> Result<(), String> {
        self.ensure_ok()?;

        if matches!(self.state, StoreState::Ready(_)) {
            return Err("Store already built".to_string());
        }

        let draft = match &self.state {
            StoreState::Draft(d) => d,
            StoreState::Ready(_) => unreachable!("checked above"),
        };

        let ready = finalize_store(
            draft,
            &self.vector_column_name,
            &self.inv_norm_column_name,
            &self.row_id_column_name,
        )?;

        self.state = StoreState::Ready(ready);
        Ok(())
    }

    /// Rename one or more columns by providing an iterator of `(from, to)` pairs.
    pub fn rename_columns<I, K, V>(&mut self, renames: I) -> Result<(), String>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.ensure_ok()?;

        let map: HashMap<String, String> = renames
            .into_iter()
            .map(|(from, to)| (from.into(), to.into()))
            .collect();

        if map.is_empty() {
            return Ok(());
        }

        let schema = self.arrow_schema();
        for from in map.keys() {
            if schema.column_with_name(from).is_none() {
                return Err(format!("Cannot rename missing column '{from}'"));
            }
        }

        let current_vector = self.vector_column_name.clone();
        let current_inv = self.inv_norm_column_name.clone();
        let current_row = self.row_id_column_name.clone();
        let new_vector = map
            .get(&current_vector)
            .cloned()
            .unwrap_or(current_vector.clone());
        let new_inv = map
            .get(&current_inv)
            .cloned()
            .unwrap_or(current_inv.clone());
        let new_row = map
            .get(&current_row)
            .cloned()
            .unwrap_or(current_row.clone());

        if new_vector == new_inv || new_vector == new_row || new_inv == new_row {
            return Err("Renaming would cause reserved columns to share the same name".to_string());
        }

        match &mut self.state {
            StoreState::Draft(draft) => draft.apply_renames(&map)?,
            StoreState::Ready(ready) => apply_ready_renames(ready, &map)?,
        }

        self.update_special_names(&map);

        if let StoreState::Ready(ready) = &mut self.state {
            let schema = ready.batch.schema();
            ready.vector_index = schema.index_of(&self.vector_column_name).map_err(|e| {
                format!(
                    "Vector column '{}' missing after rename: {e}",
                    self.vector_column_name
                )
            })?;
            ready.inv_norm_index = schema.index_of(&self.inv_norm_column_name).map_err(|e| {
                format!(
                    "Inverse norm column '{}' missing after rename: {e}",
                    self.inv_norm_column_name
                )
            })?;
            ready.row_id_index = schema.index_of(&self.row_id_column_name).map_err(|e| {
                format!(
                    "Row id column '{}' missing after rename: {e}",
                    self.row_id_column_name
                )
            })?;
        }
        Ok(())
    }

    /// Returns `true` if the store has been built and is ready for querying.
    pub fn is_ready(&self) -> bool {
        !self.is_failed() && matches!(self.state, StoreState::Ready(_))
    }

    /// Returns `true` if the store has no rows.
    pub fn is_empty(&self) -> bool {
        match &self.state {
            StoreState::Draft(draft) => draft.is_empty(),
            StoreState::Ready(ready) => ready.len() == 0,
        }
    }

    pub(crate) fn pending_error(&self) -> Option<&String> {
        self.error_message()
    }

    /// Number of rows currently loaded.
    pub fn len(&self) -> usize {
        match &self.state {
            StoreState::Draft(draft) => draft.len(),
            StoreState::Ready(ready) => ready.len(),
        }
    }

    /// Dimension of the embedding vectors. Requires the store to be built.
    pub fn dim(&self) -> i32 {
        self.ensure_ready()
            .expect("Store not built: call build() before accessing dim")
            .dim
    }

    /// Retrieve the underlying Arrow schema.
    pub fn arrow_schema(&self) -> Arc<Schema> {
        match &self.state {
            StoreState::Draft(draft) => draft.schema(),
            StoreState::Ready(ready) => ready.schema(),
        }
    }

    /// Render the schema as a printable [`OttersRecord`].
    pub fn schema(&self) -> OttersRecord {
        let schema = self.arrow_schema();
        let mut name_builder = StringBuilder::new();
        let mut type_builder = StringBuilder::new();

        for field in schema.fields() {
            name_builder.append_value(field.name());
            let dtype_repr = format!("{:?}", field.data_type());
            type_builder.append_value(&dtype_repr);
        }

        let schema_batch = Arc::new(Schema::new(vec![
            Field::new("column", DataType::Utf8, false),
            Field::new("data_type", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema_batch,
            vec![
                Arc::new(name_builder.finish()) as ArrayRef,
                Arc::new(type_builder.finish()) as ArrayRef,
            ],
        )
        .expect("failed to build schema record");

        OttersRecord::from(batch)
    }

    /// Returns the selected embedding column name (after build this is the finalized name).
    pub fn vector_column_name(&self) -> &str {
        &self.vector_column_name
    }

    /// Returns the configured inverse norm column name.
    pub fn inv_norm_column_name(&self) -> &str {
        &self.inv_norm_column_name
    }

    /// Returns the configured row id column name.
    pub fn row_id_column_name(&self) -> &str {
        &self.row_id_column_name
    }

    /// Get a column by name (available in both draft and ready states).
    pub fn column(&self, name: &str) -> Option<OttersColumn> {
        match &self.state {
            StoreState::Draft(draft) => draft.column(name),
            StoreState::Ready(ready) => ready.batch.col(name),
        }
    }

    /// Get all column names.
    pub fn column_names(&self) -> Vec<String> {
        self.arrow_schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect()
    }

    /// Get metadata column names (exclude vector, inv norms, and row id).
    pub fn metadata_columns(&self) -> Vec<String> {
        self.column_names()
            .into_iter()
            .filter(|name| {
                name != &self.vector_column_name
                    && name != &self.inv_norm_column_name
                    && name != &self.row_id_column_name
            })
            .collect()
    }

    /// Get the vectors column as a convenience wrapper. Requires the store to be built.
    pub fn vectors(&self) -> OttersColumn {
        self.ensure_ready()
            .expect("Store not built: call build() before accessing vectors")
            .batch
            .col(&self.vector_column_name)
            .expect("vector column must exist")
    }

    /// Borrow the vectors as a [`FixedSizeListArray`]. Requires the store to be built.
    pub fn vectors_array(&self) -> &FixedSizeListArray {
        self.ensure_ready()
            .expect("Store not built: call build() before accessing vectors")
            .batch
            .column(self.vector_index())
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("vector column must be FixedSizeListArray")
    }

    /// Borrow the inverse norms column as a [`Float32Array`]. Requires the store to be built.
    pub fn inv_norms_array(&self) -> &Float32Array {
        self.ensure_ready()
            .expect("Store not built: call build() before accessing inv_norms")
            .batch
            .column(self.inv_norm_index())
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("inv norm column must be Float32Array")
    }

    /// Borrow the row id column as an [`Int64Array`]. Requires the store to be built.
    pub fn row_ids_array(&self) -> &Int64Array {
        self.ensure_ready()
            .expect("Store not built: call build() before accessing row ids")
            .batch
            .column(self.row_id_index())
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("row id column must be Int64Array")
    }

    /// Get the underlying [`RecordBatch`] for ready stores.
    pub fn batch(&self) -> Option<&RecordBatch> {
        match &self.state {
            StoreState::Draft(_) => None,
            StoreState::Ready(ready) => Some(ready.batch.as_ref()),
        }
    }

    /// Clone the underlying [`RecordBatch`] when ready.
    pub fn to_recordbatch(&self) -> Option<RecordBatch> {
        self.batch().cloned()
    }

    /// Retrieve the stats for the most recent query, if available.
    pub fn get_last_query_stats(&self) -> Option<OttersRecord> {
        self.ensure_ready()
            .ok()
            .and_then(|ready| ready.last_query_stats.lock().ok()?.clone())
    }

    /// Update the stored stats for the last query.
    pub(crate) fn set_last_query_stats(&self, stats: OttersRecord) {
        if let Ok(ready) = self.ensure_ready() {
            if let Ok(mut guard) = ready.last_query_stats.lock() {
                *guard = Some(stats);
            }
        }
    }

    /// Start a new query plan for the provided query vector.
    pub fn query(&self, vector: Vec<f32>) -> crate::query::OttersQuery<'_> {
        let mut plan = crate::query::OttersQuery::new(self);
        plan.set_query_vector(vector);
        plan
    }

    fn set_embedding_column(&mut self, selection: EmbeddingSelection) -> Result<(), String> {
        let (source, alias) = match selection {
            EmbeddingSelection::Source(source) => {
                let alias = source.clone();
                (source, alias)
            }
            EmbeddingSelection::SourceAs { source, alias } => (source, alias),
        };
        self.apply_embedding_selection(source, alias)
    }

    fn apply_embedding_selection(&mut self, source: String, alias: String) -> Result<(), String> {
        if alias.is_empty() {
            return Err("Embedding column name cannot be empty".to_string());
        }
        if alias == self.row_id_column_name || alias == self.inv_norm_column_name {
            return Err("Embedding column name cannot match reserved column names".to_string());
        }

        match &mut self.state {
            StoreState::Draft(draft) => {
                let schema = draft.schema();
                let field = schema
                    .column_with_name(&source)
                    .ok_or_else(|| format!("Embedding column '{source}' not found"))?
                    .1;
                match field.data_type() {
                    DataType::FixedSizeList(inner, _)
                        if matches!(inner.data_type(), DataType::Float32) => {}
                    other => {
                        return Err(format!(
                            "Embedding column '{source}' must be FixedSizeList<Float32>, found {other:?}"
                        ));
                    }
                }

                if alias != source && schema.column_with_name(&alias).is_some() {
                    return Err(format!(
                        "Column name '{alias}' already exists; rename it before assigning as the embedding column"
                    ));
                }

                draft.selected_embedding_column = Some(source);
                self.vector_column_name = alias;
                Ok(())
            }
            StoreState::Ready(_) => {
                Err("Store already built; embedding column cannot be changed".to_string())
            }
        }
    }

    fn set_inv_norm_column_name(&mut self, name: String) -> Result<(), String> {
        if matches!(self.state, StoreState::Ready(_)) {
            return Err(
                "Cannot change inverse norm column name after build; use rename_columns instead"
                    .to_string(),
            );
        }
        if name.is_empty() {
            return Err("Inverse norm column name cannot be empty".to_string());
        }
        if name == self.vector_column_name || name == self.row_id_column_name {
            return Err("Inverse norm column name cannot match reserved column names".to_string());
        }
        self.inv_norm_column_name = name;
        Ok(())
    }

    fn set_row_id_column_name(&mut self, name: String) -> Result<(), String> {
        if matches!(self.state, StoreState::Ready(_)) {
            return Err(
                "Cannot change row id column name after build; use rename_columns instead"
                    .to_string(),
            );
        }
        if name.is_empty() {
            return Err("Row id column name cannot be empty".to_string());
        }
        if name == self.vector_column_name || name == self.inv_norm_column_name {
            return Err("Row id column name cannot match reserved column names".to_string());
        }
        self.row_id_column_name = name;
        Ok(())
    }

    fn vector_index(&self) -> usize {
        self.ensure_ready().expect("Store not built").vector_index
    }

    fn inv_norm_index(&self) -> usize {
        self.ensure_ready().expect("Store not built").inv_norm_index
    }

    fn row_id_index(&self) -> usize {
        self.ensure_ready().expect("Store not built").row_id_index
    }

    fn ensure_ready(&self) -> Result<&ReadyState, String> {
        self.ensure_ok()?;
        match &self.state {
            StoreState::Ready(ready) => Ok(ready),
            StoreState::Draft(_) => {
                Err("Store not built: call build() before querying".to_string())
            }
        }
    }

    fn update_special_names(&mut self, renames: &HashMap<String, String>) {
        if let Some(new_name) = renames.get(&self.vector_column_name) {
            self.vector_column_name = new_name.clone();
        }
        if let Some(new_name) = renames.get(&self.inv_norm_column_name) {
            self.inv_norm_column_name = new_name.clone();
        }
        if let Some(new_name) = renames.get(&self.row_id_column_name) {
            self.row_id_column_name = new_name.clone();
        }
    }
}

impl fmt::Display for OttersStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.state {
            StoreState::Draft(draft) => {
                writeln!(
                    f,
                    "OttersStore<Draft>: rows={}, columns={}",
                    draft.len(),
                    draft.schema().fields().len()
                )?;
                if let Some(err) = self.error_message() {
                    writeln!(f, "Pending error: {err}")?;
                }
                writeln!(
                    f,
                    "Call build() after selecting an embedding column to enable querying."
                )?;
                Ok(())
            }
            StoreState::Ready(ready) => write!(f, "{}", ready.batch),
        }
    }
}

fn finalize_store(
    draft: &DraftState,
    vector_column_name: &str,
    inv_norm_column_name: &str,
    row_id_column_name: &str,
) -> Result<ReadyState, String> {
    let embedding_name = draft.selected_embedding_column.as_ref().ok_or_else(|| {
        "Embedding column not set. Call with_embedding_column() before build().".to_string()
    })?;

    let schema = draft.schema();
    let embedding_index = schema
        .index_of(embedding_name)
        .map_err(|e| format!("Embedding column '{embedding_name}' missing: {e}"))?;
    let embedding_field = schema.field(embedding_index);

    let (inner_field, dim) = match embedding_field.data_type() {
        DataType::FixedSizeList(field, dim) => {
            if !matches!(field.data_type(), DataType::Float32) {
                return Err(format!(
                    "Embedding column '{embedding_name}' must contain Float32 values; found {:?}",
                    field.data_type()
                ));
            }
            (field.clone(), *dim)
        }
        other => {
            return Err(format!(
                "Embedding column '{embedding_name}' must be FixedSizeList<Float32>; found {other:?}"
            ));
        }
    };

    let combined = concat_batches(&schema, &draft.batches)
        .map_err(|e| format!("Failed to concatenate batches: {e}"))?;

    if schema.column_with_name(row_id_column_name).is_some() {
        return Err(format!(
            "Column '{row_id_column_name}' already exists. Rename or choose a different row id column name."
        ));
    }

    if schema.column_with_name(inv_norm_column_name).is_some() {
        return Err(format!(
            "Column '{inv_norm_column_name}' already exists. Rename or choose a different inverse norm column name."
        ));
    }

    let arrays = combined.columns().to_vec();
    let embedding_array = arrays[embedding_index].clone();
    let embedding_list = embedding_array
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| "Embedding column must be a FixedSizeListArray<Float32>".to_string())?;

    if embedding_list.null_count() > 0 {
        return Err(
            "Embedding column contains null values; please clean the data before building."
                .to_string(),
        );
    }

    let mut inv_builder = Column::new_float32(inv_norm_column_name.to_string());
    let mut row_builder = Column::new_int64(row_id_column_name.to_string());

    for row_idx in 0..combined.num_rows() {
        let value = embedding_list.value(row_idx);
        let float_array = value
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| "Embedding column must be Float32 values".to_string())?;

        let mut norm_sq = 0.0f32;
        for i in 0..float_array.len() {
            let v = float_array.value(i);
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        let inv_norm = if norm != 0.0 { 1.0 / norm } else { 0.0 };

        inv_builder = inv_builder.append(Some(inv_norm));
        row_builder = row_builder.append(Some(row_idx as i64));
    }

    let inv_norms = inv_builder
        .collect()
        .map_err(|e| format!("Failed to build inverse norm column: {e}"))?;
    let row_ids = row_builder
        .collect()
        .map_err(|e| format!("Failed to build row id column: {e}"))?;

    // Reorder columns: row_id, vectors (with final name), inv_norms, metadata
    let mut final_fields = Vec::with_capacity(schema.fields().len() + 2);
    let mut final_arrays: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len() + 2);

    final_fields.push(Field::new(
        row_id_column_name.to_string(),
        DataType::Int64,
        false,
    ));
    final_arrays.push(row_ids.array().clone());

    final_fields.push(Field::new(
        vector_column_name.to_string(),
        DataType::FixedSizeList(inner_field.clone(), dim),
        embedding_field.is_nullable(),
    ));
    final_arrays.push(embedding_array);

    final_fields.push(Field::new(
        inv_norm_column_name.to_string(),
        DataType::Float32,
        false,
    ));
    final_arrays.push(inv_norms.array().clone());

    for (idx, field) in schema.fields().iter().enumerate() {
        if idx == embedding_index {
            continue;
        }
        if field.name() == row_id_column_name || field.name() == inv_norm_column_name {
            return Err(format!(
                "Column name '{}' conflicts with generated columns. Rename it before building.",
                field.name()
            ));
        }
        final_fields.push((**field).clone());
        final_arrays.push(arrays[idx].clone());
    }

    let final_schema = Arc::new(Schema::new(final_fields));
    let batch = RecordBatch::try_new(final_schema.clone(), final_arrays)
        .map_err(|e| format!("Failed to create finalized RecordBatch: {e}"))?;

    let vector_index = final_schema
        .index_of(vector_column_name)
        .map_err(|e| format!("Vector column '{vector_column_name}' missing after build: {e}"))?;
    let inv_norm_index = final_schema.index_of(inv_norm_column_name).map_err(|e| {
        format!("Inverse norm column '{inv_norm_column_name}' missing after build: {e}")
    })?;
    let row_id_index = final_schema
        .index_of(row_id_column_name)
        .map_err(|e| format!("Row id column '{row_id_column_name}' missing after build: {e}"))?;

    Ok(ReadyState {
        batch: OttersRecord::from(batch),
        dim,
        vector_index,
        inv_norm_index,
        row_id_index,
        last_query_stats: Arc::new(Mutex::new(None)),
    })
}

fn load_batches_with<F>(
    pattern: &str,
    empty_msg: impl Fn(&str) -> String,
    mut reader: F,
) -> Result<Vec<RecordBatch>, String>
where
    F: FnMut(&Path) -> Result<Vec<RecordBatch>, String>,
{
    let paths = expand_glob(pattern)?;
    if paths.is_empty() {
        return Err(empty_msg(pattern));
    }

    let mut batches = Vec::new();
    for path in paths {
        let mut file_batches = reader(path.as_path())?;
        batches.append(&mut file_batches);
    }
    Ok(batches)
}

fn read_parquet_batches(path: &Path) -> Result<Vec<RecordBatch>, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open parquet file {}: {e}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Failed to read parquet metadata {}: {e}", path.display()))?
        .build()
        .map_err(|e| format!("Failed to build parquet reader {}: {e}", path.display()))?;

    reader.collect::<Result<Vec<_>, _>>().map_err(|e| {
        format!(
            "Failed to read parquet batches from {}: {e}",
            path.display()
        )
    })
}

fn read_csv_batches(path: &Path) -> Result<Vec<RecordBatch>, String> {
    let format = Format::default().with_header(true).with_delimiter(b',');

    let schema = {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open CSV file {}: {e}", path.display()))?;
        let mut reader = BufReader::new(file);
        let (schema, _) = format
            .infer_schema(&mut reader, None)
            .map_err(|e| format!("Failed to infer CSV schema from {}: {e}", path.display()))?;
        Arc::new(schema)
    };

    let file =
        File::open(path).map_err(|e| format!("Failed to open CSV file {}: {e}", path.display()))?;
    let reader = ReaderBuilder::new(schema)
        .with_format(format)
        .build(BufReader::new(file))
        .map_err(|e| format!("Failed to build CSV reader {}: {e}", path.display()))?;

    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to read CSV batches from {}: {e}", path.display()))
}

fn expand_glob(pattern: &str) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    for entry in glob(pattern).map_err(|e| format!("Invalid glob pattern '{pattern}': {e}"))? {
        match entry {
            Ok(path) => paths.push(path),
            Err(e) => return Err(format!("Failed to read glob result: {e}")),
        }
    }
    paths.sort();
    Ok(paths)
}

fn rename_batch(
    batch: &RecordBatch,
    renames: &HashMap<String, String>,
) -> Result<RecordBatch, String> {
    let schema = batch.schema();
    let mut fields = Vec::with_capacity(schema.fields().len());
    let mut seen = HashSet::with_capacity(schema.fields().len());

    for field in schema.fields() {
        let new_name = renames
            .get(field.name())
            .cloned()
            .unwrap_or_else(|| field.name().clone());
        if !seen.insert(new_name.clone()) {
            return Err(format!(
                "Renaming would produce duplicate column '{new_name}'"
            ));
        }
        let mut new_field = Field::new(new_name, field.data_type().clone(), field.is_nullable());
        if !field.metadata().is_empty() {
            new_field = new_field.with_metadata(field.metadata().clone());
        }
        fields.push(new_field);
    }

    let metadata = schema.metadata().clone();
    let new_schema = Arc::new(Schema::new_with_metadata(fields, metadata));
    RecordBatch::try_new(new_schema, batch.columns().to_vec())
        .map_err(|e| format!("Failed to rebuild RecordBatch after rename: {e}"))
}

fn apply_ready_renames(
    ready: &mut ReadyState,
    renames: &HashMap<String, String>,
) -> Result<(), String> {
    let new_batch = rename_batch(ready.batch.as_ref(), renames)?;
    ready.batch = OttersRecord::from(new_batch);
    Ok(())
}
