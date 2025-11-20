//! Arrow-native query planning and execution helpers.
//!
//! Provides building blocks for evaluating metadata expressions against an
//! [`OttersStore`](crate::store::OttersStore) and extracting candidate row ids
//! prior to vector scoring.

use arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Float64Builder,
    Int32Array, Int64Array, Int64Builder, StringArray, StringBuilder, TimestampMillisecondArray,
    UInt32Array,
};
use arrow::compute::kernels::boolean::{and_kleene, or_kleene};
use arrow::compute::kernels::filter::filter;
use arrow::compute::kernels::take::take;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use arrow_array::Datum;
use arrow_ord::cmp as ord_cmp;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

use crate::error::{OttersError, QueryError};
use crate::expr::{
    CmpOp, ColumnFilter, CompiledFilter, DataType as ExprDataType, Expr, MetadataPlan, MetricExpr,
    MetricPlan,
};
use crate::record::OttersRecord;
use crate::store::OttersStore;
use crate::vec_compute::{
    cosine_similarity, dot_product, euclidean_distance_squared, inverse_norm,
};

/// Result of applying metadata filters: a boolean mask over the store rows
/// and the corresponding filtered row ids.
#[derive(Debug, Clone)]
pub struct MetadataSelection {
    pub mask: BooleanArray,
    pub row_ids: Int64Array,
}

impl MetadataSelection {
    pub fn all(len: usize, row_ids: Int64Array) -> Self {
        let mask = BooleanArray::from(vec![true; len]);
        Self { mask, row_ids }
    }
}

const SCORE_COLUMN: &str = "score";

#[derive(Debug, Clone)]
pub struct MetadataColumnStats {
    pub column: String,
    pub evaluated: u64,
    pub passed: u64,
}

impl MetadataColumnStats {
    fn from_mask(column: String, mask: &BooleanArray) -> Self {
        let evaluated = (mask.len() - mask.null_count()) as u64;
        let passed = mask
            .iter()
            .filter(|value| matches!(value, Some(true)))
            .count() as u64;
        Self {
            column,
            evaluated,
            passed,
        }
    }
}

/// Metric to use for vector similarity scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

impl QueryMetric {
    fn as_metric_expr(self) -> MetricExpr {
        match self {
            Self::Cosine => MetricExpr::Cosine,
            Self::DotProduct => MetricExpr::DotProduct,
            Self::Euclidean => MetricExpr::Euclidean,
        }
    }
}

impl From<MetricExpr> for QueryMetric {
    fn from(expr: MetricExpr) -> Self {
        match expr {
            MetricExpr::Cosine => QueryMetric::Cosine,
            MetricExpr::DotProduct => QueryMetric::DotProduct,
            MetricExpr::Euclidean => QueryMetric::Euclidean,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ScoredRow {
    row_id: i64,
    score: f32,
}

#[derive(Debug, Clone)]
struct ScoringResult {
    rows: Vec<ScoredRow>,
    warning: Option<String>,
}

/// Final query output containing a projected RecordBatch with scores.
#[derive(Debug, Clone)]
pub struct QueryOutput {
    pub batch: OttersRecord,
    /// Warning produced during scoring (e.g., skipped candidates).
    pub warning: Option<String>,
}

impl QueryOutput {
    pub fn len(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn is_empty(&self) -> bool {
        self.batch.num_rows() == 0
    }

    /// Borrow the underlying record batch.
    pub fn record_batch(&self) -> &RecordBatch {
        self.batch.as_ref()
    }

    /// Consume the output and return the underlying record batch.
    pub fn into_record_batch(self) -> RecordBatch {
        self.batch.into_inner()
    }
}

impl fmt::Display for QueryOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.batch)?;
        if let Some(warning) = &self.warning {
            write!(f, "\nWarning: {warning}")?;
        }
        Ok(())
    }
}

/// Builder for executing vector + metadata queries against an [`OttersStore`].
pub struct OttersQuery<'a> {
    store: &'a OttersStore,
    query_vector: Option<Vec<f32>>,
    query_inv_norm: Option<f32>,
    metric: Option<QueryMetric>,
    filter_expr: Option<Expr>,
    top_k: Option<usize>,
    order_desc: Option<bool>,
    error: Option<OttersError>,
}

impl<'a> OttersQuery<'a> {
    pub fn new(store: &'a OttersStore) -> Self {
        Self {
            store,
            query_vector: None,
            query_inv_norm: None,
            metric: None,
            filter_expr: None,
            top_k: None,
            order_desc: None,
            error: None,
        }
    }

    pub fn set_error(&mut self, err: OttersError) {
        if self.error.is_none() {
            self.error = Some(err);
        }
    }

    pub fn with_query_vec(mut self, vector: Vec<f32>) -> Self {
        if self.error.is_none() {
            if let Err(err) = self.set_query_vector_internal(vector) {
                self.set_error(err);
            }
        }
        self
    }

    fn set_query_vector_internal(&mut self, vector: Vec<f32>) -> Result<(), OttersError> {
        self.store.ensure_ready()?;

        let expected = self.store.try_dim()? as usize;
        if vector.len() != expected {
            return Err(QueryError::DimensionMismatch {
                expected,
                actual: vector.len(),
            }
            .into());
        }

        let inv_norm = inverse_norm(&vector);
        self.query_vector = Some(vector);
        self.query_inv_norm = Some(inv_norm);
        Ok(())
    }

    /// Sort results in descending score order.
    pub fn order_by_desc(mut self) -> Self {
        if self.error.is_none() {
            self.order_desc = Some(true);
        }
        self
    }

    /// Sort results in ascending score order.
    pub fn order_by_asc(mut self) -> Self {
        if self.error.is_none() {
            self.order_desc = Some(false);
        }
        self
    }

    /// Select the metric to use for scoring.
    pub fn metric(mut self, metric: QueryMetric) -> Self {
        if self.error.is_none() {
            self.metric = Some(metric);
        }
        self
    }

    /// Attach a metadata expression that will be compiled during execution.
    pub fn filter(mut self, filter: Expr) -> Self {
        if self.error.is_none() {
            self.filter_expr = Some(filter);
        }
        self
    }

    /// Limit the result set to top `k` rows (post-filtered).
    ///
    /// Similarity metrics (`Cosine`, `DotProduct`) default to highest-first ordering,
    /// while distance metrics (`Euclidean`) default to smallest-first ordering.
    /// Explicit calls to [`order_by_desc`](Self::order_by_desc) or
    /// [`order_by_asc`](Self::order_by_asc) override this default.
    pub fn take(mut self, k: usize) -> Self {
        if self.error.is_none() {
            self.top_k = Some(k);
        }
        self
    }

    /// Execute the query plan and return the materialized results.
    pub fn collect(self) -> Result<QueryOutput, OttersError> {
        if let Some(err) = self.error {
            return Err(err);
        }
        QueryExecution::from_query(self)?.execute()
    }
}

struct QueryExecution<'a> {
    store: &'a OttersStore,
    query_vector: Vec<f32>,
    query_inv_norm: f32,
    metric_hint: Option<QueryMetric>,
    filter_expr: Option<Expr>,
    top_k: Option<usize>,
    order_desc: Option<bool>,
    durations: Vec<(&'static str, f64)>,
    metadata_stats: Vec<MetadataColumnStats>,
    warning: Option<String>,
}

impl<'a> QueryExecution<'a> {
    fn from_query(query: OttersQuery<'a>) -> Result<Self, OttersError> {
        let OttersQuery {
            store,
            query_vector,
            query_inv_norm,
            metric,
            filter_expr,
            top_k,
            order_desc,
            error: _,
        } = query;

        store.ensure_ready()?;

        let query = query_vector.ok_or(QueryError::MissingQueryVector)?;
        let inv_norm = query_inv_norm.unwrap_or_else(|| inverse_norm(&query));

        Ok(Self {
            store,
            query_vector: query,
            query_inv_norm: inv_norm,
            metric_hint: metric,
            filter_expr,
            top_k,
            order_desc,
            durations: Vec::new(),
            metadata_stats: Vec::new(),
            warning: None,
        })
    }

    fn execute(mut self) -> Result<QueryOutput, OttersError> {
        let total_start = Instant::now();

        let compiled = self.compile_filter()?;
        let metadata_plan = compiled
            .as_ref()
            .map(|f| f.metadata_clauses.clone())
            .unwrap_or_default();
        let metric_plan = compiled.as_ref().map(metric_plan_from).unwrap_or_default();
        let metric = self.resolve_metric(&metric_plan)?;

        let selection = self.apply_metadata(&metadata_plan)?;
        let candidate_count = selection.row_ids.len() as u64;
        let scored = self.score(&selection, metric, &metric_plan)?;
        let scored_count = scored.rows.len() as u64;
        let output = self.materialize(scored.rows, metric, scored.warning.clone())?;
        self.warning = scored.warning;

        self.durations
            .push(("total_query", duration_ms(total_start)));
        self.record_stats((candidate_count, scored_count))?;
        Ok(output)
    }

    fn compile_filter(&mut self) -> Result<Option<CompiledFilter>, OttersError> {
        match self.filter_expr.take() {
            Some(expr) => {
                let start = Instant::now();
                let compiled = compile_expr(self.store, expr)?;
                self.durations.push(("compile_filter", duration_ms(start)));
                Ok(Some(compiled))
            }
            None => Ok(None),
        }
    }

    fn resolve_metric(&self, metric_plan: &MetricPlan) -> Result<QueryMetric, OttersError> {
        self.validate_metric_columns(metric_plan)?;
        let inferred = metric_from_plan(metric_plan)?;
        match (self.metric_hint, inferred) {
            (Some(explicit), Some(inferred_metric)) if explicit != inferred_metric => {
                Err(QueryError::MetricConflict {
                    explicit,
                    inferred: inferred_metric,
                }
                .into())
            }
            (Some(explicit), _) => Ok(explicit),
            (None, Some(inferred_metric)) => Ok(inferred_metric),
            (None, None) => Ok(QueryMetric::Cosine),
        }
    }

    fn validate_metric_columns(&self, metric_plan: &MetricPlan) -> Result<(), OttersError> {
        let expected = self.store.try_embedding_column_name()?;
        for clause in metric_plan {
            for filter in clause {
                if let Some(column) = &filter.column {
                    if column != expected {
                        return Err(QueryError::MetricColumnMismatch {
                            column: column.clone(),
                            expected: expected.to_string(),
                        }
                        .into());
                    }
                }
            }
        }
        Ok(())
    }

    fn apply_metadata(&mut self, plan: &MetadataPlan) -> Result<MetadataSelection, OttersError> {
        let start = Instant::now();
        let (selection, stats) = apply_metadata_filters(self.store, plan)?;
        self.durations
            .push(("apply_metadata_filters", duration_ms(start)));
        self.metadata_stats = stats;
        Ok(selection)
    }

    fn score(
        &mut self,
        selection: &MetadataSelection,
        metric: QueryMetric,
        metric_plan: &MetricPlan,
    ) -> Result<ScoringResult, OttersError> {
        let start = Instant::now();
        let scored = score_candidates(
            self.store,
            &self.query_vector,
            self.query_inv_norm,
            metric,
            selection,
            metric_plan,
        )?;
        self.durations
            .push(("score_candidates", duration_ms(start)));
        Ok(scored)
    }

    fn materialize(
        &mut self,
        mut scored: Vec<ScoredRow>,
        metric: QueryMetric,
        warning: Option<String>,
    ) -> Result<QueryOutput, OttersError> {
        if scored.is_empty() {
            let materialize_start = Instant::now();
            let output = build_empty_output(self.store, warning)?;
            self.durations
                .push(("materialize_output", duration_ms(materialize_start)));
            return Ok(output);
        }

        let default_desc = matches!(metric, QueryMetric::Cosine | QueryMetric::DotProduct);
        let should_sort = self.order_desc.is_some() || self.top_k.is_some();
        let desc = self.order_desc.unwrap_or(default_desc);

        if should_sort {
            scored.sort_by(|a, b| match (b.score.partial_cmp(&a.score), desc) {
                (Some(ord), true) => ord,
                (Some(ord), false) => ord.reverse(),
                (None, true) => Ordering::Equal,
                (None, false) => Ordering::Equal,
            });
        }

        if let Some(limit) = self.top_k {
            if scored.len() > limit {
                scored.truncate(limit);
            }
        }

        let prepare_start = Instant::now();
        let (row_ids_array, scores_array) = build_score_arrays(&scored);
        self.durations
            .push(("prepare_results", duration_ms(prepare_start)));

        let row_ids = (*row_ids_array).clone();
        let scores = (*scores_array).clone();

        let materialize_start = Instant::now();
        let output = materialize_output(self.store, row_ids, scores, warning)?;
        self.durations
            .push(("materialize_output", duration_ms(materialize_start)));
        Ok(output)
    }

    fn record_stats(&self, vector_stats: (u64, u64)) -> Result<(), OttersError> {
        let stats_batch =
            build_query_stats_batch(&self.durations, &self.metadata_stats, Some(vector_stats))?;
        self.store.set_last_query_stats(stats_batch);
        Ok(())
    }
}

fn compile_expr(store: &OttersStore, expr: Expr) -> Result<CompiledFilter, OttersError> {
    let mut schema_map: HashMap<String, ExprDataType> = HashMap::new();
    for field in store.arrow_schema().fields() {
        if let Some(dtype) = arrow_to_otters_type(field.data_type()) {
            schema_map.insert(field.name().clone(), dtype);
        }
    }
    expr.compile(&schema_map).map_err(OttersError::from)
}

fn arrow_to_otters_type(data_type: &DataType) -> Option<ExprDataType> {
    match data_type {
        DataType::Int32 => Some(ExprDataType::Int32),
        DataType::Int64 => Some(ExprDataType::Int64),
        DataType::Float32 => Some(ExprDataType::Float32),
        DataType::Float64 => Some(ExprDataType::Float64),
        DataType::Utf8 => Some(ExprDataType::String),
        DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, _) => {
            Some(ExprDataType::DateTime)
        }
        _ => None,
    }
}

/// Apply metadata filters to the store, producing a selection mask and the
/// filtered row ids. Uses Rayon to evaluate clause masks in parallel.
pub fn apply_metadata_filters(
    store: &OttersStore,
    plan: &MetadataPlan,
) -> Result<(MetadataSelection, Vec<MetadataColumnStats>), OttersError> {
    let batch = store
        .batch()
        .expect("store must be built before applying metadata filters");

    let row_ids = row_ids_array(store)?;

    if plan.is_empty() {
        return Ok((
            MetadataSelection::all(batch.num_rows(), row_ids.clone()),
            Vec::new(),
        ));
    }

    let clause_results: Result<Vec<(BooleanArray, Vec<MetadataColumnStats>)>, OttersError> = plan
        .par_iter()
        .map(|clause| evaluate_clause(batch, clause))
        .collect();

    let mut clause_masks: Vec<BooleanArray> = Vec::new();
    let mut stats_map: HashMap<String, MetadataColumnStats> = HashMap::new();
    for (mask, clause_stats) in clause_results? {
        for stat in clause_stats {
            let entry = stats_map
                .entry(stat.column.clone())
                .or_insert(MetadataColumnStats {
                    column: stat.column.clone(),
                    evaluated: 0,
                    passed: 0,
                });
            entry.evaluated += stat.evaluated;
            entry.passed += stat.passed;
        }
        clause_masks.push(mask);
    }

    debug_assert!(!clause_masks.is_empty());

    // Reduce clause masks with logical AND (since plan is AND of clauses)
    let mut combined = clause_masks
        .pop()
        .expect("clause masks should be non-empty after evaluation");
    for mask in clause_masks.iter() {
        combined = and_kleene(&combined, mask)?;
    }

    let filtered_ids = filter_int64(&row_ids, &combined)?;
    let mut aggregated_stats: Vec<MetadataColumnStats> = stats_map.into_values().collect();
    aggregated_stats.sort_by(|a, b| a.column.cmp(&b.column));
    Ok((
        MetadataSelection {
            mask: combined,
            row_ids: filtered_ids,
        },
        aggregated_stats,
    ))
}

/// Extract the metric-only plan from a compiled expression.
/// Provided here as a convenience wrapper.
pub fn metric_plan_from(compiled: &crate::expr::CompiledFilter) -> MetricPlan {
    compiled.metric_plan()
}

fn evaluate_clause(
    batch: &RecordBatch,
    clause: &[ColumnFilter],
) -> Result<(BooleanArray, Vec<MetadataColumnStats>), OttersError> {
    let mut iter = clause.iter();
    let first = iter.next().ok_or(QueryError::EmptyClause)?;

    let (mut mask, first_stats) = evaluate_filter(batch, first)?;
    let mut stats = vec![first_stats];
    for filter in iter {
        let (rhs, rhs_stats) = evaluate_filter(batch, filter)?;
        stats.push(rhs_stats);
        mask = or_kleene(&mask, &rhs)?;
    }

    Ok((mask, stats))
}

fn evaluate_filter(
    batch: &RecordBatch,
    filter: &ColumnFilter,
) -> Result<(BooleanArray, MetadataColumnStats), OttersError> {
    match filter {
        ColumnFilter::Numeric { column, cmp, rhs } => {
            let array = column_array(batch, column)?;
            let mask = evaluate_numeric(array, *cmp, rhs)?;
            let stats = MetadataColumnStats::from_mask(column.clone(), &mask);
            Ok((mask, stats))
        }
        ColumnFilter::String { column, cmp, rhs } => {
            let array = column_array(batch, column)?;
            let mask = evaluate_utf8(array, *cmp, rhs)?;
            let stats = MetadataColumnStats::from_mask(column.clone(), &mask);
            Ok((mask, stats))
        }
    }
}

fn column_array(batch: &RecordBatch, column: &str) -> Result<ArrayRef, OttersError> {
    let idx = batch
        .schema()
        .index_of(column)
        .map_err(|_| QueryError::ColumnNotFound {
            column: column.to_string(),
        })?;
    Ok(batch.column(idx).clone())
}

fn evaluate_numeric(
    array: ArrayRef,
    cmp: CmpOp,
    rhs: &crate::expr::NumericLiteral,
) -> Result<BooleanArray, OttersError> {
    use crate::expr::NumericLiteral;
    match array.data_type() {
        arrow::datatypes::DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or(QueryError::ExpectedArrayType { expected: "Int32" })?;
            let value = match rhs {
                NumericLiteral::I64(v) => *v as i32,
                NumericLiteral::F64(_) => {
                    return Err(QueryError::ExpectedIntegerLiteral { kind: "Int32" }.into());
                }
            };
            compare_int32(arr, cmp, value).map_err(OttersError::from)
        }
        arrow::datatypes::DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or(QueryError::ExpectedArrayType { expected: "Int64" })?;
            let value = match rhs {
                NumericLiteral::I64(v) => *v,
                NumericLiteral::F64(_) => {
                    return Err(QueryError::ExpectedIntegerLiteral { kind: "Int64" }.into());
                }
            };
            compare_int64(arr, cmp, value).map_err(OttersError::from)
        }
        arrow::datatypes::DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().ok_or(
                QueryError::ExpectedArrayType {
                    expected: "Float32",
                },
            )?;
            let value = match rhs {
                NumericLiteral::I64(v) => *v as f32,
                NumericLiteral::F64(v) => *v as f32,
            };
            compare_float32(arr, cmp, value).map_err(OttersError::from)
        }
        arrow::datatypes::DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().ok_or(
                QueryError::ExpectedArrayType {
                    expected: "Float64",
                },
            )?;
            let value = match rhs {
                NumericLiteral::I64(v) => *v as f64,
                NumericLiteral::F64(v) => *v,
            };
            compare_float64(arr, cmp, value).map_err(OttersError::from)
        }
        arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .ok_or(QueryError::ExpectedArrayType {
                    expected: "Timestamp(Millisecond)",
                })?;
            let value = match rhs {
                NumericLiteral::I64(v) => *v,
                NumericLiteral::F64(_) => {
                    return Err(QueryError::ExpectedIntegerLiteral { kind: "timestamp" }.into());
                }
            };
            compare_timestamp_millis(arr, cmp, value).map_err(OttersError::from)
        }
        other => Err(QueryError::UnsupportedNumericColumn {
            datatype: other.clone(),
        }
        .into()),
    }
}

fn evaluate_utf8(array: ArrayRef, cmp: CmpOp, rhs: &str) -> Result<BooleanArray, OttersError> {
    match array.data_type() {
        arrow::datatypes::DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(QueryError::ExpectedArrayType { expected: "Utf8" })?;
            compare_utf8(arr, cmp, rhs).map_err(OttersError::from)
        }
        other => Err(QueryError::UnsupportedStringColumn {
            datatype: other.clone(),
        }
        .into()),
    }
}

fn compare_int32(array: &Int32Array, cmp: CmpOp, value: i32) -> Result<BooleanArray, ArrowError> {
    let scalar = Int32Array::new_scalar(value);
    compare_datum(array, &scalar, cmp)
}

fn compare_int64(array: &Int64Array, cmp: CmpOp, value: i64) -> Result<BooleanArray, ArrowError> {
    let scalar = Int64Array::new_scalar(value);
    compare_datum(array, &scalar, cmp)
}

fn compare_float32(
    array: &Float32Array,
    cmp: CmpOp,
    value: f32,
) -> Result<BooleanArray, ArrowError> {
    let scalar = Float32Array::new_scalar(value);
    compare_datum(array, &scalar, cmp)
}

fn compare_float64(
    array: &Float64Array,
    cmp: CmpOp,
    value: f64,
) -> Result<BooleanArray, ArrowError> {
    let scalar = Float64Array::new_scalar(value);
    compare_datum(array, &scalar, cmp)
}

fn compare_timestamp_millis(
    array: &TimestampMillisecondArray,
    cmp: CmpOp,
    value: i64,
) -> Result<BooleanArray, ArrowError> {
    let scalar = TimestampMillisecondArray::new_scalar(value);
    compare_datum(array, &scalar, cmp)
}

fn compare_utf8(array: &StringArray, cmp: CmpOp, value: &str) -> Result<BooleanArray, ArrowError> {
    let scalar = StringArray::new_scalar(value);
    compare_datum(array, &scalar, cmp)
}

fn compare_datum(lhs: &dyn Datum, rhs: &dyn Datum, cmp: CmpOp) -> Result<BooleanArray, ArrowError> {
    match cmp {
        CmpOp::Eq => ord_cmp::eq(lhs, rhs),
        CmpOp::Neq => ord_cmp::neq(lhs, rhs),
        CmpOp::Lt => ord_cmp::lt(lhs, rhs),
        CmpOp::Lte => ord_cmp::lt_eq(lhs, rhs),
        CmpOp::Gt => ord_cmp::gt(lhs, rhs),
        CmpOp::Gte => ord_cmp::gt_eq(lhs, rhs),
    }
}

fn filter_int64(array: &Int64Array, mask: &BooleanArray) -> Result<Int64Array, OttersError> {
    let filtered = filter(array, mask)?;
    let filtered = filtered
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or(QueryError::ExpectedArrayType { expected: "Int64" })
        .map_err(OttersError::from)?;
    Ok(filtered.clone())
}

fn metric_from_plan(plan: &MetricPlan) -> Result<Option<QueryMetric>, OttersError> {
    let mut metric: Option<QueryMetric> = None;
    for clause in plan {
        for filter in clause {
            let candidate: QueryMetric = filter.metric.into();
            if let Some(existing) = metric {
                if existing != candidate {
                    return Err(QueryError::MixedMetrics {
                        existing,
                        candidate,
                    }
                    .into());
                }
            } else {
                metric = Some(candidate);
            }
        }
    }
    Ok(metric)
}

fn vectors_array(store: &OttersStore) -> Result<FixedSizeListArray, OttersError> {
    let column_name = store.try_embedding_column_name()?.to_string();
    let column = store.column(&column_name)?;
    let array = column.array().clone();
    let downcasted = array
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or(QueryError::ExpectedArrayType {
            expected: "FixedSizeList<Float32>",
        })
        .map_err(OttersError::from)?;
    Ok(downcasted.clone())
}

fn inv_norms_array(store: &OttersStore) -> Result<Float32Array, OttersError> {
    let column_name = store.try_inv_norm_column_name()?.to_string();
    let column = store.column(&column_name)?;
    let array = column.array().clone();
    let downcasted = array
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or(QueryError::ExpectedArrayType {
            expected: "Float32",
        })
        .map_err(OttersError::from)?;
    Ok(downcasted.clone())
}

fn row_ids_array(store: &OttersStore) -> Result<Int64Array, OttersError> {
    let column_name = store.row_id_column_name().to_string();
    let column = store.column(&column_name)?;
    let array = column.array().clone();
    let downcasted = array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or(QueryError::ExpectedArrayType { expected: "Int64" })
        .map_err(OttersError::from)?;
    Ok(downcasted.clone())
}

fn score_candidates(
    store: &OttersStore,
    query: &[f32],
    query_inv_norm: f32,
    metric: QueryMetric,
    selection: &MetadataSelection,
    metric_plan: &MetricPlan,
) -> Result<ScoringResult, OttersError> {
    let vectors = vectors_array(store)?;
    let values_array = vectors
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or(QueryError::InvalidVectorPayload)
        .map_err(OttersError::from)?;
    let values = values_array.values();
    let dim = store.try_dim()? as usize;
    let inv_norms = inv_norms_array(store)?;

    let row_values = selection.row_ids.values();
    let offset = selection.row_ids.offset();
    let len = selection.row_ids.len();
    let candidate_slice = &row_values.as_ref()[offset..offset + len];

    let skipped = AtomicUsize::new(0);

    let results: Vec<ScoredRow> = candidate_slice
        .par_iter()
        .enumerate()
        .filter_map(|(i, row_id)| {
            if selection.row_ids.is_null(i) || *row_id < 0 {
                skipped.fetch_add(1, AtomicOrdering::Relaxed);
                return None;
            }
            let idx = *row_id as usize;
            if vectors.is_null(idx) {
                skipped.fetch_add(1, AtomicOrdering::Relaxed);
                return None;
            }
            let offset = idx.checked_mul(dim)?;
            let end = offset.checked_add(dim)?;
            if end > values.len() {
                skipped.fetch_add(1, AtomicOrdering::Relaxed);
                return None;
            }
            let embedding = &values[offset..end];
            let score = match metric {
                QueryMetric::Cosine => {
                    let inv = inv_norms.value(idx);
                    cosine_similarity(query, embedding, query_inv_norm, inv)
                }
                QueryMetric::DotProduct => dot_product(query, embedding),
                QueryMetric::Euclidean => euclidean_distance_squared(query, embedding),
            };
            if metric_plan_passes(metric_plan, score, metric) {
                Some(ScoredRow {
                    row_id: *row_id,
                    score,
                })
            } else {
                None
            }
        })
        .collect();

    let warning = match skipped.load(AtomicOrdering::Relaxed) {
        0 => None,
        n => Some(format!(
            "Skipped {n} candidates with invalid embeddings or row ids; ignore if this is expected."
        )),
    };

    Ok(ScoringResult {
        rows: results,
        warning,
    })
}

fn metric_plan_passes(plan: &MetricPlan, score: f32, metric: QueryMetric) -> bool {
    if plan.is_empty() {
        return true;
    }

    for clause in plan {
        if clause.is_empty() {
            continue;
        }
        let mut clause_ok = false;
        for filter in clause {
            if filter.metric != metric.as_metric_expr() {
                continue;
            }
            if compare_score(score, filter.cmp, filter.threshold) {
                clause_ok = true;
                break;
            }
        }
        if !clause_ok {
            return false;
        }
    }
    true
}

fn compare_score(score: f32, cmp: CmpOp, threshold: f32) -> bool {
    match cmp {
        CmpOp::Eq => score == threshold,
        CmpOp::Neq => score != threshold,
        CmpOp::Lt => score < threshold,
        CmpOp::Lte => score <= threshold,
        CmpOp::Gt => score > threshold,
        CmpOp::Gte => score >= threshold,
    }
}

fn build_score_arrays(results: &[ScoredRow]) -> (Arc<Int64Array>, Arc<Float32Array>) {
    let row_ids: Vec<i64> = results.iter().map(|r| r.row_id).collect();
    let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
    (
        Arc::new(Int64Array::from(row_ids)),
        Arc::new(Float32Array::from(scores)),
    )
}

fn materialize_output(
    store: &OttersStore,
    row_ids: Int64Array,
    scores: Float32Array,
    warning: Option<String>,
) -> Result<QueryOutput, OttersError> {
    let indices = to_u32_indices(&row_ids)?;
    let store_batch = store
        .batch()
        .expect("store must be built before materializing output");
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(store_batch.num_columns() + 1);

    for column in store_batch.columns() {
        let taken = take(column.as_ref(), &indices, None)?;
        columns.push(taken);
    }

    columns.push(Arc::new(scores) as ArrayRef);

    let mut fields: Vec<Field> = store_batch
        .schema()
        .fields()
        .iter()
        .map(|f| (**f).clone())
        .collect();
    fields.push(Field::new(SCORE_COLUMN, DataType::Float32, false));

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns)?;
    Ok(QueryOutput {
        batch: OttersRecord::from(batch),
        warning,
    })
}

fn to_u32_indices(row_ids: &Int64Array) -> Result<UInt32Array, OttersError> {
    let mut values = Vec::with_capacity(row_ids.len());
    for i in 0..row_ids.len() {
        let id = row_ids.value(i);
        if id < 0 {
            return Err(QueryError::NegativeRowId.into());
        }
        if id > u32::MAX as i64 {
            return Err(QueryError::RowIdOverflow { value: id }.into());
        }
        values.push(id as u32);
    }
    Ok(UInt32Array::from(values))
}

fn build_empty_output(
    store: &OttersStore,
    warning: Option<String>,
) -> Result<QueryOutput, OttersError> {
    let empty_ids = Int64Array::from(Vec::<i64>::new());
    let empty_scores = Float32Array::from(Vec::<f32>::new());
    materialize_output(store, empty_ids, empty_scores, warning)
}

fn duration_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn build_query_stats_batch(
    durations: &[(&str, f64)],
    metadata_stats: &[MetadataColumnStats],
    vector_stats: Option<(u64, u64)>,
) -> Result<OttersRecord, OttersError> {
    let mut category_builder = StringBuilder::new();
    let mut name_builder = StringBuilder::new();
    let mut duration_builder = Float64Builder::new();
    let mut evaluated_builder = Int64Builder::new();
    let mut passed_builder = Int64Builder::new();

    for (step, duration) in durations {
        category_builder.append_value("duration");
        name_builder.append_value(*step);
        duration_builder.append_value(*duration);
        evaluated_builder.append_null();
        passed_builder.append_null();
    }

    for stat in metadata_stats {
        category_builder.append_value("metadata_column");
        name_builder.append_value(&stat.column);
        duration_builder.append_null();
        evaluated_builder.append_value(stat.evaluated as i64);
        passed_builder.append_value(stat.passed as i64);
    }

    if let Some((evaluated, passed)) = vector_stats {
        category_builder.append_value("vector_scoring");
        name_builder.append_value("candidates_vs_passed");
        duration_builder.append_null();
        evaluated_builder.append_value(evaluated as i64);
        passed_builder.append_value(passed as i64);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("duration_ms", DataType::Float64, true),
        Field::new("evaluated", DataType::Int64, true),
        Field::new("passed", DataType::Int64, true),
    ]));

    let columns: Vec<ArrayRef> = vec![
        Arc::new(category_builder.finish()) as ArrayRef,
        Arc::new(name_builder.finish()) as ArrayRef,
        Arc::new(duration_builder.finish()) as ArrayRef,
        Arc::new(evaluated_builder.finish()) as ArrayRef,
        Arc::new(passed_builder.finish()) as ArrayRef,
    ];

    Ok(OttersRecord::from(RecordBatch::try_new(schema, columns)?))
}
