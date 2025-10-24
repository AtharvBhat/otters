//! Arrow-native query planning and execution helpers.
//!
//! Provides building blocks for evaluating metadata expressions against an
//! [`ArrowStore`](crate::store::ArrowStore) and extracting candidate row ids
//! prior to vector scoring.

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    TimestampMillisecondArray, UInt32Array,
};
use arrow::compute::kernels::boolean::{and_kleene, or_kleene};
use arrow::compute::kernels::filter::filter;
use arrow::compute::kernels::take::take;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use arrow_array::Datum;
use arrow_ord::cmp as ord_cmp;
use arrow_ord::sort::{SortOptions, sort_to_indices};
use rayon::prelude::*;
use std::sync::Arc;

use crate::expr::{CmpOp, ColumnFilter, CompiledFilter, MetadataPlan, MetricExpr, MetricPlan};
use crate::store::ArrowStore;
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

const SCORE_COLUMN: &str = "_score";

/// Metric to use for vector similarity scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

impl QueryMetric {
    fn sort_descending(self) -> bool {
        matches!(self, Self::Cosine | Self::DotProduct)
    }

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

/// Final query output containing a projected RecordBatch with scores.
#[derive(Debug, Clone)]
pub struct QueryOutput {
    pub batch: RecordBatch,
}

impl QueryOutput {
    pub fn len(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn is_empty(&self) -> bool {
        self.batch.num_rows() == 0
    }
}

/// Builder for executing vector + metadata queries against an [`ArrowStore`].
pub struct ArrowQuery<'a> {
    store: &'a ArrowStore,
    query_vector: Option<Vec<f32>>,
    query_inv_norm: Option<f32>,
    metric: Option<QueryMetric>,
    filter: Option<CompiledFilter>,
    top_k: Option<usize>,
}

impl<'a> ArrowQuery<'a> {
    pub fn new(store: &'a ArrowStore) -> Self {
        Self {
            store,
            query_vector: None,
            query_inv_norm: None,
            metric: None,
            filter: None,
            top_k: None,
        }
    }

    /// Provide the query vector to score against (single-vector queries for now).
    pub fn query(mut self, vector: Vec<f32>) -> Result<Self, String> {
        let expected = self.store.dim() as usize;
        if vector.len() != expected {
            return Err(format!(
                "Query vector dimension {} does not match store dimension {}",
                vector.len(),
                expected
            ));
        }
        let inv_norm = inverse_norm(&vector);
        self.query_vector = Some(vector);
        self.query_inv_norm = Some(inv_norm);
        Ok(self)
    }

    /// Select the metric to use for scoring.
    pub fn metric(mut self, metric: QueryMetric) -> Self {
        self.metric = Some(metric);
        self
    }

    /// Attach a compiled metadata expression.
    pub fn filter(mut self, filter: CompiledFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Limit the result set to top `k` rows (post-filtered).
    ///
    /// Similarity metrics (`Cosine`, `DotProduct`) return the highest scores first,
    /// while distance metrics (`Euclidean`) return the smallest distances first.
    pub fn take(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Execute the query plan and return the materialized results.
    pub fn collect(self) -> Result<QueryOutput, String> {
        let query = self
            .query_vector
            .ok_or_else(|| "Query vector not provided".to_string())?;
        let inv_norm = self.query_inv_norm.unwrap_or_else(|| inverse_norm(&query));

        let metadata_plan = self
            .filter
            .as_ref()
            .map(|f| f.metadata_clauses.clone())
            .unwrap_or_default();
        let metric_plan = self
            .filter
            .as_ref()
            .map(metric_plan_from)
            .unwrap_or_default();

        let inferred_metric = metric_from_plan(&metric_plan)?;
        let metric = match (self.metric, inferred_metric) {
            (Some(explicit), Some(inferred)) if explicit != inferred => {
                return Err(format!(
                    "Metric {explicit:?} specified in query conflicts with metric {inferred:?} in filter"
                ));
            }
            (Some(explicit), _) => explicit,
            (None, Some(inferred)) => inferred,
            (None, None) => QueryMetric::Cosine,
        };

        let selection = apply_metadata_filters(self.store, &metadata_plan)?;

        let scored = score_candidates(
            self.store,
            &query,
            inv_norm,
            metric,
            &selection,
            &metric_plan,
        )?;

        if scored.is_empty() {
            return build_empty_output(self.store);
        }

        let (row_ids_array, scores_array) = build_score_arrays(&scored);
        let sorted_indices =
            sort_scores(&scores_array, metric, self.top_k).map_err(|e| e.to_string())?;
        let sorted_row_ids =
            take(row_ids_array.as_ref(), &sorted_indices, None).map_err(|e| e.to_string())?;
        let sorted_scores =
            take(scores_array.as_ref(), &sorted_indices, None).map_err(|e| e.to_string())?;

        let row_ids = sorted_row_ids
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| "Failed to downcast sorted row ids".to_string())?
            .clone();
        let scores = sorted_scores
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| "Failed to downcast sorted scores".to_string())?
            .clone();

        materialize_output(self.store, row_ids, scores)
    }
}

/// Apply metadata filters to the store, producing a selection mask and the
/// filtered row ids. Uses Rayon to evaluate clause masks in parallel.
pub fn apply_metadata_filters(
    store: &ArrowStore,
    plan: &MetadataPlan,
) -> Result<MetadataSelection, String> {
    let batch = store.batch();

    if plan.is_empty() {
        return Ok(MetadataSelection::all(
            batch.num_rows(),
            store.row_ids_array().clone(),
        ));
    }

    let clause_masks: Result<Vec<BooleanArray>, String> = plan
        .par_iter()
        .map(|clause| evaluate_clause(batch, clause))
        .collect();

    let mut clause_masks = clause_masks?;
    debug_assert!(!clause_masks.is_empty());

    // Reduce clause masks with logical AND (since plan is AND of clauses)
    let mut combined = clause_masks
        .pop()
        .expect("clause masks should be non-empty after evaluation");
    for mask in clause_masks.iter() {
        combined = and_kleene(&combined, mask).map_err(|e| e.to_string())?;
    }

    let filtered_ids = filter_int64(store.row_ids_array(), &combined)?;
    Ok(MetadataSelection {
        mask: combined,
        row_ids: filtered_ids,
    })
}

/// Extract the metric-only plan from a compiled expression.
/// Provided here as a convenience wrapper.
pub fn metric_plan_from(compiled: &crate::expr::CompiledFilter) -> MetricPlan {
    compiled.metric_plan()
}

fn evaluate_clause(batch: &RecordBatch, clause: &[ColumnFilter]) -> Result<BooleanArray, String> {
    let mut iter = clause.iter();
    let first = iter
        .next()
        .ok_or_else(|| "Clause must contain at least one predicate".to_string())?;

    let mut mask = evaluate_filter(batch, first)?;
    for filter in iter {
        let rhs = evaluate_filter(batch, filter)?;
        mask = or_kleene(&mask, &rhs).map_err(|e| e.to_string())?;
    }

    Ok(mask)
}

fn evaluate_filter(batch: &RecordBatch, filter: &ColumnFilter) -> Result<BooleanArray, String> {
    match filter {
        ColumnFilter::Numeric { column, cmp, rhs } => {
            let array = column_array(batch, column)?;
            evaluate_numeric(array, *cmp, rhs)
        }
        ColumnFilter::String { column, cmp, rhs } => {
            let array = column_array(batch, column)?;
            evaluate_utf8(array, *cmp, rhs)
        }
    }
}

fn column_array(batch: &RecordBatch, column: &str) -> Result<ArrayRef, String> {
    let idx = batch
        .schema()
        .index_of(column)
        .map_err(|e| format!("Column '{column}' not found in batch: {e}"))?;
    Ok(batch.column(idx).clone())
}

fn evaluate_numeric(
    array: ArrayRef,
    cmp: CmpOp,
    rhs: &crate::expr::NumericLiteral,
) -> Result<BooleanArray, String> {
    use crate::expr::NumericLiteral;
    match array.data_type() {
        arrow::datatypes::DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| "Expected Int32 array".to_string())?;
            let value = match rhs {
                NumericLiteral::I64(v) => *v as i32,
                NumericLiteral::F64(_) => {
                    return Err("Expected integer literal for Int32 comparison".to_string());
                }
            };
            compare_int32(arr, cmp, value).map_err(|e| e.to_string())
        }
        arrow::datatypes::DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| "Expected Int64 array".to_string())?;
            let value = match rhs {
                NumericLiteral::I64(v) => *v,
                NumericLiteral::F64(_) => {
                    return Err("Expected integer literal for Int64 comparison".to_string());
                }
            };
            compare_int64(arr, cmp, value).map_err(|e| e.to_string())
        }
        arrow::datatypes::DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| "Expected Float32 array".to_string())?;
            let value = match rhs {
                NumericLiteral::I64(v) => *v as f32,
                NumericLiteral::F64(v) => *v as f32,
            };
            compare_float32(arr, cmp, value).map_err(|e| e.to_string())
        }
        arrow::datatypes::DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| "Expected Float64 array".to_string())?;
            let value = match rhs {
                NumericLiteral::I64(v) => *v as f64,
                NumericLiteral::F64(v) => *v,
            };
            compare_float64(arr, cmp, value).map_err(|e| e.to_string())
        }
        arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, _) => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .ok_or_else(|| "Expected Timestamp(Millisecond) array".to_string())?;
            let value = match rhs {
                NumericLiteral::I64(v) => *v,
                NumericLiteral::F64(_) => {
                    return Err("Expected integer literal for timestamp comparison".to_string());
                }
            };
            compare_timestamp_millis(arr, cmp, value).map_err(|e| e.to_string())
        }
        other => Err(format!(
            "Unsupported numeric column type for filtering: {other:?}"
        )),
    }
}

fn evaluate_utf8(array: ArrayRef, cmp: CmpOp, rhs: &str) -> Result<BooleanArray, String> {
    match array.data_type() {
        arrow::datatypes::DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| "Expected Utf8 array".to_string())?;
            compare_utf8(arr, cmp, rhs).map_err(|e| e.to_string())
        }
        other => Err(format!(
            "Unsupported string column type for filtering: {other:?}"
        )),
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
    let scalar = Int64Array::new_scalar(value);
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

fn filter_int64(array: &Int64Array, mask: &BooleanArray) -> Result<Int64Array, String> {
    let filtered = filter(array, mask).map_err(|e| e.to_string())?;
    let filtered = filtered
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| "Filtered row id column did not downcast to Int64Array".to_string())?;
    Ok(filtered.clone())
}

fn metric_from_plan(plan: &MetricPlan) -> Result<Option<QueryMetric>, String> {
    let mut metric: Option<QueryMetric> = None;
    for clause in plan {
        for filter in clause {
            let candidate: QueryMetric = filter.metric.into();
            if let Some(existing) = metric {
                if existing != candidate {
                    return Err(format!(
                        "Mixed metric predicates are not supported: {existing:?} vs {candidate:?}"
                    ));
                }
            } else {
                metric = Some(candidate);
            }
        }
    }
    Ok(metric)
}

fn score_candidates(
    store: &ArrowStore,
    query: &[f32],
    query_inv_norm: f32,
    metric: QueryMetric,
    selection: &MetadataSelection,
    metric_plan: &MetricPlan,
) -> Result<Vec<ScoredRow>, String> {
    let vectors = store.vectors_array();
    let values_array = vectors
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| "Vector array payload is not Float32".to_string())?;
    let values = values_array.values();
    let dim = store.dim() as usize;
    let inv_norms = store.inv_norms_array();

    let candidate_ids: Vec<i64> = selection.row_ids.values().to_vec();

    let results: Vec<ScoredRow> = candidate_ids
        .par_iter()
        .filter_map(|row_id| {
            if *row_id < 0 {
                return None;
            }
            let idx = *row_id as usize;
            if vectors.is_null(idx) {
                return None;
            }
            let offset = idx.checked_mul(dim)?;
            let end = offset.checked_add(dim)?;
            if end > values.len() {
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

    Ok(results)
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

fn sort_scores(
    scores: &Float32Array,
    metric: QueryMetric,
    limit: Option<usize>,
) -> Result<UInt32Array, ArrowError> {
    let options = SortOptions {
        descending: metric.sort_descending(),
        nulls_first: false,
    };
    sort_to_indices(scores, Some(options), limit)
}

fn materialize_output(
    store: &ArrowStore,
    row_ids: Int64Array,
    scores: Float32Array,
) -> Result<QueryOutput, String> {
    let indices = to_u32_indices(&row_ids)?;
    let store_batch = store.batch();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(store_batch.num_columns() + 1);

    for column in store_batch.columns() {
        let taken = take(column.as_ref(), &indices, None).map_err(|e| e.to_string())?;
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
    let batch = RecordBatch::try_new(schema, columns).map_err(|e| e.to_string())?;
    Ok(QueryOutput { batch })
}

fn to_u32_indices(row_ids: &Int64Array) -> Result<UInt32Array, String> {
    let mut values = Vec::with_capacity(row_ids.len());
    for i in 0..row_ids.len() {
        let id = row_ids.value(i);
        if id < 0 {
            return Err("Row id cannot be negative".to_string());
        }
        if id > u32::MAX as i64 {
            return Err(format!("Row id {id} exceeds u32::MAX"));
        }
        values.push(id as u32);
    }
    Ok(UInt32Array::from(values))
}

fn build_empty_output(store: &ArrowStore) -> Result<QueryOutput, String> {
    let empty_ids = Int64Array::from(Vec::<i64>::new());
    let empty_scores = Float32Array::from(Vec::<f32>::new());
    materialize_output(store, empty_ids, empty_scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::col::ColumnBuilder;
    use crate::expr::{col, cosine};
    use crate::type_utils::DataType;
    use std::collections::HashMap;

    fn build_store() -> ArrowStore {
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.8, 0.2, 0.0],
            vec![0.6, 0.4, 0.0],
        ];

        let mut age_builder = ColumnBuilder::new_int32("age");
        age_builder.append_i32(Some(25)).unwrap();
        age_builder.append_i32(Some(35)).unwrap();
        age_builder.append_i32(Some(45)).unwrap();
        let ages = age_builder.collect();

        ArrowStore::builder(3)
            .with_vectors(vectors)
            .unwrap()
            .with_metadata_column("age", ages)
            .unwrap()
            .build()
            .unwrap()
    }

    #[test]
    fn cosine_query_topk() {
        let store = build_store();
        let output = ArrowQuery::new(&store)
            .query(vec![1.0, 0.0, 0.0])
            .unwrap()
            .metric(QueryMetric::Cosine)
            .take(2)
            .collect()
            .unwrap();

        assert_eq!(output.batch.num_rows(), 2);
        let row_ids = output
            .batch
            .column_by_name("_row_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(row_ids.value(0), 0);
        assert_eq!(row_ids.value(1), 1);

        let scores = output
            .batch
            .column_by_name("_score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert!(scores.value(0) >= scores.value(1));
    }

    #[test]
    fn metadata_and_metric_filters() {
        let store = build_store();

        let mut schema = HashMap::new();
        schema.insert("age".to_string(), DataType::Int32);
        let expr = cosine().gt(0.7) & col("age").gte(40);
        let compiled = expr.compile(&schema).unwrap();

        let output = ArrowQuery::new(&store)
            .query(vec![1.0, 0.0, 0.0])
            .unwrap()
            .filter(compiled)
            .collect()
            .unwrap();

        assert_eq!(output.batch.num_rows(), 1);
        let row_ids = output
            .batch
            .column_by_name("_row_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(row_ids.value(0), 2);

        let score = output
            .batch
            .column_by_name("_score")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(0);
        assert!(score > 0.7);
    }
}
