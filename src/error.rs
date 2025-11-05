use std::io;
use std::path::PathBuf;

use arrow::datatypes::DataType;
use arrow::error::ArrowError;
use glob::{GlobError, PatternError};
use parquet::errors::ParquetError;
use thiserror::Error;

use crate::datetime::ParseDateTimeError;
use crate::expr::ExprError;
use crate::query::QueryMetric;

#[derive(Error, Debug)]
pub enum OttersError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Query(#[from] QueryError),
    #[error(transparent)]
    Column(#[from] ColumnError),
    #[error(transparent)]
    Expr(#[from] ExprError),
    #[error(transparent)]
    DateTime(#[from] ParseDateTimeError),
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    #[error(transparent)]
    Parquet(#[from] ParquetError),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("Store validation failed:\n{0}")]
    StoreValidation(String),
    #[error("Glob pattern '{pattern}' is invalid: {source}")]
    GlobPattern {
        pattern: String,
        #[source]
        source: PatternError,
    },
    #[error("Failed to read glob entry: {source}")]
    GlobWalk {
        #[source]
        source: GlobError,
    },
    #[error("Rayon task failed: {0}")]
    RayonJoin(String),
}

#[derive(Error, Debug)]
pub enum StoreError {
    #[error("Store is already built")]
    AlreadyBuilt,
    #[error("Store must be built before this operation")]
    NotBuilt,
    #[error("Embedding column not set; call with_embedding_column() before build()")]
    EmbeddingColumnNotSet,
    #[error("Embedding column name cannot be empty")]
    EmptyEmbeddingColumnName,
    #[error("Column '{column}' not found. Available columns: {available:?}")]
    ColumnNotFound {
        column: String,
        available: Vec<String>,
    },
    #[error(
        "Cannot create store: schema column count ({schema}) does not match data column count ({data})"
    )]
    ColumnCountMismatch { schema: usize, data: usize },
    #[error("Cannot create store without columns")]
    EmptyColumns,
    #[error("Columns have inconsistent row counts")]
    ColumnLengthMismatch,
    #[error("Cannot build store from empty batch list")]
    EmptyBatchList,
    #[error("Column '{column}' already exists")]
    ColumnAlreadyExists { column: String },
    #[error("Column '{column}' must be {expected}")]
    ColumnTypeMismatch {
        column: String,
        expected: &'static str,
    },
    #[error("Row id column '{column}' is invalid or inconsistent with store rows")]
    RowIdIntegrity { column: String },
    #[error("No {format} files matched pattern '{pattern}'")]
    NoFilesMatched {
        pattern: String,
        format: &'static str,
    },
    #[error("Embedding column '{column}' has incompatible Arrow type: {reason}")]
    EmbeddingColumnType {
        column: String,
        reason: &'static str,
    },
    #[error("Embedding column '{column}' must contain Float32 values")]
    EmbeddingVectorType { column: String },
    #[error("Embedding column '{column}' must contain Float32 or Float64 values")]
    EmbeddingValueType { column: String },
    #[error("Embedding column '{column}' contains null values")]
    EmbeddingContainsNulls { column: String },
    #[error("Embedding column '{column}' contains null vector entries")]
    EmbeddingVectorsContainNulls { column: String },
    #[error("Embedding column '{column}' contains empty vectors")]
    EmbeddingEmptyVectors { column: String },
    #[error(
        "Embedding column '{column}' has inconsistent vector dimensions: expected {expected}, found {found}"
    )]
    EmbeddingDimensionMismatch {
        column: String,
        expected: i32,
        found: i32,
    },
    #[error("Embedding column '{column}' contains only null values; unable to infer dimension")]
    EmbeddingAllNull { column: String },
    #[error(
        "Embedding column '{column}' contains inconsistent lengths: expected {expected}, found {found}"
    )]
    EmbeddingLengthMismatch {
        column: String,
        expected: usize,
        found: usize,
    },
    #[error("Failed to concatenate record batches: {source}")]
    Concatenate {
        #[source]
        source: ArrowError,
    },
    #[error("Failed to build record batch: {source}")]
    BuildBatch {
        #[source]
        source: ArrowError,
    },
    #[error("Failed to finalize store record batch: {source}")]
    Finalize {
        #[source]
        source: ArrowError,
    },
    #[error("{action} for '{path}' failed: {source}")]
    IoAction {
        path: PathBuf,
        action: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("{action} for '{path}' failed: {source}")]
    ArrowAction {
        path: PathBuf,
        action: &'static str,
        #[source]
        source: ArrowError,
    },
    #[error("{action} for '{path}' failed: {source}")]
    ParquetAction {
        path: PathBuf,
        action: &'static str,
        #[source]
        source: ParquetError,
    },
}

#[derive(Error, Debug)]
pub enum ColumnError {
    #[error("Column builder type mismatch: {detail}")]
    TypeMismatch { detail: &'static str },
    #[error("Failed to parse datetime value '{input}': {source}")]
    DateTime {
        input: String,
        #[source]
        source: ParseDateTimeError,
    },
    #[error(transparent)]
    Arrow(#[from] ArrowError),
}

#[derive(Error, Debug, Clone)]
pub enum QueryError {
    #[error("Query vector dimension {actual} does not match store dimension {expected}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Query vector not provided")]
    MissingQueryVector,
    #[error("Column '{column}' not found in record batch")]
    ColumnNotFound { column: String },
    #[error("Metric {explicit:?} conflicts with metric {inferred:?} inferred from filters")]
    MetricConflict {
        explicit: QueryMetric,
        inferred: QueryMetric,
    },
    #[error("Metric predicates must agree on metric type: {existing:?} vs {candidate:?}")]
    MixedMetrics {
        existing: QueryMetric,
        candidate: QueryMetric,
    },
    #[error(
        "Metric filter references embedding column '{column}' but store is configured with '{expected}'"
    )]
    MetricColumnMismatch { column: String, expected: String },
    #[error("Clause must contain at least one predicate")]
    EmptyClause,
    #[error("Expected array of type {expected}")]
    ExpectedArrayType { expected: &'static str },
    #[error("Unsupported numeric column type for filtering: {datatype:?}")]
    UnsupportedNumericColumn { datatype: DataType },
    #[error("Unsupported string column type for filtering: {datatype:?}")]
    UnsupportedStringColumn { datatype: DataType },
    #[error("Expected integer literal for {kind} comparison")]
    ExpectedIntegerLiteral { kind: &'static str },
    #[error("Vector array payload is not Float32")]
    InvalidVectorPayload,
    #[error("Row id cannot be negative")]
    NegativeRowId,
    #[error("Row id {value} exceeds u32::MAX")]
    RowIdOverflow { value: i64 },
}
