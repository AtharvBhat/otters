//! Arrow-based column storage
//!
//! Thin wrapper around Apache Arrow arrays for convenient column operations.

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, FixedSizeListBuilder, Float32Array, Float64Array,
    Int32Array, Int64Array, PrimitiveBuilder, StringArray, StringBuilder,
    TimestampMillisecondArray, TimestampMillisecondBuilder,
};
use arrow::datatypes::{DataType, Field, Float32Type, Float64Type, Int32Type, Int64Type, TimeUnit};
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use std::fmt;

// Re-export Arrow types for convenience
pub use arrow::datatypes::DataType as ArrowDataType;
pub use arrow::datatypes::Field as ArrowField;

#[derive(Debug)]
pub enum ColumnError {
    TypeMismatch(String),
    ParseError(String),
    ArrowError(String),
}

impl fmt::Display for ColumnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnError::TypeMismatch(msg) => write!(f, "Type mismatch: {msg}"),
            ColumnError::ParseError(msg) => write!(f, "Parse error: {msg}"),
            ColumnError::ArrowError(msg) => write!(f, "Arrow error: {msg}"),
        }
    }
}

impl std::error::Error for ColumnError {}

impl From<arrow::error::ArrowError> for ColumnError {
    fn from(e: arrow::error::ArrowError) -> Self {
        ColumnError::ArrowError(e.to_string())
    }
}

/// Arrow-backed column - just a thin wrapper around Arrow arrays
pub struct OttersColumn {
    field: Field,
    array: ArrayRef,
}

const COLUMN_DISPLAY_PREVIEW_ROWS: usize = 10;

impl fmt::Debug for OttersColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OttersColumn")
            .field("name", &self.field.name())
            .field("dtype", &self.field.data_type())
            .field("len", &self.array.len())
            .field("null_count", &self.array.null_count())
            .finish()
    }
}

impl OttersColumn {
    /// Create from existing Arrow array
    pub fn from_arrow(name: impl Into<String>, array: ArrayRef) -> Self {
        let field = Field::new(name, array.data_type().clone(), true);
        Self::from_field(field, array)
    }

    /// Create from existing Arrow [`Field`] and array, preserving metadata.
    pub fn from_field(field: Field, array: ArrayRef) -> Self {
        Self { field, array }
    }

    /// Get column name
    pub fn name(&self) -> &str {
        self.field.name()
    }

    /// Get Arrow data type
    pub fn dtype(&self) -> &DataType {
        self.field.data_type()
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.array.is_empty()
    }

    /// Get null count
    pub fn null_count(&self) -> usize {
        self.array.null_count()
    }

    /// Check if value at index is null
    pub fn is_null(&self, index: usize) -> bool {
        self.array.is_null(index)
    }

    /// Get underlying Arrow array
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// Get Field metadata
    pub fn field(&self) -> &Field {
        &self.field
    }

    /// Display first 5 rows
    pub fn head(&self) {
        self.head_n(5)
    }

    /// Display first n rows
    pub fn head_n(&self, n: usize) {
        println!("{}", ColumnPreview::new(self, n));
    }

    /// Get typed values (i32)
    pub fn i32_values(&self) -> Option<&Int32Array> {
        self.array.as_any().downcast_ref::<Int32Array>()
    }

    /// Get typed values (i64)
    pub fn i64_values(&self) -> Option<&Int64Array> {
        self.array.as_any().downcast_ref::<Int64Array>()
    }

    /// Get typed values (f32)
    pub fn f32_values(&self) -> Option<&Float32Array> {
        self.array.as_any().downcast_ref::<Float32Array>()
    }

    /// Get typed values (f64)
    pub fn f64_values(&self) -> Option<&Float64Array> {
        self.array.as_any().downcast_ref::<Float64Array>()
    }

    /// Get typed values (string)
    pub fn string_values(&self) -> Option<&StringArray> {
        self.array.as_any().downcast_ref::<StringArray>()
    }

    /// Get typed values (datetime as timestamp millis)
    pub fn datetime_values(&self) -> Option<&TimestampMillisecondArray> {
        self.array
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
    }

    /// Get typed values (vector/embedding as FixedSizeList of Float32)
    pub fn vector_values(&self) -> Option<&FixedSizeListArray> {
        self.array.as_any().downcast_ref::<FixedSizeListArray>()
    }

    /// Get a single vector as Vec<f32> (copies the data)
    pub fn vector_at(&self, index: usize) -> Option<Vec<f32>> {
        let vectors = self.vector_values()?;
        if index >= vectors.len() {
            return None;
        }

        let value_array = vectors.value(index);
        let float_array = value_array.as_any().downcast_ref::<Float32Array>()?;
        Some(float_array.values().to_vec())
    }

    /// Get the dimension of vectors in this column
    pub fn vector_dim(&self) -> Option<i32> {
        match self.field.data_type() {
            DataType::FixedSizeList(_, dim) => Some(*dim),
            _ => None,
        }
    }

    fn format_value(&self, index: usize) -> String {
        if self.is_null(index) {
            return "NULL".to_string();
        }

        match self.field.data_type() {
            DataType::Int32 => self
                .array
                .as_any()
                .downcast_ref::<Int32Array>()
                .map(|arr| arr.value(index).to_string())
                .unwrap_or_else(|| "<invalid Int32 column>".to_string()),
            DataType::Int64 => self
                .array
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(|arr| arr.value(index).to_string())
                .unwrap_or_else(|| "<invalid Int64 column>".to_string()),
            DataType::Float32 => self
                .array
                .as_any()
                .downcast_ref::<Float32Array>()
                .map(|arr| format!("{:.4}", arr.value(index)))
                .unwrap_or_else(|| "<invalid Float32 column>".to_string()),
            DataType::Float64 => self
                .array
                .as_any()
                .downcast_ref::<Float64Array>()
                .map(|arr| format!("{:.4}", arr.value(index)))
                .unwrap_or_else(|| "<invalid Float64 column>".to_string()),
            DataType::Utf8 => self
                .array
                .as_any()
                .downcast_ref::<StringArray>()
                .map(|arr| format!("\"{}\"", arr.value(index)))
                .unwrap_or_else(|| "<invalid Utf8 column>".to_string()),
            DataType::Timestamp(TimeUnit::Millisecond, _) => self
                .array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .map(|arr| {
                    let millis = arr.value(index);
                    match DateTime::from_timestamp_millis(millis) {
                        Some(dt) => format!("{} ({millis})", dt.format("%Y-%m-%d %H:%M:%S UTC")),
                        None => format!("Invalid timestamp ({millis})"),
                    }
                })
                .unwrap_or_else(|| "<invalid Timestamp column>".to_string()),
            DataType::FixedSizeList(_, dim) => {
                if let Some(vec) = self.vector_at(index) {
                    let preview: Vec<String> =
                        vec.iter().take(5).map(|v| format!("{v:.4}")).collect();
                    let preview_str = preview.join(", ");
                    if vec.len() > 5 {
                        let more = vec.len() - 5;
                        format!("[{preview_str}, ... {more} more] (dim={dim})")
                    } else {
                        format!("[{preview_str}] (dim={dim})")
                    }
                } else {
                    "<invalid FixedSizeList column>".to_string()
                }
            }
            _ => "<unsupported type>".to_string(),
        }
    }
}

/// Builder for constructing Arrow columns with convenient API
pub struct Column {
    name: String,
    builder: BuilderEnum,
    datetime_format: Option<String>,
    error: Option<ColumnError>,
}

enum BuilderEnum {
    Int32(PrimitiveBuilder<Int32Type>),
    Int64(PrimitiveBuilder<Int64Type>),
    Float32(PrimitiveBuilder<Float32Type>),
    Float64(PrimitiveBuilder<Float64Type>),
    String(StringBuilder),
    Timestamp(TimestampMillisecondBuilder),
    Vector(FixedSizeListBuilder<PrimitiveBuilder<Float32Type>>),
}

impl Column {
    /// Create Int32 column builder
    pub fn new_int32(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::Int32(PrimitiveBuilder::new()),
            datetime_format: None,
            error: None,
        }
    }

    /// Create Int64 column builder
    pub fn new_int64(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::Int64(PrimitiveBuilder::new()),
            datetime_format: None,
            error: None,
        }
    }

    /// Create Float32 column builder
    pub fn new_float32(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::Float32(PrimitiveBuilder::new()),
            datetime_format: None,
            error: None,
        }
    }

    /// Create Float64 column builder
    pub fn new_float64(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::Float64(PrimitiveBuilder::new()),
            datetime_format: None,
            error: None,
        }
    }

    /// Create String column builder
    pub fn new_string(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::String(StringBuilder::new()),
            datetime_format: None,
            error: None,
        }
    }

    /// Create Timestamp (millisecond) column builder
    pub fn new_timestamp(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::Timestamp(TimestampMillisecondBuilder::new()),
            datetime_format: None,
            error: None,
        }
    }

    /// Create Vector (FixedSizeList of Float32) column builder
    pub fn new_vector(name: impl Into<String>, dim: i32) -> Self {
        let values_builder = PrimitiveBuilder::<Float32Type>::new();
        let vector_builder = FixedSizeListBuilder::new(values_builder, dim);
        Self {
            name: name.into(),
            builder: BuilderEnum::Vector(vector_builder),
            datetime_format: None,
            error: None,
        }
    }

    /// Set custom datetime format for parsing
    pub fn with_datetime_fmt(mut self, format: impl Into<String>) -> Self {
        self.datetime_format = Some(format.into());
        self
    }

    /// Append one or more values using a unified interface.
    ///
    /// Accepts single values (`builder.append(Some(value))`) or collections
    /// like slices/arrays of `Option<T>`.
    pub fn append<V>(&mut self, values: V) -> &mut Self
    where
        V: ColumnValues,
    {
        if self.error.is_some() {
            return self;
        }

        let target = ColumnAppendTarget {
            builder: &mut self.builder,
        };
        if let Err(err) = values.append_into(target) {
            self.error = Some(err);
        }
        self
    }

    /// Append datetime from string (auto-parses common formats)
    pub fn append_datetime_str(&mut self, value: Option<&str>) -> &mut Self {
        let millis = match value {
            None => None,
            Some(s) => {
                let parsed = if let Some(fmt) = &self.datetime_format {
                    parse_datetime_fmt(s, fmt)
                } else {
                    parse_datetime(s)
                };
                match parsed {
                    Ok(ts) => Some(ts),
                    Err(err) => {
                        self.error = Some(err);
                        return self;
                    }
                }
            }
        };
        self.append([millis])
    }

    /// Build the final column (consumes the builder)
    pub fn collect(self) -> Result<OttersColumn, ColumnError> {
        // finish() returns concrete array types, we need to wrap them in Arc for ArrayRef
        use std::sync::Arc;

        let Column {
            name,
            builder,
            datetime_format: _,
            error,
        } = self;

        if let Some(err) = error {
            return Err(err);
        }

        let array: ArrayRef = match builder {
            BuilderEnum::Int32(mut b) => Arc::new(b.finish()),
            BuilderEnum::Int64(mut b) => Arc::new(b.finish()),
            BuilderEnum::Float32(mut b) => Arc::new(b.finish()),
            BuilderEnum::Float64(mut b) => Arc::new(b.finish()),
            BuilderEnum::String(mut b) => Arc::new(b.finish()),
            BuilderEnum::Timestamp(mut b) => Arc::new(b.finish()),
            BuilderEnum::Vector(mut b) => Arc::new(b.finish()),
        };

        Ok(OttersColumn::from_arrow(name, array))
    }
}

struct ColumnPreview<'a> {
    column: &'a OttersColumn,
    limit: usize,
}

impl<'a> ColumnPreview<'a> {
    fn new(column: &'a OttersColumn, limit: usize) -> Self {
        Self { column, limit }
    }
}

impl fmt::Display for ColumnPreview<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let column = self.column;
        let len = column.len();
        let preview_len = len.min(self.limit);

        writeln!(f, "Column: {} ({:?})", column.name(), column.dtype())?;

        for idx in 0..preview_len {
            writeln!(f, "  [{idx}]: {}", column.format_value(idx))?;
        }

        if len > preview_len {
            writeln!(f, "  ... ({} more rows)", len - preview_len)?;
        }

        write!(f, "Total rows: {len}")
    }
}

impl fmt::Display for OttersColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ColumnPreview::new(self, COLUMN_DISPLAY_PREVIEW_ROWS).fmt(f)
    }
}

/// Public wrapper passed to `ColumnValues` implementations.
pub struct ColumnAppendTarget<'a> {
    builder: &'a mut BuilderEnum,
}

/// Internal helper implemented for types supported by [`Column::append`].
trait ColumnType: Copy {
    fn append_option(builder: &mut BuilderEnum, value: Option<Self>) -> Result<(), ColumnError>;

    fn append_iter<I>(builder: &mut BuilderEnum, iter: I) -> Result<(), ColumnError>
    where
        I: IntoIterator<Item = Option<Self>>,
    {
        for value in iter {
            Self::append_option(builder, value)?;
        }
        Ok(())
    }

    fn append_slice(builder: &mut BuilderEnum, values: &[Option<Self>]) -> Result<(), ColumnError>
    where
        Option<Self>: Copy,
    {
        for &value in values {
            Self::append_option(builder, value)?;
        }
        Ok(())
    }
}

/// Helper trait powering [`Column::append`].
pub trait ColumnValues {
    fn append_into(self, target: ColumnAppendTarget<'_>) -> Result<(), ColumnError>;
}

macro_rules! impl_column_type_numeric {
    ($ty:ty, $variant:ident, $err:literal) => {
        impl ColumnType for $ty {
            fn append_option(
                builder: &mut BuilderEnum,
                value: Option<Self>,
            ) -> Result<(), ColumnError> {
                match builder {
                    BuilderEnum::$variant(b) => {
                        b.append_option(value);
                        Ok(())
                    }
                    _ => Err(ColumnError::TypeMismatch($err.to_string())),
                }
            }

            fn append_slice(
                builder: &mut BuilderEnum,
                values: &[Option<Self>],
            ) -> Result<(), ColumnError>
            where
                Option<Self>: Copy,
            {
                match builder {
                    BuilderEnum::$variant(b) => {
                        b.extend(values.iter().copied());
                        Ok(())
                    }
                    _ => Err(ColumnError::TypeMismatch($err.to_string())),
                }
            }
        }
    };
}

impl_column_type_numeric!(i32, Int32, "Expected Int32 builder");
impl_column_type_numeric!(f32, Float32, "Expected Float32 builder");
impl_column_type_numeric!(f64, Float64, "Expected Float64 builder");

impl ColumnType for i64 {
    fn append_option(builder: &mut BuilderEnum, value: Option<Self>) -> Result<(), ColumnError> {
        match builder {
            BuilderEnum::Int64(b) => {
                b.append_option(value);
                Ok(())
            }
            BuilderEnum::Timestamp(b) => {
                b.append_option(value);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Int64 or Timestamp builder".to_string(),
            )),
        }
    }

    fn append_slice(builder: &mut BuilderEnum, values: &[Option<Self>]) -> Result<(), ColumnError>
    where
        Option<Self>: Copy,
    {
        match builder {
            BuilderEnum::Int64(b) => {
                b.extend(values.iter().copied());
                Ok(())
            }
            BuilderEnum::Timestamp(b) => {
                b.extend(values.iter().copied());
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Int64 or Timestamp builder".to_string(),
            )),
        }
    }
}

impl ColumnType for &str {
    fn append_option(builder: &mut BuilderEnum, value: Option<Self>) -> Result<(), ColumnError> {
        match builder {
            BuilderEnum::String(b) => {
                b.append_option(value);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected String builder".to_string(),
            )),
        }
    }

    fn append_slice(builder: &mut BuilderEnum, values: &[Option<Self>]) -> Result<(), ColumnError>
    where
        Option<Self>: Copy,
    {
        match builder {
            BuilderEnum::String(b) => {
                b.extend(values.iter().copied());
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected String builder".to_string(),
            )),
        }
    }
}

impl ColumnType for &[f32] {
    fn append_option(builder: &mut BuilderEnum, value: Option<Self>) -> Result<(), ColumnError> {
        match builder {
            BuilderEnum::Vector(b) => {
                match value {
                    Some(vec) => {
                        let values = b.values();
                        values.extend(vec.iter().copied().map(Some));
                        b.append(true);
                    }
                    None => b.append(false),
                }
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Vector builder".to_string(),
            )),
        }
    }
}

impl<T> ColumnValues for &[Option<T>]
where
    T: ColumnType,
    Option<T>: Copy,
{
    fn append_into(self, target: ColumnAppendTarget<'_>) -> Result<(), ColumnError> {
        T::append_slice(target.builder, self)
    }
}

impl<T> ColumnValues for Vec<Option<T>>
where
    T: ColumnType,
{
    fn append_into(self, target: ColumnAppendTarget<'_>) -> Result<(), ColumnError> {
        T::append_iter(target.builder, self)
    }
}

impl<T> ColumnValues for Option<T>
where
    T: ColumnType,
{
    fn append_into(self, target: ColumnAppendTarget<'_>) -> Result<(), ColumnError> {
        T::append_option(target.builder, self)
    }
}

impl<T, const N: usize> ColumnValues for [Option<T>; N]
where
    T: ColumnType,
{
    fn append_into(self, target: ColumnAppendTarget<'_>) -> Result<(), ColumnError> {
        T::append_iter(target.builder, IntoIterator::into_iter(self))
    }
}

// DateTime parsing helpers
fn parse_datetime(s: &str) -> Result<i64, ColumnError> {
    // Try ISO 8601 / RFC 3339
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc).timestamp_millis());
    }

    // Try YYYY-MM-DD
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        if let Some(dt) = date.and_hms_opt(0, 0, 0) {
            return Ok(dt.and_utc().timestamp_millis());
        }
    }

    // Try YYYY-MM-DD HH:MM:SS
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Ok(dt.and_utc().timestamp_millis());
    }

    Err(ColumnError::ParseError(format!(
        "Cannot parse '{s}' as datetime. Supported formats: ISO 8601, YYYY-MM-DD, YYYY-MM-DD HH:MM:SS"
    )))
}

fn parse_datetime_fmt(s: &str, format: &str) -> Result<i64, ColumnError> {
    // Try as datetime first
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, format) {
        return Ok(dt.and_utc().timestamp_millis());
    }

    // Try as date only
    if let Ok(date) = NaiveDate::parse_from_str(s, format) {
        if let Some(dt) = date.and_hms_opt(0, 0, 0) {
            return Ok(dt.and_utc().timestamp_millis());
        }
    }

    Err(ColumnError::ParseError(format!(
        "Cannot parse '{s}' with format '{format}'"
    )))
}
