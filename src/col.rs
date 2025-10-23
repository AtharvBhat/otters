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
pub struct Column {
    field: Field,
    array: ArrayRef,
}

impl fmt::Debug for Column {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Column")
            .field("name", &self.field.name())
            .field("dtype", &self.field.data_type())
            .field("len", &self.array.len())
            .field("null_count", &self.array.null_count())
            .finish()
    }
}

impl Column {
    /// Create from existing Arrow array
    pub fn from_arrow(name: impl Into<String>, array: ArrayRef) -> Self {
        let field = Field::new(name, array.data_type().clone(), true);
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
        println!("Column: {} ({:?})", self.name(), self.field.data_type());
        let limit = self.len().min(n);

        for i in 0..limit {
            if self.is_null(i) {
                println!("  [{i}]: NULL");
            } else {
                match self.field.data_type() {
                    DataType::Int32 => {
                        let arr = self.array.as_any().downcast_ref::<Int32Array>().unwrap();
                        let val = arr.value(i);
                        println!("  [{i}]: {val}");
                    }
                    DataType::Int64 => {
                        let arr = self.array.as_any().downcast_ref::<Int64Array>().unwrap();
                        let val = arr.value(i);
                        println!("  [{i}]: {val}");
                    }
                    DataType::Float32 => {
                        let arr = self.array.as_any().downcast_ref::<Float32Array>().unwrap();
                        let val = arr.value(i);
                        println!("  [{i}]: {val:.4}");
                    }
                    DataType::Float64 => {
                        let arr = self.array.as_any().downcast_ref::<Float64Array>().unwrap();
                        let val = arr.value(i);
                        println!("  [{i}]: {val:.4}");
                    }
                    DataType::Utf8 => {
                        let arr = self.array.as_any().downcast_ref::<StringArray>().unwrap();
                        let val = arr.value(i);
                        println!("  [{i}]: \"{val}\"");
                    }
                    DataType::Timestamp(TimeUnit::Millisecond, _) => {
                        let arr = self
                            .array
                            .as_any()
                            .downcast_ref::<TimestampMillisecondArray>()
                            .unwrap();
                        let millis = arr.value(i);
                        if let Some(dt) = DateTime::from_timestamp_millis(millis) {
                            let formatted = dt.format("%Y-%m-%d %H:%M:%S UTC");
                            println!("  [{i}]: {formatted} ({millis})");
                        } else {
                            println!("  [{i}]: Invalid timestamp ({millis})");
                        }
                    }
                    DataType::FixedSizeList(_, dim) => {
                        if let Some(vec) = self.vector_at(i) {
                            let preview: Vec<String> =
                                vec.iter().take(5).map(|v| format!("{v:.4}")).collect();
                            let preview_str = preview.join(", ");
                            if vec.len() > 5 {
                                let more = vec.len() - 5;
                                println!("  [{i}]: [{preview_str}, ... {more} more] (dim={dim})");
                            } else {
                                println!("  [{i}]: [{preview_str}] (dim={dim})");
                            }
                        }
                    }
                    _ => println!("  [{i}]: <unsupported type>"),
                }
            }
        }

        if self.len() > n {
            let more = self.len() - n;
            println!("  ... ({more} more rows)");
        }
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
}

/// Builder for constructing Arrow columns with convenient API
pub struct ColumnBuilder {
    name: String,
    builder: BuilderEnum,
    datetime_format: Option<String>,
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

impl ColumnBuilder {
    /// Create Int32 column builder
    pub fn new_int32(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::Int32(PrimitiveBuilder::new()),
            datetime_format: None,
        }
    }

    /// Create Int64 column builder
    pub fn new_int64(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::Int64(PrimitiveBuilder::new()),
            datetime_format: None,
        }
    }

    /// Create Float32 column builder
    pub fn new_float32(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::Float32(PrimitiveBuilder::new()),
            datetime_format: None,
        }
    }

    /// Create Float64 column builder
    pub fn new_float64(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::Float64(PrimitiveBuilder::new()),
            datetime_format: None,
        }
    }

    /// Create String column builder
    pub fn new_string(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::String(StringBuilder::new()),
            datetime_format: None,
        }
    }

    /// Create Timestamp (millisecond) column builder
    pub fn new_timestamp(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            builder: BuilderEnum::Timestamp(TimestampMillisecondBuilder::new()),
            datetime_format: None,
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
        }
    }

    /// Set custom datetime format for parsing
    pub fn with_datetime_fmt(mut self, format: impl Into<String>) -> Self {
        self.datetime_format = Some(format.into());
        self
    }

    /// Append Int32 value
    pub fn append_i32(&mut self, value: Option<i32>) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::Int32(b) => {
                b.append_option(value);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Int32 builder".to_string(),
            )),
        }
    }

    /// Append Int64 value
    pub fn append_i64(&mut self, value: Option<i64>) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::Int64(b) => {
                b.append_option(value);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Int64 builder".to_string(),
            )),
        }
    }

    /// Append Float32 value
    pub fn append_f32(&mut self, value: Option<f32>) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::Float32(b) => {
                b.append_option(value);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Float32 builder".to_string(),
            )),
        }
    }

    /// Append Float64 value
    pub fn append_f64(&mut self, value: Option<f64>) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::Float64(b) => {
                b.append_option(value);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Float64 builder".to_string(),
            )),
        }
    }

    /// Append String value
    pub fn append_string(&mut self, value: Option<&str>) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::String(b) => {
                b.append_option(value);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected String builder".to_string(),
            )),
        }
    }

    /// Append Timestamp value (as milliseconds since epoch)
    pub fn append_timestamp(&mut self, value: Option<i64>) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::Timestamp(b) => {
                b.append_option(value);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Timestamp builder".to_string(),
            )),
        }
    }

    /// Append datetime from string (auto-parses common formats)
    pub fn append_datetime_str(&mut self, value: Option<&str>) -> Result<(), ColumnError> {
        let millis = match value {
            None => None,
            Some(s) => {
                let parsed = if let Some(fmt) = &self.datetime_format {
                    parse_datetime_fmt(s, fmt)?
                } else {
                    parse_datetime(s)?
                };
                Some(parsed)
            }
        };
        self.append_timestamp(millis)
    }

    /// Append vector value (embedding)
    pub fn append_vector(&mut self, value: Option<&[f32]>) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::Vector(b) => {
                match value {
                    Some(vec) => {
                        let values = b.values();
                        values.extend(vec.iter().copied().map(Some));
                        b.append(true);
                    }
                    None => {
                        b.append(false);
                    }
                }
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Vector builder".to_string(),
            )),
        }
    }

    /// Append many i32 values at once
    pub fn append_i32_slice(&mut self, values: &[Option<i32>]) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::Int32(b) => {
                b.extend(values.iter().copied());
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Int32 builder".to_string(),
            )),
        }
    }

    /// Append many i64 values at once
    pub fn append_i64_slice(&mut self, values: &[Option<i64>]) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::Int64(b) => {
                b.extend(values.iter().copied());
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Int64 builder".to_string(),
            )),
        }
    }

    /// Append many f32 values at once
    pub fn append_f32_slice(&mut self, values: &[Option<f32>]) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::Float32(b) => {
                b.extend(values.iter().copied());
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Float32 builder".to_string(),
            )),
        }
    }

    /// Append many f64 values at once
    pub fn append_f64_slice(&mut self, values: &[Option<f64>]) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::Float64(b) => {
                b.extend(values.iter().copied());
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Float64 builder".to_string(),
            )),
        }
    }

    /// Append many string values at once (bulk append)
    pub fn append_string_slice(&mut self, values: &[Option<&str>]) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::String(b) => {
                b.extend(values.iter().copied());
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected String builder".to_string(),
            )),
        }
    }

    /// Append many vectors at once (bulk append)
    pub fn append_vector_slice(&mut self, values: &[Option<&[f32]>]) -> Result<(), ColumnError> {
        match &mut self.builder {
            BuilderEnum::Vector(b) => {
                for &vec_opt in values {
                    match vec_opt {
                        Some(vec) => {
                            let vals = b.values();
                            vals.extend(vec.iter().copied().map(Some));
                            b.append(true);
                        }
                        None => {
                            b.append(false);
                        }
                    }
                }
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch(
                "Expected Vector builder".to_string(),
            )),
        }
    }

    /// Build the final column (consumes the builder)
    pub fn collect(self) -> Column {
        // finish() returns concrete array types, we need to wrap them in Arc for ArrayRef
        use std::sync::Arc;

        let array: ArrayRef = match self.builder {
            BuilderEnum::Int32(mut b) => Arc::new(b.finish()),
            BuilderEnum::Int64(mut b) => Arc::new(b.finish()),
            BuilderEnum::Float32(mut b) => Arc::new(b.finish()),
            BuilderEnum::Float64(mut b) => Arc::new(b.finish()),
            BuilderEnum::String(mut b) => Arc::new(b.finish()),
            BuilderEnum::Timestamp(mut b) => Arc::new(b.finish()),
            BuilderEnum::Vector(mut b) => Arc::new(b.finish()),
        };

        Column::from_arrow(self.name, array)
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
