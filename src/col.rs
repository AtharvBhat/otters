use crate::type_utils::DataType;
use bitvec::prelude::*;
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use std::fmt;

#[derive(Debug)]
pub enum ColumnData {
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    String(Vec<String>),
    DateTime(Vec<i64>), // millis since epoch
}

#[derive(Debug)]
pub struct Column {
    name: String,
    dtype: DataType,
    data: ColumnData,
    null_mask: BitVec,
    datetime_format: Option<String>, // For custom datetime parsing
}

#[derive(Debug)]
pub enum ColumnError {
    TypeMismatch { expected: DataType, got: String },
    ParseError(String),
}

#[derive(Debug)]
pub enum ColumnValue {
    Int32(Option<i32>),
    Int64(Option<i64>),
    Float32(Option<f32>),
    Float64(Option<f64>),
    String(Option<String>),
    DateTime(Option<i64>),
    DateTimeStr(Option<String>),
}

/// Raw column values wrapper for unified access
#[derive(Debug)]
pub enum ColumnValues<'a> {
    Int32(&'a Vec<i32>),
    Int64(&'a Vec<i64>),
    Float32(&'a Vec<f32>),
    Float64(&'a Vec<f64>),
    String(&'a Vec<String>),
    DateTime(&'a Vec<i64>),
}

impl<'a> ColumnValues<'a> {
    pub fn len(&self) -> usize {
        match self {
            ColumnValues::Int32(vec) => vec.len(),
            ColumnValues::Int64(vec) => vec.len(),
            ColumnValues::Float32(vec) => vec.len(),
            ColumnValues::Float64(vec) => vec.len(),
            ColumnValues::String(vec) => vec.len(),
            ColumnValues::DateTime(vec) => vec.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn data_type(&self) -> DataType {
        match self {
            ColumnValues::Int32(_) => DataType::Int32,
            ColumnValues::Int64(_) => DataType::Int64,
            ColumnValues::Float32(_) => DataType::Float32,
            ColumnValues::Float64(_) => DataType::Float64,
            ColumnValues::String(_) => DataType::String,
            ColumnValues::DateTime(_) => DataType::DateTime,
        }
    }
}

impl fmt::Display for ColumnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnError::TypeMismatch { expected, got } => {
                write!(f, "Type mismatch: expected {expected:?}, got {got}")
            }
            ColumnError::ParseError(msg) => write!(f, "Parse error: {msg}"),
        }
    }
}

impl std::error::Error for ColumnError {}

// Type Conversions for ColumnValue
impl From<Option<i32>> for ColumnValue {
    fn from(value: Option<i32>) -> Self {
        ColumnValue::Int32(value)
    }
}

impl From<i32> for ColumnValue {
    fn from(value: i32) -> Self {
        ColumnValue::Int32(Some(value))
    }
}

impl From<Option<i64>> for ColumnValue {
    fn from(value: Option<i64>) -> Self {
        ColumnValue::Int64(value)
    }
}

impl From<i64> for ColumnValue {
    fn from(value: i64) -> Self {
        ColumnValue::Int64(Some(value))
    }
}

impl From<Option<f32>> for ColumnValue {
    fn from(value: Option<f32>) -> Self {
        ColumnValue::Float32(value)
    }
}

impl From<f32> for ColumnValue {
    fn from(value: f32) -> Self {
        ColumnValue::Float32(Some(value))
    }
}

impl From<Option<f64>> for ColumnValue {
    fn from(value: Option<f64>) -> Self {
        ColumnValue::Float64(value)
    }
}

impl From<f64> for ColumnValue {
    fn from(value: f64) -> Self {
        ColumnValue::Float64(Some(value))
    }
}

impl From<Option<String>> for ColumnValue {
    fn from(value: Option<String>) -> Self {
        ColumnValue::String(value)
    }
}

impl From<String> for ColumnValue {
    fn from(value: String) -> Self {
        ColumnValue::String(Some(value))
    }
}

impl From<&str> for ColumnValue {
    fn from(value: &str) -> Self {
        ColumnValue::String(Some(value.to_string()))
    }
}

impl From<Option<&str>> for ColumnValue {
    fn from(value: Option<&str>) -> Self {
        ColumnValue::String(value.map(|s| s.to_string()))
    }
}

// For datetime columns, strings should be interpreted as datetime strings
// This is context-dependent and handled in the push method

// Additional convenience implementations
impl From<&String> for ColumnValue {
    fn from(value: &String) -> Self {
        ColumnValue::String(Some(value.clone()))
    }
}

impl From<Option<&String>> for ColumnValue {
    fn from(value: Option<&String>) -> Self {
        ColumnValue::String(value.cloned())
    }
}

// For datetime strings specifically (when you want to parse them as datetime)
impl ColumnValue {
    pub fn datetime_str<S: Into<String>>(value: Option<S>) -> Self {
        ColumnValue::DateTimeStr(value.map(|s| s.into()))
    }
}

impl Column {
    pub fn new(name: &str, dtype: DataType) -> Self {
        let data = match dtype {
            DataType::Int32 => ColumnData::Int32(Vec::new()),
            DataType::Int64 => ColumnData::Int64(Vec::new()),
            DataType::Float32 => ColumnData::Float32(Vec::new()),
            DataType::Float64 => ColumnData::Float64(Vec::new()),
            DataType::String => ColumnData::String(Vec::new()),
            DataType::DateTime => ColumnData::DateTime(Vec::new()),
        };

        Column {
            name: name.to_string(),
            dtype,
            data,
            null_mask: BitVec::new(),
            datetime_format: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        match &self.data {
            ColumnData::Int32(v) => v.len(),
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float32(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::String(v) => v.len(),
            ColumnData::DateTime(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn push_i32(&mut self, value: Option<i32>) -> Result<(), ColumnError> {
        match &mut self.data {
            ColumnData::Int32(vec) => {
                let is_null = value.is_none();
                vec.push(value.unwrap_or(i32::MIN));
                self.null_mask.push(is_null);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch {
                expected: DataType::Int32,
                got: "i32".to_string(),
            }),
        }
    }

    fn push_i64(&mut self, value: Option<i64>) -> Result<(), ColumnError> {
        match &mut self.data {
            ColumnData::Int64(vec) => {
                let is_null = value.is_none();
                vec.push(value.unwrap_or(i64::MIN));
                self.null_mask.push(is_null);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch {
                expected: DataType::Int64,
                got: "i64".to_string(),
            }),
        }
    }

    fn push_f32(&mut self, value: Option<f32>) -> Result<(), ColumnError> {
        match &mut self.data {
            ColumnData::Float32(vec) => {
                let is_null = value.is_none();
                vec.push(value.unwrap_or(f32::NAN));
                self.null_mask.push(is_null);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch {
                expected: DataType::Float32,
                got: "f32".to_string(),
            }),
        }
    }

    fn push_f64(&mut self, value: Option<f64>) -> Result<(), ColumnError> {
        match &mut self.data {
            ColumnData::Float64(vec) => {
                let is_null = value.is_none();
                vec.push(value.unwrap_or(f64::NAN));
                self.null_mask.push(is_null);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch {
                expected: DataType::Float64,
                got: "f64".to_string(),
            }),
        }
    }

    fn push_string(&mut self, value: Option<String>) -> Result<(), ColumnError> {
        match &mut self.data {
            ColumnData::String(vec) => {
                let is_null = value.is_none();
                vec.push(value.unwrap_or_default());
                self.null_mask.push(is_null);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch {
                expected: DataType::String,
                got: "String".to_string(),
            }),
        }
    }

    fn push_datetime(&mut self, value: Option<i64>) -> Result<(), ColumnError> {
        match &mut self.data {
            ColumnData::DateTime(vec) => {
                let is_null = value.is_none();
                vec.push(value.unwrap_or(i64::MIN));
                self.null_mask.push(is_null);
                Ok(())
            }
            _ => Err(ColumnError::TypeMismatch {
                expected: DataType::DateTime,
                got: "DateTime".to_string(),
            }),
        }
    }

    fn push_datetime_str(&mut self, value: Option<&str>) -> Result<(), ColumnError> {
        match value {
            None => self.push_datetime(None),
            Some(s) => {
                let millis = parse_datetime(s)?;
                self.push_datetime(Some(millis))
            }
        }
    }

    fn push_datetime_str_fmt(
        &mut self,
        value: Option<&str>,
        format: &str,
    ) -> Result<(), ColumnError> {
        match value {
            None => self.push_datetime(None),
            Some(s) => {
                let millis = parse_datetime_fmt(s, format)?;
                self.push_datetime(Some(millis))
            }
        }
    }

    pub fn with_datetime_fmt(mut self, format: &str) -> Self {
        self.datetime_format = Some(format.to_string());
        self
    }

    /// Unified push method - works with any compatible type based on column data type
    pub fn push<T>(&mut self, value: T) -> Result<(), ColumnError>
    where
        T: Into<ColumnValue>,
    {
        let column_value = value.into();
        match (self.dtype, column_value) {
            (DataType::Int32, ColumnValue::Int32(v)) => self.push_i32(v),
            (DataType::Int64, ColumnValue::Int64(v)) => self.push_i64(v),
            (DataType::Float32, ColumnValue::Float32(v)) => self.push_f32(v),
            (DataType::Float64, ColumnValue::Float64(v)) => self.push_f64(v),
            (DataType::String, ColumnValue::String(v)) => self.push_string(v),
            (DataType::DateTime, ColumnValue::DateTime(v)) => self.push_datetime(v),
            (DataType::DateTime, ColumnValue::DateTimeStr(v)) => {
                let format = self.datetime_format.clone();
                match format {
                    Some(fmt) => self.push_datetime_str_fmt(v.as_deref(), &fmt),
                    None => self.push_datetime_str(v.as_deref()),
                }
            }
            // Handle strings as datetime strings when the column type is DateTime
            (DataType::DateTime, ColumnValue::String(v)) => {
                let format = self.datetime_format.clone();
                match format {
                    Some(fmt) => self.push_datetime_str_fmt(v.as_deref(), &fmt),
                    None => self.push_datetime_str(v.as_deref()),
                }
            }
            _ => Err(ColumnError::TypeMismatch {
                expected: self.dtype,
                got: "incompatible type".to_string(),
            }),
        }
    }

    /// Add multiple values from a vector to the column
    pub fn from<T>(mut self, values: Vec<T>) -> Result<Self, ColumnError>
    where
        T: Into<ColumnValue>,
    {
        for value in values {
            self.push(value)?;
        }
        Ok(self)
    }

    /// Display first 5 rows
    pub fn head(&self) {
        self.head_n(5)
    }

    /// Display first n rows
    pub fn head_n(&self, n: usize) {
        println!("Column: {} ({:?})", self.name, self.dtype);
        let limit = self.len().min(n);

        for i in 0..limit {
            let is_null = self.null_mask.get(i).is_some_and(|bit| *bit);

            if is_null {
                println!("  [{i}]: NULL");
            } else {
                match &self.data {
                    ColumnData::Int32(vec) => println!("  [{}]: {}", i, vec[i]),
                    ColumnData::Int64(vec) => println!("  [{}]: {}", i, vec[i]),
                    ColumnData::Float32(vec) => println!("  [{}]: {:.4}", i, vec[i]),
                    ColumnData::Float64(vec) => println!("  [{}]: {:.4}", i, vec[i]),
                    ColumnData::String(vec) => println!("  [{}]: \"{}\"", i, vec[i]),
                    ColumnData::DateTime(vec) => {
                        if let Some(dt) = DateTime::from_timestamp_millis(vec[i]) {
                            println!(
                                "  [{}]: {} ({})",
                                i,
                                dt.format("%Y-%m-%d %H:%M:%S UTC"),
                                vec[i]
                            );
                        } else {
                            println!("  [{}]: Invalid timestamp ({})", i, vec[i]);
                        }
                    }
                }
            }
        }

        if self.len() > n {
            println!("  ... ({} more rows)", self.len() - n);
        }
    }

    pub fn i32_values(&self) -> Option<&[i32]> {
        match &self.data {
            ColumnData::Int32(vec) => Some(vec),
            _ => None,
        }
    }

    pub fn i64_values(&self) -> Option<&[i64]> {
        match &self.data {
            ColumnData::Int64(vec) => Some(vec),
            _ => None,
        }
    }

    pub fn f32_values(&self) -> Option<&[f32]> {
        match &self.data {
            ColumnData::Float32(vec) => Some(vec),
            _ => None,
        }
    }

    pub fn f64_values(&self) -> Option<&[f64]> {
        match &self.data {
            ColumnData::Float64(vec) => Some(vec),
            _ => None,
        }
    }

    pub fn string_values(&self) -> Option<&[String]> {
        match &self.data {
            ColumnData::String(vec) => Some(vec),
            _ => None,
        }
    }

    pub fn datetime_values(&self) -> Option<&[i64]> {
        match &self.data {
            ColumnData::DateTime(vec) => Some(vec),
            _ => None,
        }
    }

    pub fn null_mask(&self) -> &BitVec {
        &self.null_mask
    }

    /// Get raw stored values as a unified wrapper
    pub fn values(&self) -> ColumnValues<'_> {
        match &self.data {
            ColumnData::Int32(vec) => ColumnValues::Int32(vec),
            ColumnData::Int64(vec) => ColumnValues::Int64(vec),
            ColumnData::Float32(vec) => ColumnValues::Float32(vec),
            ColumnData::Float64(vec) => ColumnValues::Float64(vec),
            ColumnData::String(vec) => ColumnValues::String(vec),
            ColumnData::DateTime(vec) => ColumnValues::DateTime(vec),
        }
    }
}

// DateTime parsing helpers
fn parse_datetime(s: &str) -> Result<i64, ColumnError> {
    // Try ISO 8601 / RFC 3339
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(dt.with_timezone(&Utc).timestamp_millis());
    }

    // Try YYYY-MM-DD
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d")
        && let Some(dt) = date.and_hms_opt(0, 0, 0)
    {
        return Ok(dt.and_utc().timestamp_millis());
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
    if let Ok(date) = NaiveDate::parse_from_str(s, format)
        && let Some(dt) = date.and_hms_opt(0, 0, 0)
    {
        return Ok(dt.and_utc().timestamp_millis());
    }

    Err(ColumnError::ParseError(format!(
        "Cannot parse '{s}' with format '{format}'"
    )))
}
