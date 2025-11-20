//! Shared datetime parsing helpers used across the crate.
//!
//! Provides consistent handling of accepted timestamp/date formats and
//! uniform error reporting for callers that need either `Result` or `Option`.

use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use std::fmt;

/// Error returned when a datetime string cannot be parsed into milliseconds.
#[derive(Debug, Clone)]
pub struct ParseDateTimeError {
    input: String,
    reason: &'static str,
}

impl ParseDateTimeError {
    fn new(input: impl Into<String>, reason: &'static str) -> Self {
        Self {
            input: input.into(),
            reason,
        }
    }
}

impl fmt::Display for ParseDateTimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cannot parse '{}' as {}", self.input, self.reason)
    }
}

impl std::error::Error for ParseDateTimeError {}

/// Parse a datetime string using common default formats.
///
/// Supported formats mirror the previous behaviour:
/// - RFC3339 / ISO8601
/// - `YYYY-MM-DD`
/// - `YYYY-MM-DD HH:MM:SS`
pub fn parse_datetime_millis(input: &str) -> Result<i64, ParseDateTimeError> {
    parse_with_default_formats(input).ok_or_else(|| {
        ParseDateTimeError::new(
            input,
            "datetime. Supported formats: ISO 8601, YYYY-MM-DD, YYYY-MM-DD HH:MM:SS",
        )
    })
}

/// Parse a datetime string using an explicit format string.
pub fn parse_datetime_millis_with_format(
    input: &str,
    format: &str,
) -> Result<i64, ParseDateTimeError> {
    parse_with_specified_format(input, format)
        .ok_or_else(|| ParseDateTimeError::new(input, "datetime with the provided format string"))
}

/// Parse a datetime string with default formats, returning `None` on failure.
pub fn try_parse_datetime_millis(input: &str) -> Option<i64> {
    parse_with_default_formats(input)
}

fn parse_with_default_formats(input: &str) -> Option<i64> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(input) {
        return Some(dt.with_timezone(&Utc).timestamp_millis());
    }

    if let Ok(date) = NaiveDate::parse_from_str(input, "%Y-%m-%d")
        && let Some(dt) = date.and_hms_opt(0, 0, 0)
    {
        return Some(dt.and_utc().timestamp_millis());
    }

    if let Ok(dt) = NaiveDateTime::parse_from_str(input, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.and_utc().timestamp_millis());
    }

    None
}

fn parse_with_specified_format(input: &str, format: &str) -> Option<i64> {
    if let Ok(dt) = NaiveDateTime::parse_from_str(input, format) {
        return Some(dt.and_utc().timestamp_millis());
    }

    if let Ok(date) = NaiveDate::parse_from_str(input, format)
        && let Some(dt) = date.and_hms_opt(0, 0, 0)
    {
        return Some(dt.and_utc().timestamp_millis());
    }

    None
}
