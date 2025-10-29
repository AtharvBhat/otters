//! Helper wrappers for presenting Arrow record batches.

use crate::col::OttersColumn;
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use std::fmt;
use std::ops::{Deref, DerefMut};

/// Wrapper around [`RecordBatch`] that implements [`Display`] for pretty printing.
#[derive(Clone)]
pub struct OttersRecord(RecordBatch);

const RECORD_DISPLAY_PREVIEW_ROWS: usize = 10;

impl OttersRecord {
    /// Create a new displayable record.
    pub fn new(batch: RecordBatch) -> Self {
        Self(batch)
    }

    /// Consume the wrapper and return the underlying batch.
    pub fn into_inner(self) -> RecordBatch {
        self.0
    }

    /// Fetch a column by name as an [`OttersColumn`], cloning the underlying array.
    pub fn col(&self, name: &str) -> Option<OttersColumn> {
        let schema = self.0.schema();
        let (index, field) = schema.column_with_name(name)?;
        let array = self.0.column(index).clone();
        Some(OttersColumn::from_field(field.clone(), array))
    }
}

impl fmt::Display for OttersRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_rows = self.0.num_rows();
        let preview_len = total_rows.min(RECORD_DISPLAY_PREVIEW_ROWS);
        let preview = if preview_len == total_rows {
            self.0.clone()
        } else {
            self.0.slice(0, preview_len)
        };

        match pretty_format_batches(&[preview]) {
            Ok(formatted) => {
                let formatted = formatted.to_string();
                if formatted.ends_with('\n') {
                    write!(f, "{formatted}")?;
                } else {
                    writeln!(f, "{formatted}")?;
                }
            }
            Err(err) => {
                return write!(f, "Failed to format RecordBatch: {err}");
            }
        }

        if total_rows > preview_len {
            writeln!(f, "... ({} more rows)", total_rows - preview_len)?;
        }

        write!(f, "Total rows: {total_rows}")
    }
}

impl fmt::Debug for OttersRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("OttersRecord").field(&self.0).finish()
    }
}

impl Deref for OttersRecord {
    type Target = RecordBatch;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for OttersRecord {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<RecordBatch> for OttersRecord {
    fn from(value: RecordBatch) -> Self {
        Self::new(value)
    }
}

impl From<OttersRecord> for RecordBatch {
    fn from(value: OttersRecord) -> Self {
        value.into_inner()
    }
}

impl AsRef<RecordBatch> for OttersRecord {
    fn as_ref(&self) -> &RecordBatch {
        &self.0
    }
}

impl AsMut<RecordBatch> for OttersRecord {
    fn as_mut(&mut self) -> &mut RecordBatch {
        &mut self.0
    }
}
