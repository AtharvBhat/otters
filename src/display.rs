//! Helper wrappers for presenting Arrow record batches.

use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use std::fmt;
use std::ops::{Deref, DerefMut};

/// Wrapper around [`RecordBatch`] that implements [`Display`] for pretty printing.
#[derive(Clone)]
pub struct DisplayBatch(RecordBatch);

impl DisplayBatch {
    /// Create a new displayable batch.
    pub fn new(batch: RecordBatch) -> Self {
        Self(batch)
    }

    /// Consume the wrapper and return the underlying batch.
    pub fn into_inner(self) -> RecordBatch {
        self.0
    }
}

impl fmt::Display for DisplayBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match pretty_format_batches(&[self.0.clone()]) {
            Ok(formatted) => write!(f, "{formatted}"),
            Err(err) => write!(f, "Failed to format RecordBatch: {err}"),
        }
    }
}

impl fmt::Debug for DisplayBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DisplayBatch").field(&self.0).finish()
    }
}

impl Deref for DisplayBatch {
    type Target = RecordBatch;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DisplayBatch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<RecordBatch> for DisplayBatch {
    fn from(value: RecordBatch) -> Self {
        Self::new(value)
    }
}

impl From<DisplayBatch> for RecordBatch {
    fn from(value: DisplayBatch) -> Self {
        value.into_inner()
    }
}

impl AsRef<RecordBatch> for DisplayBatch {
    fn as_ref(&self) -> &RecordBatch {
        &self.0
    }
}

impl AsMut<RecordBatch> for DisplayBatch {
    fn as_mut(&mut self) -> &mut RecordBatch {
        &mut self.0
    }
}
