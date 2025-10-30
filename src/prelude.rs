//! Convenient re-exports for common types and functions
//!
//! Import everything you typically need with:
//! `use otters::prelude::*;`.

// Main store type
pub use crate::store::OttersStore;
// Query planner
pub use crate::query::{OttersQuery, QueryMetric};

// Commonly used compute functions
pub use crate::vec_compute::{
    cosine_similarity, dot_product, euclidean_distance_squared, inverse_norm,
};

// Column functionality and expression DSL
pub use crate::col::*;
pub use crate::expr::*;
pub use crate::record::OttersRecord;
