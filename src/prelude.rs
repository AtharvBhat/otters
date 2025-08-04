//! Prelude module for the otters crate.
//!
//! This module re-exports the most commonly used items from the otters crate,
//! allowing users to quickly import everything they need with:
//!
//! ```rust
//! use otters::prelude::*;
//! ```

// Main data structures and query types
pub use crate::vec::{QueryBatch, VecQueryPlan, VecStore};

// Enums for configuration
pub use crate::vec::{Cmp, Metric, TakeScope, TakeType};

// Commonly used compute functions
pub use crate::vec::{cosine_similarity, dot_product, euclidean_distance_squared};

// Column functionality
pub use crate::col;
