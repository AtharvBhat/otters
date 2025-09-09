//! Convenient re-exports for common types and functions
//!
//! Import everything you typically need with:
//! `use otters::prelude::*;`.

// Main data structures and query types
pub use crate::vec::{QueryBatch, VecQueryPlan, VecStore};

// Enums for configuration
pub use crate::vec::{Cmp, Metric, TakeType};

// Commonly used compute functions
pub use crate::vec::{cosine_similarity, dot_product, euclidean_distance_squared};

// Column functionality and expression DSL
pub use crate::col::*;
pub use crate::expr::*;

// Metadata store and result types
pub use crate::meta::{MetaBuildStats, MetaQueryResults, MetaQueryStats, MetaStore};

// Public data types
pub use crate::type_utils::DataType;
