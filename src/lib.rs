//! Otters is a minimal library for SIMD-accelerated exact search with expressive
//! metadata filtering. It pairs columnar storage with a lightweight query planner
//! and runs tight SIMD kernels for scoring, aiming to be "Polars for vector search."
pub mod col;
pub mod expr;
pub mod meta;
pub mod prelude;
pub mod type_utils;
pub mod vec;

mod display;
mod meta_compute;
mod vec_compute;

pub use crate::meta_compute::{BloomBuild, ZoneStat, build_zone_stat_for_range};
