pub mod col;
pub mod expr;
pub mod meta;
pub mod prelude;
pub mod type_utils;
pub mod vec;

// Internal modules (not part of the public API surface)
mod meta_compute;
mod vec_compute;
mod display;

// Selective public re-exports from internal compute module
pub use crate::meta_compute::{
    BloomBuild, ZoneStat, build_zone_stat_for_range,
};
