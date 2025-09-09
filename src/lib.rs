#![doc = include_str!("../README.md")]

pub mod col;
pub mod expr;
pub mod meta;
pub mod prelude;
#[doc(hidden)]
pub mod type_utils;
pub mod vec;

mod display;
mod meta_compute;
mod vec_compute;
