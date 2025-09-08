use std::collections::HashMap;

use bitvec::bitvec;
use bitvec::prelude::BitVec;
use fastbloom::BloomFilter;

use crate::col::Column;
use crate::expr::{CmpOp, ColumnFilter, CompiledFilter, NumericLiteral};
use crate::type_utils::{
    DataType, apply_rows_mask_f32, apply_rows_mask_f64, apply_rows_mask_i32, apply_rows_mask_i64,
};
use crate::vec::{Cmp as VecCmp, Metric, VecStore};

#[derive(Debug, Clone)]
pub enum ZoneStat {
    Int { min: i64, max: i64, non_null: usize },
    Float { min: f64, max: f64, non_null: usize },
    DateTime { min: i64, max: i64, non_null: usize },
    String { bloom: BloomFilter, non_null: usize },
}

#[derive(Debug, Clone, Copy)]
pub enum BloomBuild {
    Fpr(f64),
    Bits(usize),
}

pub fn build_zone_stat_for_range(
    col: &Column,
    dtype: DataType,
    start: usize,
    end: usize,
    bloom_cfg: BloomBuild,
) -> Result<ZoneStat, String> {
    let mask = col.null_mask();
    match dtype {
        DataType::Int32 => {
            let vals = col.i32_values().ok_or("expected Int32 column")?;
            let (min_v, max_v, non_null) = (start..end)
                .filter(|&i| !mask.get(i).is_some_and(|b| *b))
                .map(|i| vals[i] as i64)
                .fold((i64::MAX, i64::MIN, 0usize), |(mn, mx, cnt), v| {
                    (mn.min(v), mx.max(v), cnt + 1)
                });
            Ok(ZoneStat::Int {
                min: min_v,
                max: max_v,
                non_null,
            })
        }
        DataType::Int64 => {
            let vals = col.i64_values().ok_or("expected Int64 column")?;
            let (min_v, max_v, non_null) = (start..end)
                .filter(|&i| !mask.get(i).is_some_and(|b| *b))
                .map(|i| vals[i])
                .fold((i64::MAX, i64::MIN, 0usize), |(mn, mx, cnt), v| {
                    (mn.min(v), mx.max(v), cnt + 1)
                });
            Ok(ZoneStat::Int {
                min: min_v,
                max: max_v,
                non_null,
            })
        }
        DataType::Float32 => {
            let vals = col.f32_values().ok_or("expected Float32 column")?;
            let (min_v, max_v, non_null) = (start..end)
                .filter(|&i| !mask.get(i).is_some_and(|b| *b))
                .map(|i| vals[i] as f64)
                .fold(
                    (f64::INFINITY, f64::NEG_INFINITY, 0usize),
                    |(mn, mx, cnt), v| (mn.min(v), mx.max(v), cnt + 1),
                );
            Ok(ZoneStat::Float {
                min: min_v,
                max: max_v,
                non_null,
            })
        }
        DataType::Float64 => {
            let vals = col.f64_values().ok_or("expected Float64 column")?;
            let (min_v, max_v, non_null) = (start..end)
                .filter(|&i| !mask.get(i).is_some_and(|b| *b))
                .map(|i| vals[i])
                .fold(
                    (f64::INFINITY, f64::NEG_INFINITY, 0usize),
                    |(mn, mx, cnt), v| (mn.min(v), mx.max(v), cnt + 1),
                );
            Ok(ZoneStat::Float {
                min: min_v,
                max: max_v,
                non_null,
            })
        }
        DataType::String => {
            // Build a Bloom filter sized for this chunk using selected config
            let mut bloom = match bloom_cfg {
                BloomBuild::Fpr(p) => BloomFilter::with_false_pos(p).expected_items(end - start),
                BloomBuild::Bits(bits) => {
                    BloomFilter::with_num_bits(bits).expected_items(end - start)
                }
            };
            let vals = col.string_values().ok_or("expected String column")?;
            let non_null: usize = (start..end)
                .filter(|&i| !mask.get(i).is_some_and(|b| *b))
                .map(|i| {
                    bloom.insert(vals[i].as_bytes());
                    1usize
                })
                .sum();
            Ok(ZoneStat::String { bloom, non_null })
        }
        DataType::DateTime => {
            let vals = col.datetime_values().ok_or("expected DateTime column")?;
            let (min_v, max_v, non_null) = (start..end)
                .filter(|&i| !mask.get(i).is_some_and(|b| *b))
                .map(|i| vals[i])
                .fold((i64::MAX, i64::MIN, 0usize), |(mn, mx, cnt), v| {
                    (mn.min(v), mx.max(v), cnt + 1)
                });
            Ok(ZoneStat::DateTime {
                min: min_v,
                max: max_v,
                non_null,
            })
        }
    }
}

#[derive(Debug)]
pub struct MetaChunk {
    pub base_offset: usize,
    pub len: usize,
    pub vec_store: VecStore,
    pub stats: HashMap<String, ZoneStat>,
}

#[derive(Default)]
pub struct ChunkAgg {
    pub results: Vec<(usize, f32)>,
    pub before: usize,
    pub after: usize,
    pub compared: usize,
}

// Process a single chunk: run per-chunk VecStore query, apply row-level meta filter,
// and collect results and counters
#[allow(clippy::too_many_arguments)]
pub fn process_chunk(
    chunk: &MetaChunk,
    metric: &Metric,
    queries: &[Vec<f32>],
    vec_filter: Option<(f32, VecCmp)>,
    meta_filter: Option<&CompiledFilter>,
    columns: &HashMap<String, Column>,
    k: usize,
) -> ChunkAgg {
    let mut agg = ChunkAgg {
        results: Vec::new(),
        before: 0,
        after: 0,
        compared: chunk.len * queries.len(),
    };

    let row_mask_opt: Option<BitVec> =
        meta_filter.map(|cf| build_row_mask_for_chunk(cf, columns, chunk.base_offset, chunk.len));

    let metric2 = *metric;
    let mut plan = chunk.vec_store.query(queries.to_owned(), metric2);
    if let Some((thr, cmp)) = &vec_filter {
        plan = plan.filter(*thr, cmp.clone());
    }
    if let Some(rm) = row_mask_opt.clone() {
        plan = plan.with_row_mask(rm);
    }
    plan = plan.take(k);

    if let Ok(vecs) = plan.collect() {
        agg.before += vecs.len();
        for sr in vecs.into_iter() {
            let global_idx = chunk.base_offset + sr.index;
            agg.results.push((global_idx, sr.score));
            agg.after += 1;
        }
    }

    agg
}

pub fn build_row_mask_for_chunk(
    compiled: &CompiledFilter,
    columns: &HashMap<String, Column>,
    base: usize,
    len: usize,
) -> BitVec {
    compiled
        .clauses
        .iter()
        .fold(bitvec![1; len], |mut acc, clause| {
            let mut clause_mask = bitvec![0; len];
            clause.iter().for_each(|leaf| match leaf {
                ColumnFilter::Numeric { column, cmp, rhs } => {
                    apply_numeric_leaf_row_mask(
                        columns,
                        base,
                        len,
                        column,
                        *cmp,
                        rhs,
                        &mut clause_mask,
                    );
                }
                ColumnFilter::String { column, cmp, rhs } => {
                    apply_string_leaf_row_mask(
                        columns,
                        base,
                        len,
                        column,
                        *cmp,
                        rhs,
                        &mut clause_mask,
                    );
                }
            });
            acc &= clause_mask;
            acc
        })
}

// Leaf helpers for row mask
fn apply_numeric_leaf_row_mask(
    columns: &HashMap<String, Column>,
    base: usize,
    len: usize,
    column: &str,
    cmp: CmpOp,
    rhs: &NumericLiteral,
    clause_mask: &mut BitVec,
) {
    if let Some(col) = columns.get(column) {
        match col.dtype() {
            DataType::Float32 => {
                let vals = col.f32_values().unwrap();
                let nulls = col.null_mask();
                let thr = match rhs {
                    NumericLiteral::F64(v) => *v as f32,
                    NumericLiteral::I64(v) => *v as f32,
                };
                apply_rows_mask_f32(vals, nulls, base, len, cmp, thr, clause_mask);
            }
            DataType::Int32 => {
                let vals = col.i32_values().unwrap();
                let nulls = col.null_mask();
                let thr = match rhs {
                    NumericLiteral::I64(v) => *v as i32,
                    NumericLiteral::F64(v) => *v as i32,
                };
                apply_rows_mask_i32(vals, nulls, base, len, cmp, thr, clause_mask);
            }
            DataType::Float64 => {
                let vals = col.f64_values().unwrap();
                let nulls = col.null_mask();
                let thr = match rhs {
                    NumericLiteral::F64(v) => *v,
                    NumericLiteral::I64(v) => *v as f64,
                };
                apply_rows_mask_f64(vals, nulls, base, len, cmp, thr, clause_mask);
            }
            DataType::Int64 | DataType::DateTime => {
                let vals = if col.dtype() == DataType::Int64 {
                    col.i64_values().unwrap()
                } else {
                    col.datetime_values().unwrap()
                };
                let nulls = col.null_mask();
                let thr = match rhs {
                    NumericLiteral::I64(v) => *v,
                    NumericLiteral::F64(v) => *v as i64,
                };
                apply_rows_mask_i64(vals, nulls, base, len, cmp, thr, clause_mask);
            }
            DataType::String => {}
        }
    }
}

fn apply_string_leaf_row_mask(
    columns: &HashMap<String, Column>,
    base: usize,
    len: usize,
    column: &str,
    cmp: CmpOp,
    rhs: &str,
    clause_mask: &mut BitVec,
) {
    if let Some(col) = columns.get(column) {
        let vals = col.string_values().unwrap();
        let nulls = col.null_mask();
        vals[base..base + len]
            .iter()
            .enumerate()
            .filter(|(off, _)| !nulls.get(base + *off).map(|b| *b).unwrap_or(false))
            .for_each(|(off, v)| {
                let sat = match cmp {
                    CmpOp::Eq => v == rhs,
                    CmpOp::Neq => v != rhs,
                    _ => false,
                };
                if sat {
                    clause_mask.set(off, true);
                }
            });
    }
}
