use std::collections::HashMap;

use crate::col::Column;
use crate::expr::{CmpOp, ColumnFilter, CompiledFilter, NumericLiteral};
use crate::type_utils::DataType;
use fastbloom::BloomFilter;
// no wide/SIMD here; scalar loops are clearer and sufficient

#[derive(Debug, Clone)]
pub enum ZoneStat {
    Int { min: i64, max: i64, non_null: usize },
    Float { min: f64, max: f64, non_null: usize },
    DateTime { min: i64, max: i64, non_null: usize },
    String { bloom: BloomFilter, non_null: usize },
}

// SIMD helpers moved to crate::types

#[allow(clippy::needless_range_loop)]
pub fn build_zone_stat_for_range(
    col: &Column,
    dtype: DataType,
    start: usize,
    end: usize,
) -> Result<ZoneStat, String> {
    let mask = col.null_mask();
    match dtype {
        DataType::Int32 => {
            let vals = col.i32_values().ok_or("expected Int32 column")?;
            let mut min_v = i64::MAX;
            let mut max_v = i64::MIN;
            let mut non_null = 0usize;
            for i in start..end {
                if !mask.get(i).is_some_and(|b| *b) {
                    let v = vals[i] as i64;
                    if v < min_v {
                        min_v = v;
                    }
                    if v > max_v {
                        max_v = v;
                    }
                    non_null += 1;
                }
            }
            Ok(ZoneStat::Int {
                min: min_v,
                max: max_v,
                non_null,
            })
        }
        DataType::Int64 => {
            let vals = col.i64_values().ok_or("expected Int64 column")?;
            let mut min_v = i64::MAX;
            let mut max_v = i64::MIN;
            let mut non_null = 0usize;
            for i in start..end {
                if !mask.get(i).is_some_and(|b| *b) {
                    let v = vals[i];
                    if v < min_v {
                        min_v = v;
                    }
                    if v > max_v {
                        max_v = v;
                    }
                    non_null += 1;
                }
            }
            Ok(ZoneStat::Int {
                min: min_v,
                max: max_v,
                non_null,
            })
        }
        DataType::Float32 => {
            let vals = col.f32_values().ok_or("expected Float32 column")?;
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            let mut non_null = 0usize;
            for i in start..end {
                if !mask.get(i).is_some_and(|b| *b) {
                    let v = vals[i] as f64;
                    if v < min_v {
                        min_v = v;
                    }
                    if v > max_v {
                        max_v = v;
                    }
                    non_null += 1;
                }
            }
            Ok(ZoneStat::Float {
                min: min_v,
                max: max_v,
                non_null,
            })
        }
        DataType::Float64 => {
            let vals = col.f64_values().ok_or("expected Float64 column")?;
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            let mut non_null = 0usize;
            for i in start..end {
                if !mask.get(i).is_some_and(|b| *b) {
                    let v = vals[i];
                    if v < min_v {
                        min_v = v;
                    }
                    if v > max_v {
                        max_v = v;
                    }
                    non_null += 1;
                }
            }
            Ok(ZoneStat::Float {
                min: min_v,
                max: max_v,
                non_null,
            })
        }
        DataType::String => {
            // Build a Bloom filter sized for this chunk with ~1% false positive rate
            let mut bloom = BloomFilter::with_false_pos(0.01).expected_items(end - start);
            let vals = col.string_values().ok_or("expected String column")?;
            let mut non_null = 0usize;
            for i in start..end {
                if !mask.get(i).is_some_and(|b| *b) {
                    bloom.insert(vals[i].as_bytes());
                    non_null += 1;
                }
            }
            Ok(ZoneStat::String { bloom, non_null })
        }
        DataType::DateTime => {
            let vals = col.datetime_values().ok_or("expected DateTime column")?;
            let mut min_v = i64::MAX;
            let mut max_v = i64::MIN;
            let mut non_null = 0usize;
            for i in start..end {
                if !mask.get(i).is_some_and(|b| *b) {
                    let v = vals[i];
                    if v < min_v {
                        min_v = v;
                    }
                    if v > max_v {
                        max_v = v;
                    }
                    non_null += 1;
                }
            }
            Ok(ZoneStat::DateTime {
                min: min_v,
                max: max_v,
                non_null,
            })
        }
    }
}

fn cmp_i64(v: i64, op: CmpOp, rhs: &NumericLiteral) -> bool {
    let r = match rhs {
        NumericLiteral::I64(x) => *x,
        NumericLiteral::F64(_) => return false,
    };
    match op {
        CmpOp::Eq => v == r,
        CmpOp::Neq => v != r,
        CmpOp::Lt => v < r,
        CmpOp::Lte => v <= r,
        CmpOp::Gt => v > r,
        CmpOp::Gte => v >= r,
    }
}

fn cmp_f64(v: f64, op: CmpOp, rhs: &NumericLiteral) -> bool {
    let r = match rhs {
        NumericLiteral::I64(x) => *x as f64,
        NumericLiteral::F64(x) => *x,
    };
    match op {
        CmpOp::Eq => v == r,
        CmpOp::Neq => v != r,
        CmpOp::Lt => v < r,
        CmpOp::Lte => v <= r,
        CmpOp::Gt => v > r,
        CmpOp::Gte => v >= r,
    }
}

pub fn eval_leaf_row(leaf: &ColumnFilter, cols: &HashMap<String, Column>, idx: usize) -> bool {
    match leaf {
        ColumnFilter::Numeric { column, cmp, rhs } => {
            let col = match cols.get(column) {
                Some(c) => c,
                None => return false,
            };
            let mask = col.null_mask();
            if mask.get(idx).is_some_and(|b| *b) {
                return false;
            }
            match col.dtype() {
                DataType::Int32 => {
                    let v = col.i32_values().unwrap()[idx] as i64;
                    cmp_i64(v, *cmp, rhs)
                }
                DataType::Int64 => {
                    let v = col.i64_values().unwrap()[idx];
                    cmp_i64(v, *cmp, rhs)
                }
                DataType::Float32 => {
                    let v = col.f32_values().unwrap()[idx] as f64;
                    cmp_f64(v, *cmp, rhs)
                }
                DataType::Float64 => {
                    let v = col.f64_values().unwrap()[idx];
                    cmp_f64(v, *cmp, rhs)
                }
                DataType::DateTime => {
                    let v = col.datetime_values().unwrap()[idx];
                    cmp_i64(v, *cmp, rhs)
                }
                DataType::String => false,
            }
        }
        ColumnFilter::String { column, cmp, rhs } => {
            let col = match cols.get(column) {
                Some(c) => c,
                None => return false,
            };
            let mask = col.null_mask();
            if mask.get(idx).is_some_and(|b| *b) {
                return false;
            }
            let v = &col.string_values().unwrap()[idx];
            match cmp {
                CmpOp::Eq => v == rhs,
                CmpOp::Neq => v != rhs,
                _ => false,
            }
        }
    }
}

pub fn chunk_may_satisfy(clause: &[ColumnFilter], stats: &HashMap<String, ZoneStat>) -> bool {
    clause.iter().any(|leaf| match leaf {
        ColumnFilter::Numeric { column, cmp, rhs } => match (stats.get(column), rhs) {
            (Some(ZoneStat::Int { min, max, non_null }), NumericLiteral::I64(rv))
            | (Some(ZoneStat::DateTime { min, max, non_null }), NumericLiteral::I64(rv)) => {
                if *non_null == 0 {
                    return false;
                }
                let (min, max, rv) = (*min, *max, *rv);
                match cmp {
                    CmpOp::Eq => (min <= rv) && (rv <= max),
                    CmpOp::Neq => true,
                    CmpOp::Lt => min < rv,
                    CmpOp::Lte => min <= rv,
                    CmpOp::Gt => max > rv,
                    CmpOp::Gte => max >= rv,
                }
            }
            (Some(ZoneStat::Float { min, max, non_null }), NumericLiteral::F64(rv)) => {
                if *non_null == 0 {
                    return false;
                }
                let (min, max, rv) = (*min, *max, *rv);
                match cmp {
                    CmpOp::Eq => (min <= rv) && (rv <= max),
                    CmpOp::Neq => true,
                    CmpOp::Lt => min < rv,
                    CmpOp::Lte => min <= rv,
                    CmpOp::Gt => max > rv,
                    CmpOp::Gte => max >= rv,
                }
            }
            (Some(ZoneStat::Float { min, max, non_null }), NumericLiteral::I64(rv)) => {
                if *non_null == 0 {
                    return false;
                }
                let rv = *rv as f64;
                match cmp {
                    CmpOp::Eq => (*min <= rv) && (rv <= *max),
                    CmpOp::Neq => true,
                    CmpOp::Lt => *min < rv,
                    CmpOp::Lte => *min <= rv,
                    CmpOp::Gt => *max > rv,
                    CmpOp::Gte => *max >= rv,
                }
            }
            _ => true,
        },
        ColumnFilter::String { column, cmp, rhs } => match stats.get(column) {
            Some(ZoneStat::String { bloom, non_null }) => {
                if *non_null == 0 {
                    return false;
                }
                match cmp {
                    CmpOp::Eq => bloom.contains(rhs.as_bytes()),
                    CmpOp::Neq => true,
                    _ => true,
                }
            }
            _ => true,
        },
    })
}

pub fn chunk_passes_plan(plan: &CompiledFilter, stats: &HashMap<String, ZoneStat>) -> bool {
    for clause in &plan.clauses {
        if !chunk_may_satisfy(clause, stats) {
            return false;
        }
    }
    true
}
