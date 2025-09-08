//! Low-level SIMD wrapper types and masking helpers.
//!
//! Supplies ad‑hoc 8‑wide vectors (built from two 4‑wide lanes) plus masking
//! utilities used by metadata pruning and row-level filtering. Where direct
//! `wide` crate support exists we delegate; otherwise we compose manually.
// Wrapper types that provide 8-wide SIMD operations using two 4-wide operations
// Not all traits are implemented for all types in wide. Falls back to scalar logic when needed.
#![allow(non_camel_case_types)]

use bitvec::prelude::BitVec;
use wide::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    Int32,
    Int64,
    Float32,
    Float64,
    String,
    DateTime,
}

#[derive(Debug, Clone, Copy)]
pub struct i64x8 {
    low: i64x4,
    high: i64x4,
}

impl i64x8 {
    #[inline]
    pub fn splat(value: i64) -> Self {
        Self {
            low: i64x4::splat(value),
            high: i64x4::splat(value),
        }
    }

    #[inline]
    pub fn from_slice(slice: &[i64]) -> Self {
        debug_assert_eq!(slice.len(), 8);
        Self {
            low: i64x4::from(&slice[0..4]),
            high: i64x4::from(&slice[4..8]),
        }
    }

    #[inline]
    pub fn cmp_eq(self, other: Self) -> u8 {
        let low_mask = self.low.cmp_eq(other.low).move_mask();
        let high_mask = self.high.cmp_eq(other.high).move_mask();
        (low_mask as u8) | ((high_mask as u8) << 4)
    }

    #[inline]
    pub fn cmp_gt(self, other: Self) -> u8 {
        let low_mask = self.low.cmp_gt(other.low).move_mask();
        let high_mask = self.high.cmp_gt(other.high).move_mask();
        (low_mask as u8) | ((high_mask as u8) << 4)
    }

    #[inline]
    pub fn cmp_gte(self, other: Self) -> u8 {
        !self.cmp_lt(other)
    }

    #[inline]
    pub fn cmp_lt(self, other: Self) -> u8 {
        let low_mask = self.low.cmp_lt(other.low).move_mask();
        let high_mask = self.high.cmp_lt(other.high).move_mask();
        (low_mask as u8) | ((high_mask as u8) << 4)
    }

    #[inline]
    pub fn cmp_lte(self, other: Self) -> u8 {
        !self.cmp_gt(other)
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        let low_mask = self.low.cmp_gt(other.low);
        let high_mask = self.high.cmp_gt(other.high);
        Self {
            low: low_mask.blend(other.low, self.low),
            high: high_mask.blend(other.high, self.high),
        }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        let low_mask = self.low.cmp_lt(other.low);
        let high_mask = self.high.cmp_lt(other.high);
        Self {
            low: low_mask.blend(other.low, self.low),
            high: high_mask.blend(other.high, self.high),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct f64x8 {
    low: f64x4,
    high: f64x4,
}

impl f64x8 {
    #[inline]
    pub fn splat(value: f64) -> Self {
        Self {
            low: f64x4::splat(value),
            high: f64x4::splat(value),
        }
    }

    #[inline]
    pub fn from_slice(slice: &[f64]) -> Self {
        debug_assert_eq!(slice.len(), 8);
        Self {
            low: f64x4::from(&slice[0..4]),
            high: f64x4::from(&slice[4..8]),
        }
    }

    #[inline]
    pub fn cmp_eq(self, other: Self) -> u8 {
        let low_mask = self.low.cmp_eq(other.low).move_mask();
        let high_mask = self.high.cmp_eq(other.high).move_mask();
        (low_mask as u8) | ((high_mask as u8) << 4)
    }

    #[inline]
    pub fn cmp_gt(self, other: Self) -> u8 {
        let low_mask = self.low.cmp_gt(other.low).move_mask();
        let high_mask = self.high.cmp_gt(other.high).move_mask();
        (low_mask as u8) | ((high_mask as u8) << 4)
    }

    #[inline]
    pub fn cmp_ge(self, other: Self) -> u8 {
        let low_mask = self.low.cmp_ge(other.low).move_mask();
        let high_mask = self.high.cmp_ge(other.high).move_mask();
        (low_mask as u8) | ((high_mask as u8) << 4)
    }

    #[inline]
    pub fn cmp_lt(self, other: Self) -> u8 {
        let low_mask = self.low.cmp_lt(other.low).move_mask();
        let high_mask = self.high.cmp_lt(other.high).move_mask();
        (low_mask as u8) | ((high_mask as u8) << 4)
    }

    #[inline]
    pub fn cmp_le(self, other: Self) -> u8 {
        let low_mask = self.low.cmp_le(other.low).move_mask();
        let high_mask = self.high.cmp_le(other.high).move_mask();
        (low_mask as u8) | ((high_mask as u8) << 4)
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self {
            low: self.low.min(other.low),
            high: self.high.min(other.high),
        }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self {
            low: self.low.max(other.low),
            high: self.high.max(other.high),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct u64x8 {
    low: u64x4,
    high: u64x4,
}

impl u64x8 {
    #[inline]
    pub fn splat(value: u64) -> Self {
        Self {
            low: u64x4::splat(value),
            high: u64x4::splat(value),
        }
    }

    #[inline]
    pub fn from_slice(slice: &[u64]) -> Self {
        debug_assert_eq!(slice.len(), 8);
        Self {
            low: u64x4::from(&slice[0..4]),
            high: u64x4::from(&slice[4..8]),
        }
    }

    #[inline]
    pub fn cmp_eq(self, other: Self) -> u8 {
        // Compare as i64 by transmute with explicit types
        use std::mem::transmute;
        let self_i64 = i64x8 {
            low: unsafe { transmute::<u64x4, i64x4>(self.low) },
            high: unsafe { transmute::<u64x4, i64x4>(self.high) },
        };
        let other_i64 = i64x8 {
            low: unsafe { transmute::<u64x4, i64x4>(other.low) },
            high: unsafe { transmute::<u64x4, i64x4>(other.high) },
        };
        self_i64.cmp_eq(other_i64)
    }

    #[inline]
    pub fn cmp_gt(self, other: Self) -> u8 {
        let mut result = 0u8;
        let self_low: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(self.low) };
        let other_low: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(other.low) };
        let self_high: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(self.high) };
        let other_high: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(other.high) };
        for i in 0..4 {
            if self_low[i] > other_low[i] {
                result |= 1 << i;
            }
        }
        for i in 0..4 {
            if self_high[i] > other_high[i] {
                result |= 1 << (i + 4);
            }
        }
        result
    }

    #[inline]
    pub fn cmp_gte(self, other: Self) -> u8 {
        !self.cmp_lt(other)
    }

    #[inline]
    pub fn cmp_lt(self, other: Self) -> u8 {
        let mut result = 0u8;
        let self_low: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(self.low) };
        let other_low: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(other.low) };
        let self_high: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(self.high) };
        let other_high: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(other.high) };
        for i in 0..4 {
            if self_low[i] < other_low[i] {
                result |= 1 << i;
            }
        }
        for i in 0..4 {
            if self_high[i] < other_high[i] {
                result |= 1 << (i + 4);
            }
        }
        result
    }

    #[inline]
    pub fn cmp_lte(self, other: Self) -> u8 {
        !self.cmp_gt(other)
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        let self_low: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(self.low) };
        let other_low: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(other.low) };
        let self_high: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(self.high) };
        let other_high: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(other.high) };
        let mut result_low = [0u64; 4];
        let mut result_high = [0u64; 4];
        for i in 0..4 {
            result_low[i] = self_low[i].min(other_low[i]);
        }
        for i in 0..4 {
            result_high[i] = self_high[i].min(other_high[i]);
        }
        Self {
            low: unsafe { std::mem::transmute::<[u64; 4], u64x4>(result_low) },
            high: unsafe { std::mem::transmute::<[u64; 4], u64x4>(result_high) },
        }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        let self_low: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(self.low) };
        let other_low: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(other.low) };
        let self_high: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(self.high) };
        let other_high: [u64; 4] = unsafe { std::mem::transmute::<u64x4, [u64; 4]>(other.high) };
        let mut result_low = [0u64; 4];
        let mut result_high = [0u64; 4];
        for i in 0..4 {
            result_low[i] = self_low[i].max(other_low[i]);
        }
        for i in 0..4 {
            result_high[i] = self_high[i].max(other_high[i]);
        }
        Self {
            low: unsafe { std::mem::transmute::<[u64; 4], u64x4>(result_low) },
            high: unsafe { std::mem::transmute::<[u64; 4], u64x4>(result_high) },
        }
    }
}
// ==============================
// SIMD helper functions (8-wide)
// ==============================

pub fn mask8_rows_f32(
    vals: &[f32],
    nulls: &BitVec,
    base: usize,
    off: usize,
    cmp: crate::expr::CmpOp,
    thr: f32,
) -> u8 {
    let start = base + off;
    let v = f32x8::from(&vals[start..start + 8]);
    let t = f32x8::splat(thr);
    let m = match cmp {
        crate::expr::CmpOp::Eq => v.cmp_eq(t),
        crate::expr::CmpOp::Neq => v.cmp_ne(t),
        crate::expr::CmpOp::Lt => v.cmp_lt(t),
        crate::expr::CmpOp::Lte => v.cmp_le(t),
        crate::expr::CmpOp::Gt => v.cmp_gt(t),
        crate::expr::CmpOp::Gte => v.cmp_ge(t),
    };
    let arr = m.to_array();
    let mut bits: u8 = 0;
    for (j, &val) in arr.iter().enumerate() {
        let ok = val != 0.0;
        let not_null = !nulls.get(start + j).map(|b| *b).unwrap_or(false);
        if ok && not_null {
            bits |= 1 << j;
        }
    }
    bits
}

pub fn mask8_rows_i32(
    vals: &[i32],
    nulls: &BitVec,
    base: usize,
    off: usize,
    cmp: crate::expr::CmpOp,
    thr: i32,
) -> u8 {
    let start = base + off;
    let v = i32x8::from(&vals[start..start + 8]);
    let t = i32x8::splat(thr);
    let arr: [i32; 8] = match cmp {
        crate::expr::CmpOp::Eq => v.cmp_eq(t).to_array(),
        crate::expr::CmpOp::Lt => v.cmp_lt(t).to_array(),
        crate::expr::CmpOp::Gt => v.cmp_gt(t).to_array(),
        crate::expr::CmpOp::Lte => {
            let gt = v.cmp_gt(t).to_array();
            let mut out = [0; 8];
            for j in 0..8 {
                out[j] = if gt[j] != 0 { 0 } else { 1 };
            }
            out
        }
        crate::expr::CmpOp::Gte => {
            let lt = v.cmp_lt(t).to_array();
            let mut out = [0; 8];
            for j in 0..8 {
                out[j] = if lt[j] != 0 { 0 } else { 1 };
            }
            out
        }
        crate::expr::CmpOp::Neq => {
            let eq = v.cmp_eq(t).to_array();
            let mut out = [0; 8];
            for j in 0..8 {
                out[j] = if eq[j] != 0 { 0 } else { 1 };
            }
            out
        }
    };
    let mut bits: u8 = 0;
    for (j, &val) in arr.iter().enumerate() {
        let ok = val != 0;
        let not_null = !nulls.get(start + j).map(|b| *b).unwrap_or(false);
        if ok && not_null {
            bits |= 1 << j;
        }
    }
    bits
}

#[inline]
pub fn mask8_rows_f64(
    vals: &[f64],
    nulls: &BitVec,
    base: usize,
    off: usize,
    cmp: crate::expr::CmpOp,
    thr: f64,
) -> u8 {
    let start = base + off;
    let v = f64x8::from_slice(&vals[start..start + 8]);
    let t = f64x8::splat(thr);
    let bits = match cmp {
        crate::expr::CmpOp::Eq => v.cmp_eq(t),
        crate::expr::CmpOp::Neq => !v.cmp_eq(t),
        crate::expr::CmpOp::Lt => v.cmp_lt(t),
        crate::expr::CmpOp::Lte => v.cmp_le(t),
        crate::expr::CmpOp::Gt => v.cmp_gt(t),
        crate::expr::CmpOp::Gte => v.cmp_ge(t),
    };
    let mut nn: u8 = 0;
    for j in 0..8 {
        if !nulls.get(start + j).map(|b| *b).unwrap_or(false) {
            nn |= 1 << j;
        }
    }
    bits & nn
}

#[inline]
pub fn mask8_rows_i64(
    vals: &[i64],
    nulls: &BitVec,
    base: usize,
    off: usize,
    cmp: crate::expr::CmpOp,
    thr: i64,
) -> u8 {
    let start = base + off;
    let v = i64x8::from_slice(&vals[start..start + 8]);
    let t = i64x8::splat(thr);
    let bits = match cmp {
        crate::expr::CmpOp::Eq => v.cmp_eq(t),
        crate::expr::CmpOp::Neq => !v.cmp_eq(t),
        crate::expr::CmpOp::Lt => v.cmp_lt(t),
        crate::expr::CmpOp::Lte => v.cmp_lte(t),
        crate::expr::CmpOp::Gt => v.cmp_gt(t),
        crate::expr::CmpOp::Gte => v.cmp_gte(t),
    };
    let mut nn: u8 = 0;
    for j in 0..8 {
        if !nulls.get(start + j).map(|b| *b).unwrap_or(false) {
            nn |= 1 << j;
        }
    }
    bits & nn
}

#[inline]
pub fn mask8_ranges_f32(
    min: &[f32],
    max: &[f32],
    non_null: &[usize],
    off: usize,
    cmp: crate::expr::CmpOp,
    thr: f32,
) -> u8 {
    let minv = f32x8::from(&min[off..off + 8]);
    let maxv = f32x8::from(&max[off..off + 8]);
    let t = f32x8::splat(thr);
    let arr = match cmp {
        crate::expr::CmpOp::Eq => (minv.cmp_le(t) & maxv.cmp_ge(t)).to_array(),
        crate::expr::CmpOp::Lt => minv.cmp_lt(t).to_array(),
        crate::expr::CmpOp::Lte => minv.cmp_le(t).to_array(),
        crate::expr::CmpOp::Gt => maxv.cmp_gt(t).to_array(),
        crate::expr::CmpOp::Gte => maxv.cmp_ge(t).to_array(),
        crate::expr::CmpOp::Neq => [1.0; 8],
    };
    let mut bits: u8 = 0;
    for j in 0..8 {
        if arr[j] != 0.0 && non_null[off + j] > 0 {
            bits |= 1 << j;
        }
    }
    bits
}

#[inline]
pub fn mask8_ranges_f64(
    min: &[f64],
    max: &[f64],
    non_null: &[usize],
    off: usize,
    cmp: crate::expr::CmpOp,
    thr: f64,
) -> u8 {
    let minv = f64x8::from_slice(&min[off..off + 8]);
    let maxv = f64x8::from_slice(&max[off..off + 8]);
    let t = f64x8::splat(thr);
    let bits = match cmp {
        crate::expr::CmpOp::Eq => minv.cmp_le(t) & maxv.cmp_ge(t),
        crate::expr::CmpOp::Lt => minv.cmp_lt(t),
        crate::expr::CmpOp::Lte => minv.cmp_le(t),
        crate::expr::CmpOp::Gt => maxv.cmp_gt(t),
        crate::expr::CmpOp::Gte => maxv.cmp_ge(t),
        crate::expr::CmpOp::Neq => 0xFF,
    };
    let mut nn: u8 = 0;
    for j in 0..8 {
        if non_null[off + j] > 0 {
            nn |= 1 << j;
        }
    }
    bits & nn
}

#[inline]
pub fn mask8_ranges_i32(
    min: &[i32],
    max: &[i32],
    non_null: &[usize],
    off: usize,
    cmp: crate::expr::CmpOp,
    thr: i32,
) -> u8 {
    let minv = i32x8::from(&min[off..off + 8]);
    let maxv = i32x8::from(&max[off..off + 8]);
    let t = i32x8::splat(thr);
    let arr: [i32; 8] = match cmp {
        crate::expr::CmpOp::Eq => {
            let gt = minv.cmp_gt(t).to_array();
            let lt = maxv.cmp_lt(t).to_array();
            let mut out = [0; 8];
            for j in 0..8 {
                let lte = if gt[j] != 0 { 0 } else { 1 };
                let gte = if lt[j] != 0 { 0 } else { 1 };
                out[j] = if lte != 0 && gte != 0 { 1 } else { 0 };
            }
            out
        }
        crate::expr::CmpOp::Lt => minv.cmp_lt(t).to_array(),
        crate::expr::CmpOp::Lte => {
            let gt = minv.cmp_gt(t).to_array();
            let mut out = [0; 8];
            for j in 0..8 {
                out[j] = if gt[j] != 0 { 0 } else { 1 };
            }
            out
        }
        crate::expr::CmpOp::Gt => maxv.cmp_gt(t).to_array(),
        crate::expr::CmpOp::Gte => {
            let lt = maxv.cmp_lt(t).to_array();
            let mut out = [0; 8];
            for j in 0..8 {
                out[j] = if lt[j] != 0 { 0 } else { 1 };
            }
            out
        }
        crate::expr::CmpOp::Neq => [1; 8],
    };
    let mut bits: u8 = 0;
    for j in 0..8 {
        if arr[j] != 0 && non_null[off + j] > 0 {
            bits |= 1 << j;
        }
    }
    bits
}

#[inline]
pub fn mask8_ranges_i64(
    min: &[i64],
    max: &[i64],
    non_null: &[usize],
    off: usize,
    cmp: crate::expr::CmpOp,
    thr: i64,
) -> u8 {
    let minv = i64x8::from_slice(&min[off..off + 8]);
    let maxv = i64x8::from_slice(&max[off..off + 8]);
    let t = i64x8::splat(thr);
    let bits = match cmp {
        crate::expr::CmpOp::Eq => minv.cmp_lte(t) & maxv.cmp_gte(t),
        crate::expr::CmpOp::Lt => minv.cmp_lt(t),
        crate::expr::CmpOp::Lte => minv.cmp_lte(t),
        crate::expr::CmpOp::Gt => maxv.cmp_gt(t),
        crate::expr::CmpOp::Gte => maxv.cmp_gte(t),
        crate::expr::CmpOp::Neq => 0xFF,
    };
    let mut nn: u8 = 0;
    for j in 0..8 {
        if non_null[off + j] > 0 {
            nn |= 1 << j;
        }
    }
    bits & nn
}

#[inline]
pub fn apply_rows_mask_f32(
    vals: &[f32],
    nulls: &BitVec,
    base: usize,
    len: usize,
    cmp: crate::expr::CmpOp,
    thr: f32,
    out: &mut BitVec,
) {
    let mut off = 0;
    while off + 8 <= len {
        let bits = mask8_rows_f32(vals, nulls, base, off, cmp, thr);
        for j in 0..8 {
            if (bits >> j) & 1 == 1 {
                out.set(off + j, true);
            }
        }
        off += 8;
    }
    while off < len {
        let v = vals[base + off];
        let is_null = nulls.get(base + off).map(|b| *b).unwrap_or(false);
        let sat = match cmp {
            crate::expr::CmpOp::Eq => v == thr,
            crate::expr::CmpOp::Neq => v != thr,
            crate::expr::CmpOp::Lt => v < thr,
            crate::expr::CmpOp::Lte => v <= thr,
            crate::expr::CmpOp::Gt => v > thr,
            crate::expr::CmpOp::Gte => v >= thr,
        };
        if sat && !is_null {
            out.set(off, true);
        }
        off += 1;
    }
}

#[inline]
pub fn apply_rows_mask_i32(
    vals: &[i32],
    nulls: &BitVec,
    base: usize,
    len: usize,
    cmp: crate::expr::CmpOp,
    thr: i32,
    out: &mut BitVec,
) {
    let mut off = 0;
    while off + 8 <= len {
        let bits = mask8_rows_i32(vals, nulls, base, off, cmp, thr);
        for j in 0..8 {
            if (bits >> j) & 1 == 1 {
                out.set(off + j, true);
            }
        }
        off += 8;
    }
    while off < len {
        let v = vals[base + off];
        let is_null = nulls.get(base + off).map(|b| *b).unwrap_or(false);
        let sat = match cmp {
            crate::expr::CmpOp::Eq => v == thr,
            crate::expr::CmpOp::Neq => v != thr,
            crate::expr::CmpOp::Lt => v < thr,
            crate::expr::CmpOp::Lte => v <= thr,
            crate::expr::CmpOp::Gt => v > thr,
            crate::expr::CmpOp::Gte => v >= thr,
        };
        if sat && !is_null {
            out.set(off, true);
        }
        off += 1;
    }
}

#[inline]
pub fn apply_rows_mask_f64(
    vals: &[f64],
    nulls: &BitVec,
    base: usize,
    len: usize,
    cmp: crate::expr::CmpOp,
    thr: f64,
    out: &mut BitVec,
) {
    let mut off = 0;
    while off + 8 <= len {
        let bits = mask8_rows_f64(vals, nulls, base, off, cmp, thr);
        for j in 0..8 {
            if (bits >> j) & 1 == 1 {
                out.set(off + j, true);
            }
        }
        off += 8;
    }
    while off < len {
        let is_null = nulls.get(base + off).map(|b| *b).unwrap_or(false);
        let v = vals[base + off];
        let sat = match cmp {
            crate::expr::CmpOp::Eq => v == thr,
            crate::expr::CmpOp::Neq => v != thr,
            crate::expr::CmpOp::Lt => v < thr,
            crate::expr::CmpOp::Lte => v <= thr,
            crate::expr::CmpOp::Gt => v > thr,
            crate::expr::CmpOp::Gte => v >= thr,
        };
        if sat && !is_null {
            out.set(off, true);
        }
        off += 1;
    }
}

#[inline]
pub fn apply_rows_mask_i64(
    vals: &[i64],
    nulls: &BitVec,
    base: usize,
    len: usize,
    cmp: crate::expr::CmpOp,
    thr: i64,
    out: &mut BitVec,
) {
    let mut off = 0;
    while off + 8 <= len {
        let bits = mask8_rows_i64(vals, nulls, base, off, cmp, thr);
        for j in 0..8 {
            if (bits >> j) & 1 == 1 {
                out.set(off + j, true);
            }
        }
        off += 8;
    }
    while off < len {
        let is_null = nulls.get(base + off).map(|b| *b).unwrap_or(false);
        let v = vals[base + off];
        let sat = match cmp {
            crate::expr::CmpOp::Eq => v == thr,
            crate::expr::CmpOp::Neq => v != thr,
            crate::expr::CmpOp::Lt => v < thr,
            crate::expr::CmpOp::Lte => v <= thr,
            crate::expr::CmpOp::Gt => v > thr,
            crate::expr::CmpOp::Gte => v >= thr,
        };
        if sat && !is_null {
            out.set(off, true);
        }
        off += 1;
    }
}

// High-level typed helpers write into BitVec variants below; boolean-slice variants removed as unused.
#[inline]
pub fn apply_chunk_mask_ranges_f32_bits(
    min: &[f32],
    max: &[f32],
    non_null: &[usize],
    n_chunks: usize,
    cmp: crate::expr::CmpOp,
    thr: f32,
    out: &mut BitVec,
) {
    let mut i = 0;
    while i + 8 <= n_chunks {
        let bits = mask8_ranges_f32(min, max, non_null, i, cmp, thr);
        for j in 0..8 {
            if (bits >> j) & 1 == 1 {
                out.set(i + j, true);
            }
        }
        i += 8;
    }
    while i < n_chunks {
        let mn = min[i];
        let mx = max[i];
        let sat = match cmp {
            crate::expr::CmpOp::Eq => mn <= thr && thr <= mx,
            crate::expr::CmpOp::Lt => mn < thr,
            crate::expr::CmpOp::Lte => mn <= thr,
            crate::expr::CmpOp::Gt => mx > thr,
            crate::expr::CmpOp::Gte => mx >= thr,
            crate::expr::CmpOp::Neq => true,
        } && non_null[i] > 0;
        if sat {
            out.set(i, true);
        }
        i += 1;
    }
}

#[inline]
pub fn apply_chunk_mask_ranges_f64_bits(
    min: &[f64],
    max: &[f64],
    non_null: &[usize],
    n_chunks: usize,
    cmp: crate::expr::CmpOp,
    thr: f64,
    out: &mut BitVec,
) {
    let mut i = 0;
    while i + 8 <= n_chunks {
        let bits = mask8_ranges_f64(min, max, non_null, i, cmp, thr);
        for j in 0..8 {
            if (bits >> j) & 1 == 1 {
                out.set(i + j, true);
            }
        }
        i += 8;
    }
    while i < n_chunks {
        let mn = min[i];
        let mx = max[i];
        let sat = match cmp {
            crate::expr::CmpOp::Eq => mn <= thr && thr <= mx,
            crate::expr::CmpOp::Lt => mn < thr,
            crate::expr::CmpOp::Lte => mn <= thr,
            crate::expr::CmpOp::Gt => mx > thr,
            crate::expr::CmpOp::Gte => mx >= thr,
            crate::expr::CmpOp::Neq => true,
        } && non_null[i] > 0;
        if sat {
            out.set(i, true);
        }
        i += 1;
    }
}

#[inline]
pub fn apply_chunk_mask_ranges_i32_bits(
    min: &[i32],
    max: &[i32],
    non_null: &[usize],
    n_chunks: usize,
    cmp: crate::expr::CmpOp,
    thr: i32,
    out: &mut BitVec,
) {
    let mut i = 0;
    while i + 8 <= n_chunks {
        let bits = mask8_ranges_i32(min, max, non_null, i, cmp, thr);
        for j in 0..8 {
            if (bits >> j) & 1 == 1 {
                out.set(i + j, true);
            }
        }
        i += 8;
    }
    while i < n_chunks {
        let mn = min[i];
        let mx = max[i];
        let sat = match cmp {
            crate::expr::CmpOp::Eq => mn <= thr && thr <= mx,
            crate::expr::CmpOp::Lt => mn < thr,
            crate::expr::CmpOp::Lte => mn <= thr,
            crate::expr::CmpOp::Gt => mx > thr,
            crate::expr::CmpOp::Gte => mx >= thr,
            crate::expr::CmpOp::Neq => true,
        } && non_null[i] > 0;
        if sat {
            out.set(i, true);
        }
        i += 1;
    }
}

#[inline]
pub fn apply_chunk_mask_ranges_i64_bits(
    min: &[i64],
    max: &[i64],
    non_null: &[usize],
    n_chunks: usize,
    cmp: crate::expr::CmpOp,
    thr: i64,
    out: &mut BitVec,
) {
    let mut i = 0;
    while i + 8 <= n_chunks {
        let bits = mask8_ranges_i64(min, max, non_null, i, cmp, thr);
        for j in 0..8 {
            if (bits >> j) & 1 == 1 {
                out.set(i + j, true);
            }
        }
        i += 8;
    }
    while i < n_chunks {
        let mn = min[i];
        let mx = max[i];
        let sat = match cmp {
            crate::expr::CmpOp::Eq => mn <= thr && thr <= mx,
            crate::expr::CmpOp::Lt => mn < thr,
            crate::expr::CmpOp::Lte => mn <= thr,
            crate::expr::CmpOp::Gt => mx > thr,
            crate::expr::CmpOp::Gte => mx >= thr,
            crate::expr::CmpOp::Neq => true,
        } && non_null[i] > 0;
        if sat {
            out.set(i, true);
        }
        i += 1;
    }
}
