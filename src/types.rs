// Wrapper types that provide 8-wide SIMD operations using two 4-wide operations
// partially vibe-coded :p
// Not all traits are implemented for all types in wide.
// Fall back non SIMD operations if necessary
#![allow(non_camel_case_types)]

use wide::*;

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
        // a >= b is equivalent to !(a < b)
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
        // a <= b is equivalent to !(a > b)
        !self.cmp_gt(other)
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        // Use conditional selection: choose other if self > other, else choose self
        let low_mask = self.low.cmp_gt(other.low);
        let high_mask = self.high.cmp_gt(other.high);

        Self {
            low: low_mask.blend(other.low, self.low),
            high: high_mask.blend(other.high, self.high),
        }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        // Use conditional selection: choose other if self < other, else choose self
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
        // For u64x4, let's try unsafe transmute to i64x4 for comparison
        use std::mem;
        let self_i64 = unsafe {
            i64x8 {
                low: mem::transmute(self.low),
                high: mem::transmute(self.high),
            }
        };
        let other_i64 = unsafe {
            i64x8 {
                low: mem::transmute(other.low),
                high: mem::transmute(other.high),
            }
        };
        self_i64.cmp_eq(other_i64)
    }

    #[inline]
    pub fn cmp_gt(self, other: Self) -> u8 {
        // For unsigned comparison, we need to handle the sign bit differently
        // We can't directly transmute to signed for gt comparison
        // Instead, we'll use element-wise comparison
        let mut result = 0u8;

        // Convert to arrays for element-wise comparison
        let self_low: [u64; 4] = unsafe { std::mem::transmute(self.low) };
        let other_low: [u64; 4] = unsafe { std::mem::transmute(other.low) };
        let self_high: [u64; 4] = unsafe { std::mem::transmute(self.high) };
        let other_high: [u64; 4] = unsafe { std::mem::transmute(other.high) };

        // Check low 4 elements
        for i in 0..4 {
            if self_low[i] > other_low[i] {
                result |= 1 << i;
            }
        }

        // Check high 4 elements
        for i in 0..4 {
            if self_high[i] > other_high[i] {
                result |= 1 << (i + 4);
            }
        }

        result
    }

    #[inline]
    pub fn cmp_gte(self, other: Self) -> u8 {
        // a >= b is equivalent to !(a < b)
        !self.cmp_lt(other)
    }

    #[inline]
    pub fn cmp_lt(self, other: Self) -> u8 {
        // For unsigned comparison
        let mut result = 0u8;

        // Convert to arrays for element-wise comparison
        let self_low: [u64; 4] = unsafe { std::mem::transmute(self.low) };
        let other_low: [u64; 4] = unsafe { std::mem::transmute(other.low) };
        let self_high: [u64; 4] = unsafe { std::mem::transmute(self.high) };
        let other_high: [u64; 4] = unsafe { std::mem::transmute(other.high) };

        // Check low 4 elements
        for i in 0..4 {
            if self_low[i] < other_low[i] {
                result |= 1 << i;
            }
        }

        // Check high 4 elements
        for i in 0..4 {
            if self_high[i] < other_high[i] {
                result |= 1 << (i + 4);
            }
        }

        result
    }

    #[inline]
    pub fn cmp_lte(self, other: Self) -> u8 {
        // a <= b is equivalent to !(a > b)
        !self.cmp_gt(other)
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        let mut result_low = [0u64; 4];
        let mut result_high = [0u64; 4];

        // Convert to arrays for element-wise comparison
        let self_low: [u64; 4] = unsafe { std::mem::transmute(self.low) };
        let other_low: [u64; 4] = unsafe { std::mem::transmute(other.low) };
        let self_high: [u64; 4] = unsafe { std::mem::transmute(self.high) };
        let other_high: [u64; 4] = unsafe { std::mem::transmute(other.high) };

        // Min for low 4 elements
        for i in 0..4 {
            result_low[i] = self_low[i].min(other_low[i]);
        }

        // Min for high 4 elements
        for i in 0..4 {
            result_high[i] = self_high[i].min(other_high[i]);
        }

        Self {
            low: unsafe { std::mem::transmute(result_low) },
            high: unsafe { std::mem::transmute(result_high) },
        }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        let mut result_low = [0u64; 4];
        let mut result_high = [0u64; 4];

        // Convert to arrays for element-wise comparison
        let self_low: [u64; 4] = unsafe { std::mem::transmute(self.low) };
        let other_low: [u64; 4] = unsafe { std::mem::transmute(other.low) };
        let self_high: [u64; 4] = unsafe { std::mem::transmute(self.high) };
        let other_high: [u64; 4] = unsafe { std::mem::transmute(other.high) };

        // Max for low 4 elements
        for i in 0..4 {
            result_low[i] = self_low[i].max(other_low[i]);
        }

        // Max for high 4 elements
        for i in 0..4 {
            result_high[i] = self_high[i].max(other_high[i]);
        }

        Self {
            low: unsafe { std::mem::transmute(result_low) },
            high: unsafe { std::mem::transmute(result_high) },
        }
    }
}
