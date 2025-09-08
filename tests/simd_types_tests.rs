// This file contains tests for the SIMD types defined in `otters::type_utils`.
// Tests for SIMD wrapper types and masking helpers.
use otters::type_utils::*;

#[cfg(test)]
mod i64x8_tests {
    use super::*;

    #[test]
    fn test_i64x8_splat() {
        let vec = i64x8::splat(42);
        let slice = [42i64; 8];
        let expected = i64x8::from_slice(&slice);

        // Test by converting back to check equality
        let vec_mask = vec.cmp_eq(expected);
        assert_eq!(vec_mask, 0xFF); // All 8 bits should be set
    }

    #[test]
    fn test_i64x8_from_slice() {
        let data = [1, 2, 3, 4, 5, 6, 7, 8];
        let vec = i64x8::from_slice(&data);

        // Test by comparing with known values
        let expected1 = i64x8::splat(1);
        let expected8 = i64x8::splat(8);

        // Should not be equal to splat of any single value
        assert_ne!(vec.cmp_eq(expected1), 0xFF);
        assert_ne!(vec.cmp_eq(expected8), 0xFF);
    }

    #[test]
    fn test_i64x8_cmp_eq() {
        let vec1 = i64x8::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let vec2 = i64x8::from_slice(&[1, 2, 3, 4, 9, 10, 11, 12]);

        let mask = vec1.cmp_eq(vec2);
        // First 4 elements should be equal (bits 0-3 set), last 4 should not (bits 4-7 clear)
        assert_eq!(mask & 0x0F, 0x0F); // Lower 4 bits set
        assert_eq!(mask & 0xF0, 0x00); // Upper 4 bits clear
    }

    #[test]
    fn test_i64x8_cmp_gt() {
        let vec1 = i64x8::from_slice(&[5, 4, 3, 2, 1, 0, -1, -2]);
        let vec2 = i64x8::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let mask = vec1.cmp_gt(vec2);
        // Elements 0, 1 should be greater, others should not
        assert_eq!(mask & 0x03, 0x03); // First 2 bits set
    }

    #[test]
    fn test_i64x8_cmp_gte() {
        let vec1 = i64x8::from_slice(&[5, 4, 3, 2, 1, 0, -1, -2]);
        let vec2 = i64x8::from_slice(&[5, 3, 3, 3, 1, 1, 0, 0]);

        let mask = vec1.cmp_gte(vec2);
        // Elements where vec1 >= vec2: 0 (5>=5), 1 (4>=3), 2 (3>=3), 4 (1>=1)
        let expected_bits = 0b00010111; // bits 0, 1, 2, 4
        assert_eq!(mask & expected_bits, expected_bits);
    }

    #[test]
    fn test_i64x8_cmp_lt() {
        let vec1 = i64x8::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let vec2 = i64x8::from_slice(&[5, 4, 3, 2, 1, 0, -1, -2]);

        let mask = vec1.cmp_lt(vec2);
        // Elements 0, 1 should be less than corresponding elements in vec2
        assert_eq!(mask & 0x03, 0x03); // First 2 bits set
    }

    #[test]
    fn test_i64x8_cmp_lte() {
        let vec1 = i64x8::from_slice(&[1, 3, 3, 4, 1, 0, -1, -2]);
        let vec2 = i64x8::from_slice(&[5, 3, 3, 2, 1, 0, 0, 0]);

        let mask = vec1.cmp_lte(vec2);
        // Elements where vec1 <= vec2: 0 (1<=5), 1 (3<=3), 2 (3<=3), 4 (1<=1), 5 (0<=0)
        let expected_bits = 0b00110111; // bits 0, 1, 2, 4, 5
        assert_eq!(mask & expected_bits, expected_bits);
    }

    #[test]
    fn test_i64x8_min() {
        let vec1 = i64x8::from_slice(&[5, 2, 7, 1, 9, 3, 8, 4]);
        let vec2 = i64x8::from_slice(&[3, 6, 4, 8, 2, 7, 1, 9]);

        let result = vec1.min(vec2);
        let expected = i64x8::from_slice(&[3, 2, 4, 1, 2, 3, 1, 4]);

        let mask = result.cmp_eq(expected);
        assert_eq!(mask, 0xFF); // All elements should match
    }

    #[test]
    fn test_i64x8_max() {
        let vec1 = i64x8::from_slice(&[5, 2, 7, 1, 9, 3, 8, 4]);
        let vec2 = i64x8::from_slice(&[3, 6, 4, 8, 2, 7, 1, 9]);

        let result = vec1.max(vec2);
        let expected = i64x8::from_slice(&[5, 6, 7, 8, 9, 7, 8, 9]);

        let mask = result.cmp_eq(expected);
        assert_eq!(mask, 0xFF); // All elements should match
    }
}

#[cfg(test)]
mod f64x8_tests {
    use super::*;

    #[test]
    fn test_f64x8_splat() {
        let vec = f64x8::splat(std::f64::consts::PI);
        let slice = [std::f64::consts::PI; 8];
        let expected = f64x8::from_slice(&slice);

        let mask = vec.cmp_eq(expected);
        assert_eq!(mask, 0xFF); // All 8 bits should be set
    }

    #[test]
    fn test_f64x8_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let vec = f64x8::from_slice(&data);

        let expected_all_ones = f64x8::splat(1.0);
        let mask = vec.cmp_eq(expected_all_ones);
        assert_eq!(mask & 0xFE, 0x00); // Only first element should match
        assert_eq!(mask & 0x01, 0x01); // First element should match
    }

    #[test]
    fn test_f64x8_cmp_eq() {
        let vec1 = f64x8::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let vec2 = f64x8::from_slice(&[1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);

        let mask = vec1.cmp_eq(vec2);
        assert_eq!(mask & 0x0F, 0x0F); // Lower 4 bits set
        assert_eq!(mask & 0xF0, 0x00); // Upper 4 bits clear
    }

    #[test]
    fn test_f64x8_cmp_gt() {
        let vec1 = f64x8::from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]);
        let vec2 = f64x8::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let mask = vec1.cmp_gt(vec2);
        // Elements 0, 1 should be greater
        assert_eq!(mask & 0x03, 0x03); // First 2 bits set
    }

    #[test]
    fn test_f64x8_cmp_ge() {
        let vec1 = f64x8::from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]);
        let vec2 = f64x8::from_slice(&[5.0, 3.0, 3.0, 3.0, 1.0, 1.0, 0.0, 0.0]);

        let mask = vec1.cmp_ge(vec2);
        // Elements where vec1 >= vec2: 0, 1, 2, 4
        let expected_bits = 0b00010111;
        assert_eq!(mask & expected_bits, expected_bits);
    }

    #[test]
    fn test_f64x8_cmp_lt() {
        let vec1 = f64x8::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let vec2 = f64x8::from_slice(&[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]);

        let mask = vec1.cmp_lt(vec2);
        // Elements 0, 1 should be less
        assert_eq!(mask & 0x03, 0x03);
    }

    #[test]
    fn test_f64x8_cmp_le() {
        let vec1 = f64x8::from_slice(&[1.0, 3.0, 3.0, 4.0, 1.0, 0.0, -1.0, -2.0]);
        let vec2 = f64x8::from_slice(&[5.0, 3.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0]);

        let mask = vec1.cmp_le(vec2);
        // Elements where vec1 <= vec2: 0, 1, 2, 4, 5
        let expected_bits = 0b00110111;
        assert_eq!(mask & expected_bits, expected_bits);
    }

    #[test]
    fn test_f64x8_min() {
        let vec1 = f64x8::from_slice(&[5.0, 2.0, 7.0, 1.0, 9.0, 3.0, 8.0, 4.0]);
        let vec2 = f64x8::from_slice(&[3.0, 6.0, 4.0, 8.0, 2.0, 7.0, 1.0, 9.0]);

        let result = vec1.min(vec2);
        let expected = f64x8::from_slice(&[3.0, 2.0, 4.0, 1.0, 2.0, 3.0, 1.0, 4.0]);

        let mask = result.cmp_eq(expected);
        assert_eq!(mask, 0xFF);
    }

    #[test]
    fn test_f64x8_max() {
        let vec1 = f64x8::from_slice(&[5.0, 2.0, 7.0, 1.0, 9.0, 3.0, 8.0, 4.0]);
        let vec2 = f64x8::from_slice(&[3.0, 6.0, 4.0, 8.0, 2.0, 7.0, 1.0, 9.0]);

        let result = vec1.max(vec2);
        let expected = f64x8::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0]);

        let mask = result.cmp_eq(expected);
        assert_eq!(mask, 0xFF);
    }
}

#[cfg(test)]
mod u64x8_tests {
    use super::*;

    #[test]
    fn test_u64x8_splat() {
        let vec = u64x8::splat(42);
        let slice = [42u64; 8];
        let expected = u64x8::from_slice(&slice);

        let mask = vec.cmp_eq(expected);
        assert_eq!(mask, 0xFF); // All 8 bits should be set
    }

    #[test]
    fn test_u64x8_from_slice() {
        let data = [1, 2, 3, 4, 5, 6, 7, 8];
        let vec = u64x8::from_slice(&data);

        let expected_all_ones = u64x8::splat(1);
        let mask = vec.cmp_eq(expected_all_ones);
        assert_eq!(mask & 0xFE, 0x00); // Only first element should match
        assert_eq!(mask & 0x01, 0x01); // First element should match
    }

    #[test]
    fn test_u64x8_cmp_eq() {
        let vec1 = u64x8::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let vec2 = u64x8::from_slice(&[1, 2, 3, 4, 9, 10, 11, 12]);

        let mask = vec1.cmp_eq(vec2);
        assert_eq!(mask & 0x0F, 0x0F); // Lower 4 bits set
        assert_eq!(mask & 0xF0, 0x00); // Upper 4 bits clear
    }

    #[test]
    fn test_u64x8_cmp_gt() {
        let vec1 = u64x8::from_slice(&[10, 9, 8, 7, 6, 5, 4, 3]);
        let vec2 = u64x8::from_slice(&[5, 6, 7, 8, 9, 10, 11, 12]);

        let mask = vec1.cmp_gt(vec2);
        // Elements 0, 1, 2 should be greater (10>5, 9>6, 8>7)
        assert_eq!(mask & 0x07, 0x07); // First 3 bits set
    }

    #[test]
    fn test_u64x8_cmp_gte() {
        let vec1 = u64x8::from_slice(&[10, 9, 8, 7, 6, 5, 4, 3]);
        let vec2 = u64x8::from_slice(&[10, 8, 8, 8, 6, 6, 5, 4]);

        let mask = vec1.cmp_gte(vec2);
        // Elements where vec1 >= vec2: 0 (10>=10), 1 (9>=8), 2 (8>=8), 4 (6>=6)
        let expected_bits = 0b00010111; // bits 0, 1, 2, 4
        assert_eq!(mask & expected_bits, expected_bits);
    }

    #[test]
    fn test_u64x8_cmp_lt() {
        let vec1 = u64x8::from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let vec2 = u64x8::from_slice(&[5, 4, 3, 2, 1, 0, 11, 12]);

        let mask = vec1.cmp_lt(vec2);
        // Elements where vec1 < vec2: 0 (1<5), 1 (2<4), 6 (7<11), 7 (8<12)
        let expected_bits = 0b11000011; // bits 0, 1, 6, 7
        assert_eq!(mask & expected_bits, expected_bits);
    }

    #[test]
    fn test_u64x8_cmp_lte() {
        let vec1 = u64x8::from_slice(&[1, 4, 3, 4, 1, 0, 7, 8]);
        let vec2 = u64x8::from_slice(&[5, 4, 3, 2, 1, 0, 6, 12]);

        let mask = vec1.cmp_lte(vec2);
        // Elements where vec1 <= vec2: 0 (1<=5), 1 (4<=4), 2 (3<=3), 4 (1<=1), 5 (0<=0), 7 (8<=12)
        let expected_bits = 0b10110111; // bits 0, 1, 2, 4, 5, 7
        assert_eq!(mask & expected_bits, expected_bits);
    }

    #[test]
    fn test_u64x8_min() {
        let vec1 = u64x8::from_slice(&[5, 2, 7, 1, 9, 3, 8, 4]);
        let vec2 = u64x8::from_slice(&[3, 6, 4, 8, 2, 7, 1, 9]);

        let result = vec1.min(vec2);
        let expected = u64x8::from_slice(&[3, 2, 4, 1, 2, 3, 1, 4]);

        let mask = result.cmp_eq(expected);
        assert_eq!(mask, 0xFF); // All elements should match
    }

    #[test]
    fn test_u64x8_max() {
        let vec1 = u64x8::from_slice(&[5, 2, 7, 1, 9, 3, 8, 4]);
        let vec2 = u64x8::from_slice(&[3, 6, 4, 8, 2, 7, 1, 9]);

        let result = vec1.max(vec2);
        let expected = u64x8::from_slice(&[5, 6, 7, 8, 9, 7, 8, 9]);

        let mask = result.cmp_eq(expected);
        assert_eq!(mask, 0xFF); // All elements should match
    }

    #[test]
    fn test_u64x8_large_values() {
        // Test with large unsigned values that would be negative if interpreted as signed
        let large_val1 = u64::MAX / 2 + 100;
        let large_val2 = u64::MAX / 2 + 50;

        let vec1 = u64x8::splat(large_val1);
        let vec2 = u64x8::splat(large_val2);

        let gt_mask = vec1.cmp_gt(vec2);
        assert_eq!(gt_mask, 0xFF); // large_val1 > large_val2

        let lt_mask = vec1.cmp_lt(vec2);
        assert_eq!(lt_mask, 0x00); // large_val1 not < large_val2
    }
}

#[cfg(test)]
mod cross_type_consistency_tests {
    use super::*;

    #[test]
    fn test_signed_unsigned_consistency_for_small_values() {
        // For small positive values, signed and unsigned should behave the same
        let small_values = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let signed_values: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

        let u_vec1 = u64x8::from_slice(&small_values);
        let u_vec2 = u64x8::from_slice(&[2, 1, 4, 3, 6, 5, 8, 7]);

        let i_vec1 = i64x8::from_slice(&signed_values);
        let i_vec2 = i64x8::from_slice(&[2, 1, 4, 3, 6, 5, 8, 7]);

        // Comparisons should yield the same results for small positive values
        assert_eq!(u_vec1.cmp_gt(u_vec2), i_vec1.cmp_gt(i_vec2));
        assert_eq!(u_vec1.cmp_lt(u_vec2), i_vec1.cmp_lt(i_vec2));
        assert_eq!(u_vec1.cmp_eq(u_vec2), i_vec1.cmp_eq(i_vec2));
    }

    #[test]
    fn test_min_max_consistency() {
        let values1 = [5, 2, 7, 1, 9, 3, 8, 4];
        let values2 = [3, 6, 4, 8, 2, 7, 1, 9];

        // Test with multiple types
        let i_vec1 = i64x8::from_slice(&values1.map(|x| x as i64));
        let i_vec2 = i64x8::from_slice(&values2.map(|x| x as i64));

        let u_vec1 = u64x8::from_slice(&values1.map(|x| x as u64));
        let u_vec2 = u64x8::from_slice(&values2.map(|x| x as u64));

        let f_vec1 = f64x8::from_slice(&values1.map(|x| x as f64));
        let f_vec2 = f64x8::from_slice(&values2.map(|x| x as f64));

        // Min and max should produce the same results (for positive values)
        let i_min = i_vec1.min(i_vec2);
        let u_min = u_vec1.min(u_vec2);
        let f_min = f_vec1.min(f_vec2);

        let i_max = i_vec1.max(i_vec2);
        let u_max = u_vec1.max(u_vec2);
        let f_max = f_vec1.max(f_vec2);

        // Convert back to compare (this is a basic consistency check)
        let expected_min = [3, 2, 4, 1, 2, 3, 1, 4];
        let expected_max = [5, 6, 7, 8, 9, 7, 8, 9];

        let expected_min_i = i64x8::from_slice(&expected_min.map(|x| x as i64));
        let expected_max_i = i64x8::from_slice(&expected_max.map(|x| x as i64));
        let expected_min_u = u64x8::from_slice(&expected_min.map(|x| x as u64));
        let expected_max_u = u64x8::from_slice(&expected_max.map(|x| x as u64));
        let expected_min_f = f64x8::from_slice(&expected_min.map(|x| x as f64));
        let expected_max_f = f64x8::from_slice(&expected_max.map(|x| x as f64));

        assert_eq!(i_min.cmp_eq(expected_min_i), 0xFF);
        assert_eq!(i_max.cmp_eq(expected_max_i), 0xFF);
        assert_eq!(u_min.cmp_eq(expected_min_u), 0xFF);
        assert_eq!(u_max.cmp_eq(expected_max_u), 0xFF);
        assert_eq!(f_min.cmp_eq(expected_min_f), 0xFF);
        assert_eq!(f_max.cmp_eq(expected_max_f), 0xFF);
    }
}
