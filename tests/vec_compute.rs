use otters::vec_compute::{
    cosine_similarity, dot_product, euclidean_distance_squared, inverse_norm,
};

#[test]
fn dot_and_euclidean_agree_on_simple_vectors() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [1.0f32, 2.0, 4.0];
    assert_eq!(dot_product(&a, &b), 1.0 + 4.0 + 12.0);
    assert_eq!(euclidean_distance_squared(&a, &b), 1.0);
}

#[test]
fn cosine_uses_inverse_norms() {
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    let inv_a = inverse_norm(&a);
    let inv_b = inverse_norm(&b);
    let sim = cosine_similarity(&a, &b, inv_a, inv_b);
    assert_eq!(sim, 0.0);

    let inv_a = inverse_norm(&a);
    let inv_same = inverse_norm(&a);
    let sim_self = cosine_similarity(&a, &a, inv_a, inv_same);
    assert!((sim_self - 1.0).abs() < 1e-6);
}

#[test]
fn inverse_norm_of_zero_vector_is_zero() {
    let zero = [0.0f32, 0.0];
    assert_eq!(inverse_norm(&zero), 0.0);
}
