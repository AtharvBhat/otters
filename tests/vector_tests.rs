// Tests for vector operations
use otters::vec::RowAlignedVecs;
use otters::vec::TopKIterator;

/// Helper function to create a standard set of test vectors for basic functionality tests
fn create_test_vectors() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
        vec![0.5, 0.5, 0.5],
    ]
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[test]
fn test_row_aligned_creation() {
    let mut row_vecs = RowAlignedVecs::new(3);
    assert_eq!(row_vecs.add_vector(vec![1.0, 2.0, 3.0]), Ok(()));

    // Test dimension mismatch
    assert!(row_vecs.add_vector(vec![1.0, 2.0]).is_err());
}

// ============================================================================
// Algorithm Correctness Tests
// ============================================================================

#[test]
fn test_cosine_similarity_basic() {
    let test_vectors = create_test_vectors();

    let mut row_vecs = RowAlignedVecs::new(3);
    row_vecs.add_vectors(test_vectors.clone()).unwrap();

    // Query with the first vector (should have similarity 1.0 with itself)
    let query = &test_vectors[0];
    let similarities: Vec<_> = row_vecs.search_vec_cosine(query).collect();

    // Check that the similarity with itself is approximately 1.0
    // The iterator returns (index, similarity) pairs
    // We need to find the one with index 0 (the identical vector)
    let self_similarity = similarities.iter().find(|(idx, _)| *idx == 0).unwrap();
    assert!((self_similarity.1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_euclidean_distance_basic() {
    let test_vectors = create_test_vectors();

    let mut row_vecs = RowAlignedVecs::new(3);
    row_vecs.add_vectors(test_vectors.clone()).unwrap();

    // Query with the first vector (should have distance 0.0 with itself)
    let query = &test_vectors[0];
    let distances: Vec<_> = row_vecs.search_vec_euclidean(query).collect();

    // Check that the distance with itself is approximately 0.0
    assert!(distances[0].1.abs() < 1e-6);
}

#[test]
fn test_top_k_selection() {
    let mut row_vecs = RowAlignedVecs::new(2);

    // Add vectors with known similarities to query [1.0, 0.0]
    row_vecs.add_vector(vec![1.0, 0.0]).unwrap(); // identical
    row_vecs.add_vector(vec![0.8, 0.6]).unwrap(); // high similarity
    row_vecs.add_vector(vec![0.0, 1.0]).unwrap(); // orthogonal
    row_vecs.add_vector(vec![-1.0, 0.0]).unwrap(); // opposite

    let query = vec![1.0, 0.0];

    // Test cosine similarity top-k
    let top_2_cosine = row_vecs.search_vec_cosine(&query).take_max(2);
    assert_eq!(top_2_cosine.len(), 2);

    // Test euclidean distance top-k
    let closest_2_euclidean = row_vecs.search_vec_euclidean(&query).take_min(2);
    assert_eq!(closest_2_euclidean.len(), 2);
}

#[test]
fn test_dot_product_simd() {
    let row_vecs = RowAlignedVecs::new(16);

    let vec1 = vec![1.0; 16];
    let vec2 = vec![2.0; 16];

    let result = row_vecs.dot_product(&vec1, &vec2);
    assert_eq!(result, 32.0); // 16 * 1.0 * 2.0
}

#[test]
fn test_euclidean_distance_simd() {
    let row_vecs = RowAlignedVecs::new(16);

    let vec1 = vec![1.0; 16];
    let vec2 = vec![2.0; 16];

    let result = row_vecs.euclidean_distance_squared(&vec1, &vec2);
    assert_eq!(result, 16.0); // 16 * (1.0 - 2.0)^2
}

#[test]
fn test_orthogonal_vectors_cosine() {
    let mut row_vecs = RowAlignedVecs::new(2);

    // Add orthogonal vectors
    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    row_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0];
    let results = row_vecs.search_vec_cosine(&query).take_max(2);

    // First vector should have similarity 1.0 (parallel)
    assert!((results[0].1 - 1.0).abs() < 1e-6);

    // Second vector should have similarity 0.0 (orthogonal)
    assert!(results[1].1.abs() < 1e-6);
}

#[test]
fn test_empty_results() {
    let row_vecs = RowAlignedVecs::new(3);
    let query = vec![1.0, 0.0, 0.0];

    let results = row_vecs.search_vec_cosine(&query).take_max(5);
    assert_eq!(results.len(), 0);
}

#[test]
fn test_take_more_than_available() {
    let mut row_vecs = RowAlignedVecs::new(2);

    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    row_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0];

    // Ask for more results than available
    let results = row_vecs.search_vec_cosine(&query).take_max(10);

    // Should return all available results (2)
    assert_eq!(results.len(), 2);
}

#[test]
fn test_normalized_vectors() {
    let mut row_vecs = RowAlignedVecs::new(3);

    // Add vectors with different magnitudes
    let vectors = vec![
        vec![2.0, 0.0, 0.0], // magnitude 2
        vec![0.0, 3.0, 0.0], // magnitude 3
        vec![0.0, 0.0, 4.0], // magnitude 4
    ];

    row_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0, 0.0]; // unit vector
    let results = row_vecs.search_vec_cosine(&query).take_max(3);

    // First result should be the parallel vector with similarity 1.0
    assert_eq!(results[0].0, 0);
    assert!((results[0].1 - 1.0).abs() < 1e-6);

    // Other vectors should have similarity 0.0 (orthogonal)
    assert!(results[1].1.abs() < 1e-6);
    assert!(results[2].1.abs() < 1e-6);
}

#[test]
fn test_zero_vectors() {
    let mut row_vecs = RowAlignedVecs::new(3);

    // Add zero vector
    let vectors = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
    row_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let results = row_vecs.search_vec_cosine(&query).take_max(2);

    // Zero vector should handle division by zero gracefully
    // Results should still be computed for the non-zero vector
    assert_eq!(results.len(), 2);

    // Non-zero vector should have similarity 1.0
    let non_zero_result = results.iter().find(|(idx, _)| *idx == 1).unwrap();
    assert!((non_zero_result.1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_negative_values() {
    let mut row_vecs = RowAlignedVecs::new(3);

    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![-1.0, 0.0, 0.0], // Opposite direction
        vec![0.0, -1.0, 0.0], // Different axis, negative
    ];

    row_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0, 0.0];
    let results = row_vecs.search_vec_cosine(&query).take_max(3);

    // First result should be parallel vector (similarity 1.0)
    assert_eq!(results[0].0, 0);
    assert!((results[0].1 - 1.0).abs() < 1e-6);

    // Second result should be anti-parallel vector (similarity -1.0)
    let anti_parallel = results.iter().find(|(idx, _)| *idx == 1).unwrap();
    assert!((anti_parallel.1 - (-1.0)).abs() < 1e-6);

    // Third result should be orthogonal (similarity 0.0)
    let orthogonal = results.iter().find(|(idx, _)| *idx == 2).unwrap();
    assert!(orthogonal.1.abs() < 1e-6);
}

#[test]
fn test_euclidean_distance_edge_cases() {
    let mut row_vecs = RowAlignedVecs::new(2);

    let vectors = vec![
        vec![0.0, 0.0],   // Same as query
        vec![3.0, 4.0],   // Squared distance 25.0 from origin
        vec![-3.0, -4.0], // Squared distance 25.0 from origin, opposite direction
    ];

    row_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![0.0, 0.0];
    let results = row_vecs.search_vec_euclidean(&query).take_min(3);

    // First result should be identical vector with squared distance 0.0
    assert_eq!(results[0].0, 0);
    assert!(results[0].1.abs() < 1e-6);

    // Other two should have squared distance 25.0 (3² + 4² = 9 + 16 = 25)
    assert!((results[1].1 - 25.0).abs() < 1e-6);
    assert!((results[2].1 - 25.0).abs() < 1e-6);
}

#[test]
fn test_single_dimension() {
    let mut row_vecs = RowAlignedVecs::new(1);

    let vectors = vec![vec![1.0], vec![2.0], vec![-1.0], vec![0.5]];
    row_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0];
    let results = row_vecs.search_vec_cosine(&query).take_max(4);

    // Should work correctly with 1D vectors
    assert_eq!(results.len(), 4);
}

#[test]
fn test_high_dimensional() {
    let dim = 100;
    let mut row_vecs = RowAlignedVecs::new(dim);

    // Create a few high-dimensional vectors
    let mut vec1 = vec![0.0; dim];
    vec1[0] = 1.0;

    let mut vec2 = vec![0.0; dim];
    vec2[1] = 1.0;

    let mut vec3 = vec![0.0; dim];
    vec3[0] = 0.7;
    vec3[1] = 0.7;

    let vectors = vec![vec1.clone(), vec2, vec3];
    row_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec1; // Query with first dimension = 1, rest = 0
    let results = row_vecs.search_vec_cosine(&query).take_max(3);

    // Should handle high dimensions correctly
    assert_eq!(results.len(), 3);

    // First result should be identical vector
    assert_eq!(results[0].0, 0);
    assert!((results[0].1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_take_zero_results() {
    let mut row_vecs = RowAlignedVecs::new(2);

    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    row_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0];
    let results = row_vecs.search_vec_cosine(&query).take_max(0);

    // Should return empty results when k=0
    assert_eq!(results.len(), 0);
}

#[test]
fn test_mixed_magnitude_euclidean() {
    let mut row_vecs = RowAlignedVecs::new(2);

    let vectors = vec![
        vec![1.0, 1.0],     // Squared distance 2.0 from origin
        vec![100.0, 100.0], // Squared distance 20000.0 from origin
        vec![0.1, 0.1],     // Squared distance 0.02 from origin
    ];

    row_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![0.0, 0.0];
    let results = row_vecs.search_vec_euclidean(&query).take_min(3);

    // Should correctly order by squared distance regardless of magnitude
    assert_eq!(results[0].0, 2); // Smallest vector closest
    assert!(results[0].1 < results[1].1);
    assert!(results[1].1 < results[2].1);
}
