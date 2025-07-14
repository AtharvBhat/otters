// This entire file is basically AI generated dont pay too much attention to it
use otters::vec::{ColumnAlignedVecs, RowAlignedVecs};

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

#[test]
fn test_column_aligned_creation() {
    let mut col_vecs = ColumnAlignedVecs::new(3);
    assert_eq!(col_vecs.add_vector(vec![1.0, 2.0, 3.0]), Ok(()));

    // Test dimension mismatch
    assert!(col_vecs.add_vector(vec![1.0, 2.0]).is_err());
}

// ============================================================================
// Algorithm Correctness Tests
// ============================================================================

#[test]
fn test_cosine_similarity_basic() {
    let test_vectors = create_test_vectors();

    let mut row_vecs = RowAlignedVecs::new(3);
    let mut col_vecs = ColumnAlignedVecs::new(3);

    row_vecs.add_vectors(test_vectors.clone()).unwrap();
    col_vecs.add_vectors(test_vectors.clone()).unwrap();

    let query = vec![1.0, 0.0, 0.0];

    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(5);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(5);

    // Results should be the same
    assert_eq!(row_results.len(), col_results.len());

    // First result should be vector [1,0,0] with similarity close to 1.0
    assert_eq!(row_results[0].0, 0); // Index 0
    assert!((row_results[0].1 - 1.0).abs() < 1e-6); // Similarity close to 1.0

    // Results should match between row and column implementations
    for (row_result, col_result) in row_results.iter().zip(col_results.iter()) {
        assert_eq!(row_result.0, col_result.0); // Same index
        assert!((row_result.1 - col_result.1).abs() < 1e-6); // Same similarity
    }
}

#[test]
fn test_euclidean_distance_basic() {
    let test_vectors = create_test_vectors();

    let mut row_vecs = RowAlignedVecs::new(3);
    let mut col_vecs = ColumnAlignedVecs::new(3);

    row_vecs.add_vectors(test_vectors.clone()).unwrap();
    col_vecs.add_vectors(test_vectors.clone()).unwrap();

    let query = vec![1.0, 0.0, 0.0];

    let row_results: Vec<_> = row_vecs.search_vec_euclidean(&query).take_min(5);
    let col_results: Vec<_> = col_vecs.search_vec_euclidean(&query).take_min(5);

    // Results should be the same
    assert_eq!(row_results.len(), col_results.len());

    // First result should be vector [1,0,0] with distance 0.0
    assert_eq!(row_results[0].0, 0); // Index 0
    assert!(row_results[0].1.abs() < 1e-6); // Distance close to 0.0

    // Results should match between row and column implementations
    for (row_result, col_result) in row_results.iter().zip(col_results.iter()) {
        assert_eq!(row_result.0, col_result.0); // Same index
        assert!((row_result.1 - col_result.1).abs() < 1e-6); // Same distance
    }
}

// ============================================================================
// Mathematical Edge Cases
// ============================================================================

#[test]
fn test_orthogonal_vectors_cosine() {
    let mut row_vecs = RowAlignedVecs::new(2);
    let mut col_vecs = ColumnAlignedVecs::new(2);

    // Add orthogonal vectors
    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    row_vecs.add_vectors(vectors.clone()).unwrap();
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0];

    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(2);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(2);

    // First vector should have similarity 1.0 (parallel)
    assert!((row_results[0].1 - 1.0).abs() < 1e-6);
    assert!((col_results[0].1 - 1.0).abs() < 1e-6);

    // Second vector should have similarity 0.0 (orthogonal)
    assert!(row_results[1].1.abs() < 1e-6);
    assert!(col_results[1].1.abs() < 1e-6);
}

// ============================================================================
// Boundary Condition Tests
// ============================================================================

#[test]
fn test_empty_results() {
    let row_vecs = RowAlignedVecs::new(3);
    let col_vecs = ColumnAlignedVecs::new(3);

    let query = vec![1.0, 0.0, 0.0];

    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(5);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(5);

    assert_eq!(row_results.len(), 0);
    assert_eq!(col_results.len(), 0);
}

#[test]
fn test_take_more_than_available() {
    let mut row_vecs = RowAlignedVecs::new(2);
    let mut col_vecs = ColumnAlignedVecs::new(2);

    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    row_vecs.add_vectors(vectors.clone()).unwrap();
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0];

    // Ask for more results than available
    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(10);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(10);

    // Should return all available results (2)
    assert_eq!(row_results.len(), 2);
    assert_eq!(col_results.len(), 2);
}

#[test]
fn test_normalized_vectors() {
    let mut row_vecs = RowAlignedVecs::new(3);
    let mut col_vecs = ColumnAlignedVecs::new(3);

    // Add vectors with different magnitudes
    let vectors = vec![
        vec![2.0, 0.0, 0.0], // magnitude 2
        vec![0.0, 3.0, 0.0], // magnitude 3
        vec![0.0, 0.0, 4.0], // magnitude 4
    ];

    row_vecs.add_vectors(vectors.clone()).unwrap();
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0, 0.0]; // unit vector

    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(3);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(3);

    // First result should be the parallel vector with similarity 1.0
    assert_eq!(row_results[0].0, 0);
    assert!((row_results[0].1 - 1.0).abs() < 1e-6);

    // Other vectors should have similarity 0.0 (orthogonal)
    assert!(row_results[1].1.abs() < 1e-6);
    assert!(row_results[2].1.abs() < 1e-6);

    // Results should match between implementations
    for (row_result, col_result) in row_results.iter().zip(col_results.iter()) {
        assert_eq!(row_result.0, col_result.0);
        assert!((row_result.1 - col_result.1).abs() < 1e-6);
    }
}

#[test]
fn test_zero_vectors() {
    let mut row_vecs = RowAlignedVecs::new(3);
    let mut col_vecs = ColumnAlignedVecs::new(3);

    // Add zero vector
    let vectors = vec![vec![0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];

    row_vecs.add_vectors(vectors.clone()).unwrap();
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0, 0.0];

    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(2);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(2);

    // Zero vector should handle division by zero gracefully
    // Results should still be computed for the non-zero vector
    assert_eq!(row_results.len(), 2);
    assert_eq!(col_results.len(), 2);

    // Non-zero vector should have similarity 1.0
    let non_zero_result = row_results.iter().find(|(idx, _)| *idx == 1).unwrap();
    assert!((non_zero_result.1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_negative_values() {
    let mut row_vecs = RowAlignedVecs::new(3);
    let mut col_vecs = ColumnAlignedVecs::new(3);

    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![-1.0, 0.0, 0.0], // Opposite direction
        vec![0.0, -1.0, 0.0], // Different axis, negative
    ];

    row_vecs.add_vectors(vectors.clone()).unwrap();
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0, 0.0];

    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(3);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(3);

    // First result should be parallel vector (similarity 1.0)
    assert_eq!(row_results[0].0, 0);
    assert!((row_results[0].1 - 1.0).abs() < 1e-6);

    // Second result should be anti-parallel vector (similarity -1.0)
    let anti_parallel = row_results.iter().find(|(idx, _)| *idx == 1).unwrap();
    assert!((anti_parallel.1 - (-1.0)).abs() < 1e-6);

    // Third result should be orthogonal (similarity 0.0)
    let orthogonal = row_results.iter().find(|(idx, _)| *idx == 2).unwrap();
    assert!(orthogonal.1.abs() < 1e-6);

    // Results should match between implementations
    for (row_result, col_result) in row_results.iter().zip(col_results.iter()) {
        assert_eq!(row_result.0, col_result.0);
        assert!((row_result.1 - col_result.1).abs() < 1e-6);
    }
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

#[test]
fn test_very_small_values() {
    let mut row_vecs = RowAlignedVecs::new(3);
    let mut col_vecs = ColumnAlignedVecs::new(3);

    // Test with very small but non-zero values
    let vectors = vec![
        vec![1e-10, 0.0, 0.0],
        vec![0.0, 1e-10, 0.0],
        vec![1e-10, 1e-10, 0.0],
    ];

    row_vecs.add_vectors(vectors.clone()).unwrap();
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1e-10, 0.0, 0.0];

    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(3);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(3);

    // Should handle small values without numerical instability
    assert_eq!(row_results.len(), 3);
    assert_eq!(col_results.len(), 3);

    // Results should match between implementations
    for (row_result, col_result) in row_results.iter().zip(col_results.iter()) {
        assert_eq!(row_result.0, col_result.0);
        assert!((row_result.1 - col_result.1).abs() < 1e-6);
    }
}

#[test]
fn test_large_values() {
    let mut row_vecs = RowAlignedVecs::new(3);
    let mut col_vecs = ColumnAlignedVecs::new(3);

    // Test with large values
    let vectors = vec![
        vec![1e6, 0.0, 0.0],
        vec![0.0, 1e6, 0.0],
        vec![1e6, 1e6, 0.0],
    ];

    row_vecs.add_vectors(vectors.clone()).unwrap();
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1e6, 0.0, 0.0];

    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(3);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(3);

    // Should handle large values without overflow
    assert_eq!(row_results.len(), 3);
    assert_eq!(col_results.len(), 3);

    // First result should be parallel vector (similarity 1.0)
    assert_eq!(row_results[0].0, 0);
    assert!((row_results[0].1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_euclidean_distance_edge_cases() {
    let mut row_vecs = RowAlignedVecs::new(2);
    let mut col_vecs = ColumnAlignedVecs::new(2);

    let vectors = vec![
        vec![0.0, 0.0],   // Same as query
        vec![3.0, 4.0],   // Squared distance 25.0 from origin
        vec![-3.0, -4.0], // Squared distance 25.0 from origin, opposite direction
    ];

    row_vecs.add_vectors(vectors.clone()).unwrap();
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![0.0, 0.0];

    let row_results: Vec<_> = row_vecs.search_vec_euclidean(&query).take_min(3);
    let col_results: Vec<_> = col_vecs.search_vec_euclidean(&query).take_min(3);

    // First result should be identical vector with squared distance 0.0
    assert_eq!(row_results[0].0, 0);
    assert!(row_results[0].1.abs() < 1e-6);

    // Other two should have squared distance 25.0 (3² + 4² = 9 + 16 = 25)
    assert!((row_results[1].1 - 25.0).abs() < 1e-6);
    assert!((row_results[2].1 - 25.0).abs() < 1e-6);

    // Results should match between implementations
    for (row_result, col_result) in row_results.iter().zip(col_results.iter()) {
        assert_eq!(row_result.0, col_result.0);
        assert!((row_result.1 - col_result.1).abs() < 1e-6);
    }
}

#[test]
fn test_single_dimension() {
    let mut row_vecs = RowAlignedVecs::new(1);
    let mut col_vecs = ColumnAlignedVecs::new(1);

    let vectors = vec![vec![1.0], vec![2.0], vec![-1.0], vec![0.5]];

    row_vecs.add_vectors(vectors.clone()).unwrap();
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0];

    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(4);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(4);

    // Should work correctly with 1D vectors
    assert_eq!(row_results.len(), 4);
    assert_eq!(col_results.len(), 4);

    // Results should match between implementations
    for (row_result, col_result) in row_results.iter().zip(col_results.iter()) {
        assert_eq!(row_result.0, col_result.0);
        assert!((row_result.1 - col_result.1).abs() < 1e-6);
    }
}

// ============================================================================
// Performance and Scalability Tests
// ============================================================================

#[test]
fn test_high_dimensional() {
    let dim = 100;
    let mut row_vecs = RowAlignedVecs::new(dim);
    let mut col_vecs = ColumnAlignedVecs::new(dim);

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
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec1; // Query with first dimension = 1, rest = 0

    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(3);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(3);

    // Should handle high dimensions correctly
    assert_eq!(row_results.len(), 3);
    assert_eq!(col_results.len(), 3);

    // First result should be identical vector
    assert_eq!(row_results[0].0, 0);
    assert!((row_results[0].1 - 1.0).abs() < 1e-6);

    // Results should match between implementations
    for (row_result, col_result) in row_results.iter().zip(col_results.iter()) {
        assert_eq!(row_result.0, col_result.0);
        assert!((row_result.1 - col_result.1).abs() < 1e-6);
    }
}

#[test]
fn test_take_zero_results() {
    let mut row_vecs = RowAlignedVecs::new(2);
    let mut col_vecs = ColumnAlignedVecs::new(2);

    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    row_vecs.add_vectors(vectors.clone()).unwrap();
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![1.0, 0.0];

    let row_results: Vec<_> = row_vecs.search_vec_cosine(&query).take_max(0);
    let col_results: Vec<_> = col_vecs.search_vec_cosine(&query).take_max(0);

    // Should return empty results when k=0
    assert_eq!(row_results.len(), 0);
    assert_eq!(col_results.len(), 0);
}

#[test]
fn test_mixed_magnitude_euclidean() {
    let mut row_vecs = RowAlignedVecs::new(2);
    let mut col_vecs = ColumnAlignedVecs::new(2);

    let vectors = vec![
        vec![1.0, 1.0],     // Squared distance 2.0 from origin
        vec![100.0, 100.0], // Squared distance 20000.0 from origin
        vec![0.1, 0.1],     // Squared distance 0.02 from origin
    ];

    row_vecs.add_vectors(vectors.clone()).unwrap();
    col_vecs.add_vectors(vectors.clone()).unwrap();

    let query = vec![0.0, 0.0];

    let row_results: Vec<_> = row_vecs.search_vec_euclidean(&query).take_min(3);
    let col_results: Vec<_> = col_vecs.search_vec_euclidean(&query).take_min(3);

    // Should correctly order by squared distance regardless of magnitude
    assert_eq!(row_results[0].0, 2); // Smallest vector closest
    assert!(row_results[0].1 < row_results[1].1);
    assert!(row_results[1].1 < row_results[2].1);

    // Results should match between implementations
    for (row_result, col_result) in row_results.iter().zip(col_results.iter()) {
        assert_eq!(row_result.0, col_result.0);
        assert!((row_result.1 - col_result.1).abs() < 1e-6);
    }
}
