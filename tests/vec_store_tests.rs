// These tests have been AI generated to cover the VecStore functionality in the otters library.
use otters::vec::{Cmp, Metric, VecQueryPlan, VecStore};

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
// Basic VecStore Tests
// ============================================================================

#[test]
fn test_vecstore_creation() {
    let mut store = VecStore::new(3);
    assert_eq!(store.add_vector(vec![1.0, 2.0, 3.0]), Ok(()));

    // Test dimension mismatch
    assert!(store.add_vector(vec![1.0, 2.0]).is_err());
}

#[test]
fn test_vecstore_add_vectors() {
    let mut store = VecStore::new(3);
    let vectors = create_test_vectors();

    assert!(store.add_vectors(vectors).is_ok());
}

#[test]
fn test_query_plan_creation() {
    let store = VecStore::new(3);

    // Test single vector query
    let single_query = vec![1.0, 0.0, 0.0];
    let query_plan = store.query(single_query, Metric::Cosine);
    // Should be able to collect successfully
    assert!(query_plan.collect().is_ok());

    // Test multiple vectors query
    let multi_query = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let query_plan = store.query(multi_query, Metric::Cosine);
    // Should be able to collect successfully
    assert!(query_plan.collect().is_ok());
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_dimension_mismatch_error_handling() {
    let mut store = VecStore::new(3);
    store.add_vector(vec![1.0, 0.0, 0.0]).unwrap();

    // Query with wrong dimension should store error and fail at collect
    let query = vec![1.0, 0.0]; // 2D query for 3D store
    let result = store.query(query, Metric::Cosine).take(5).collect();

    assert!(result.is_err());
    let error_msg = result.unwrap_err();
    assert!(error_msg.contains("Query vector length 2 does not match expected dimension 3"));
}

#[test]
fn test_empty_query_batch_error_handling() {
    let store = VecStore::new(3);

    // Empty query batch should store error and fail at collect
    let queries: Vec<Vec<f32>> = vec![];
    let result = store.query(queries, Metric::Cosine).take(5).collect();

    assert!(result.is_err());
    let error_msg = result.unwrap_err();
    assert_eq!(error_msg, "No queries provided");
}

#[test]
fn test_error_propagation_through_chain() {
    let store = VecStore::new(3);

    // Query with wrong dimension
    let query = vec![1.0, 0.0]; // Wrong dimension
    let result = store
        .query(query, Metric::Cosine)
        .filter(0.5, Cmp::Gt) // This should not execute due to error
        .take(5) // This should not execute due to error
        .take_closest(3) // This should not execute due to error
        .collect();

    assert!(result.is_err());
    let error_msg = result.unwrap_err();
    assert!(error_msg.contains("Query vector length 2 does not match expected dimension 3"));
}

#[test]
fn test_successful_chain_after_valid_query() {
    let mut store = VecStore::new(2);

    // Add test vectors
    store.add_vector(vec![1.0, 0.0]).unwrap();
    store.add_vector(vec![0.8, 0.6]).unwrap();
    store.add_vector(vec![0.0, 1.0]).unwrap();

    // Valid query should work through entire chain
    let query = vec![1.0, 0.0];
    let result = store
        .query(query, Metric::Cosine)
        .filter(0.5, Cmp::Gt)
        .take(5)
        .collect();

    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 1); // One query

    // Should only include vectors with similarity > 0.5
    for (_, similarity) in &results[0] {
        assert!(*similarity > 0.5);
    }
}

#[test]
fn test_mixed_dimension_batch_error() {
    let mut store = VecStore::new(3);
    store.add_vector(vec![1.0, 0.0, 0.0]).unwrap();

    // Batch with mixed dimensions should fail
    let queries = vec![
        vec![1.0, 0.0, 0.0], // Valid 3D
        vec![1.0, 0.0],      // Invalid 2D
        vec![1.0, 0.0, 0.0], // Valid 3D
    ];

    let result = store.query(queries, Metric::Cosine).take(5).collect();

    assert!(result.is_err());
    let error_msg = result.unwrap_err();
    assert!(error_msg.contains("Query vector length 2 does not match expected dimension 3"));
}

// ============================================================================
// Cosine Similarity Tests
// ============================================================================

#[test]
fn test_cosine_similarity_basic() {
    let mut store = VecStore::new(3);
    let test_vectors = create_test_vectors();
    store.add_vectors(test_vectors.clone()).unwrap();

    // Query with the first vector (should have similarity 1.0 with itself)
    let query = vec![1.0, 0.0, 0.0];
    let results = store
        .query(query, Metric::Cosine)
        .take(5)
        .collect()
        .unwrap();

    // Should return results for the single query
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 5); // All 5 vectors

    // Check that the similarity with itself is approximately 1.0
    let self_similarity = results[0].iter().find(|(idx, _)| *idx == 0).unwrap();
    assert!((self_similarity.1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_orthogonal_vectors() {
    let mut store = VecStore::new(2);

    // Add orthogonal vectors
    store.add_vector(vec![1.0, 0.0]).unwrap();
    store.add_vector(vec![0.0, 1.0]).unwrap();

    let query = vec![1.0, 0.0];
    let results = store
        .query(query, Metric::Cosine)
        .take(2)
        .collect()
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 2);

    // Find the parallel and orthogonal vectors
    let parallel = results[0].iter().find(|(idx, _)| *idx == 0).unwrap();
    let orthogonal = results[0].iter().find(|(idx, _)| *idx == 1).unwrap();

    // Parallel vector should have similarity 1.0
    assert!((parallel.1 - 1.0).abs() < 1e-6);

    // Orthogonal vector should have similarity 0.0
    assert!(orthogonal.1.abs() < 1e-6);
}

// ============================================================================
// Euclidean Distance Tests
// ============================================================================

#[test]
fn test_euclidean_distance_basic() {
    let mut store = VecStore::new(3);
    let test_vectors = create_test_vectors();
    store.add_vectors(test_vectors.clone()).unwrap();

    // Query with the first vector (should have distance 0.0 with itself)
    let query = vec![1.0, 0.0, 0.0];
    let results = store
        .query(query, Metric::Euclidean)
        .take_closest(5)
        .collect()
        .unwrap();

    // Should return results for the single query
    assert_eq!(results.len(), 1);

    // Check that the distance with itself is approximately 0.0
    let self_distance = results[0].iter().find(|(idx, _)| *idx == 0).unwrap();
    assert!(self_distance.1.abs() < 1e-6);
}

// ============================================================================
// Top-K Selection Tests
// ============================================================================

#[test]
fn test_top_k_cosine() {
    let mut store = VecStore::new(2);

    // Add vectors with known similarities to query [1.0, 0.0]
    store.add_vector(vec![1.0, 0.0]).unwrap(); // identical (index 0)
    store.add_vector(vec![0.8, 0.6]).unwrap(); // high similarity (index 1) 
    store.add_vector(vec![0.0, 1.0]).unwrap(); // orthogonal (index 2)
    store.add_vector(vec![-1.0, 0.0]).unwrap(); // opposite (index 3)

    let query = vec![1.0, 0.0];

    // Test cosine similarity top-2
    let results = store
        .query(query, Metric::Cosine)
        .take(2)
        .collect()
        .unwrap();

    assert_eq!(results.len(), 1); // Single query
    assert_eq!(results[0].len(), 2); // Top 2 results

    // Results should be sorted by similarity (highest first for cosine)
    assert!(results[0][0].1 >= results[0][1].1);
}

#[test]
fn test_top_k_euclidean() {
    let mut store = VecStore::new(2);

    // Add vectors with known distances to query [1.0, 0.0]
    store.add_vector(vec![1.0, 0.0]).unwrap(); // identical (index 0)
    store.add_vector(vec![1.1, 0.0]).unwrap(); // close (index 1)
    store.add_vector(vec![0.0, 1.0]).unwrap(); // distance sqrt(2) (index 2)
    store.add_vector(vec![-1.0, 0.0]).unwrap(); // distance 2 (index 3)

    let query = vec![1.0, 0.0];

    // Test euclidean distance closest-2
    let results = store
        .query(query, Metric::Euclidean)
        .take_closest(2)
        .collect()
        .unwrap();

    assert_eq!(results.len(), 1); // Single query
    assert_eq!(results[0].len(), 2); // Top 2 closest results

    // Results should be sorted by distance (smallest first for closest)
    assert!(results[0][0].1 <= results[0][1].1);
}

#[test]
fn test_take_more_than_available() {
    let mut store = VecStore::new(2);

    store.add_vector(vec![1.0, 0.0]).unwrap();
    store.add_vector(vec![0.0, 1.0]).unwrap();

    let query = vec![1.0, 0.0];

    // Ask for more results than available
    let results = store
        .query(query, Metric::Cosine)
        .take(10)
        .collect()
        .unwrap();

    // Should return all available results (2)
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 2);
}

#[test]
fn test_take_zero_results() {
    let mut store = VecStore::new(2);

    store.add_vector(vec![1.0, 0.0]).unwrap();
    store.add_vector(vec![0.0, 1.0]).unwrap();

    let query = vec![1.0, 0.0];
    let results = store
        .query(query, Metric::Cosine)
        .take(0)
        .collect()
        .unwrap();

    // Should return empty results when k=0
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 0);
}

// ============================================================================
// Filtering Tests
// ============================================================================

#[test]
fn test_filtering() {
    let mut store = VecStore::new(2);

    // Add vectors with known similarities
    store.add_vector(vec![1.0, 0.0]).unwrap(); // similarity 1.0
    store.add_vector(vec![0.8, 0.6]).unwrap(); // similarity 0.8
    store.add_vector(vec![0.0, 1.0]).unwrap(); // similarity 0.0
    store.add_vector(vec![-1.0, 0.0]).unwrap(); // similarity -1.0

    let query = vec![1.0, 0.0];

    // Filter for similarities > 0.5
    let results = store
        .query(query, Metric::Cosine)
        .filter(0.5, Cmp::Gt)
        .take(10)
        .collect()
        .unwrap();

    assert_eq!(results.len(), 1); // Single query

    // Should only return vectors with similarity > 0.5
    for (_, similarity) in &results[0] {
        assert!(*similarity > 0.5);
    }
}

// ============================================================================
// Batch Query Tests
// ============================================================================

#[test]
fn test_batch_queries() {
    let mut store = VecStore::new(3);
    let test_vectors = create_test_vectors();
    store.add_vectors(test_vectors).unwrap();

    let queries = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    let results = store
        .query(queries, Metric::Cosine)
        .take(3)
        .collect()
        .unwrap();

    // Should return results for each query
    assert_eq!(results.len(), 3);

    // Each query should return 3 results
    for query_results in &results {
        assert_eq!(query_results.len(), 3);
    }
}

#[test]
fn test_global_vs_local_scope() {
    let mut store = VecStore::new(2);

    // Add test vectors
    store.add_vector(vec![1.0, 0.0]).unwrap();
    store.add_vector(vec![0.8, 0.6]).unwrap();
    store.add_vector(vec![0.0, 1.0]).unwrap();

    let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    // Test local scope (results per query)
    let local_results = store
        .query(queries.clone(), Metric::Cosine)
        .take(2)
        .collect()
        .unwrap();

    assert_eq!(local_results.len(), 2); // One result per query

    // Test global scope (merged results)
    let global_results = store
        .query(queries, Metric::Cosine)
        .take_global(3)
        .collect()
        .unwrap();

    assert_eq!(global_results.len(), 1); // Single merged result
    assert_eq!(global_results[0].len(), 3); // Top 3 globally
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_empty_store() {
    let store = VecStore::new(3);
    let query = vec![1.0, 0.0, 0.0];

    let results = store
        .query(query, Metric::Cosine)
        .take(5)
        .collect()
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 0); // No vectors in store
}

// ============================================================================
// SIMD Function Tests
// ============================================================================

#[test]
fn test_dot_product() {
    use otters::vec::dot_product;

    let vec1 = vec![1.0, 2.0, 3.0, 4.0];
    let vec2 = vec![2.0, 3.0, 4.0, 5.0];

    let result = dot_product(&vec1, &vec2);
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    assert_eq!(result, 40.0);
}

#[test]
fn test_euclidean_distance_squared() {
    use otters::vec::euclidean_distance_squared;

    let vec1 = vec![1.0, 2.0];
    let vec2 = vec![4.0, 6.0];

    let result = euclidean_distance_squared(&vec1, &vec2);
    // (1-4)² + (2-6)² = 9 + 16 = 25
    assert_eq!(result, 25.0);
}

#[test]
fn test_cosine_similarity() {
    use otters::vec::cosine_similarity;

    let vec1 = vec![1.0, 0.0];
    let vec2 = vec![1.0, 0.0];

    let result = cosine_similarity(&vec1, &vec2, 1.0, 1.0);
    assert!((result - 1.0).abs() < 1e-6);
}

// ============================================================================
// Mathematical Correctness Tests
// ============================================================================

#[test]
fn test_cosine_similarity_correctness() {
    let mut store = VecStore::new(2);

    // Add vectors with known cosine similarities
    store.add_vector(vec![1.0, 0.0]).unwrap(); // parallel: similarity = 1.0
    store.add_vector(vec![-1.0, 0.0]).unwrap(); // anti-parallel: similarity = -1.0
    store.add_vector(vec![0.0, 1.0]).unwrap(); // orthogonal: similarity = 0.0
    store.add_vector(vec![1.0, 1.0]).unwrap(); // 45°: similarity = cos(45°) = √2/2 ≈ 0.707

    let query = vec![1.0, 0.0];

    // Test with no default filter - should include all similarities including negative ones
    let results = store
        .query(query, Metric::Cosine)
        .take(4)
        .collect()
        .unwrap();

    assert_eq!(results[0].len(), 4, "Should have all 4 vectors");

    // Find specific vectors by checking their expected similarity values
    let mut found_parallel = false;
    let mut found_anti_parallel = false;
    let mut found_orthogonal = false;
    let mut found_45deg = false;

    for (idx, sim) in &results[0] {
        if (sim - 1.0).abs() < 1e-6 {
            found_parallel = true;
            assert_eq!(*idx, 0, "Parallel vector should be at index 0, got {}", idx);
        } else if (sim - (-1.0)).abs() < 1e-6 {
            found_anti_parallel = true;
            assert_eq!(
                *idx, 1,
                "Anti-parallel vector should be at index 1, got {}",
                idx
            );
        } else if sim.abs() < 1e-6 {
            found_orthogonal = true;
            assert_eq!(
                *idx, 2,
                "Orthogonal vector should be at index 2, got {}",
                idx
            );
        } else if (sim - (1.0 / 2.0_f32.sqrt())).abs() < 1e-5 {
            found_45deg = true;
            assert_eq!(*idx, 3, "45° vector should be at index 3, got {}", idx);
        }
    }

    assert!(
        found_parallel,
        "Should find parallel vector with similarity 1.0"
    );
    assert!(
        found_anti_parallel,
        "Should find anti-parallel vector with similarity -1.0"
    );
    assert!(
        found_orthogonal,
        "Should find orthogonal vector with similarity 0.0"
    );
    assert!(
        found_45deg,
        "Should find 45° vector with similarity ≈ 0.707"
    );
}

#[test]
fn test_euclidean_distance_correctness() {
    let mut store = VecStore::new(2);

    // Add vectors with known euclidean distances from origin [0,0]
    store.add_vector(vec![0.0, 0.0]).unwrap(); // distance = 0
    store.add_vector(vec![3.0, 4.0]).unwrap(); // distance = 5 (3-4-5 triangle)
    store.add_vector(vec![1.0, 1.0]).unwrap(); // distance = √2 ≈ 1.414
    store.add_vector(vec![0.0, 5.0]).unwrap(); // distance = 5
    store.add_vector(vec![-3.0, -4.0]).unwrap(); // distance = 5

    let query = vec![0.0, 0.0];
    let results = store
        .query(query, Metric::Euclidean)
        .take_closest(5)
        .collect()
        .unwrap();

    // Find specific results by index
    let identical = results[0].iter().find(|(idx, _)| *idx == 0).unwrap();
    let triangle = results[0].iter().find(|(idx, _)| *idx == 1).unwrap();
    let diagonal = results[0].iter().find(|(idx, _)| *idx == 2).unwrap();
    let vertical = results[0].iter().find(|(idx, _)| *idx == 3).unwrap();
    let negative = results[0].iter().find(|(idx, _)| *idx == 4).unwrap();

    // Verify mathematical correctness (squared distances)
    assert!(
        identical.1.abs() < 1e-6,
        "Identical vectors should have distance 0"
    );
    assert!(
        (triangle.1 - 25.0).abs() < 1e-6,
        "3-4-5 triangle should have squared distance 25"
    );
    assert!(
        (diagonal.1 - 2.0).abs() < 1e-6,
        "Unit diagonal should have squared distance 2"
    );
    assert!(
        (vertical.1 - 25.0).abs() < 1e-6,
        "Vertical distance 5 should have squared distance 25"
    );
    assert!(
        (negative.1 - 25.0).abs() < 1e-6,
        "Negative coordinates should have squared distance 25"
    );
}

#[test]
fn test_top_k_ranking_correctness() {
    let mut store = VecStore::new(2);

    // Add vectors with known similarities to [1.0, 0.0]
    store.add_vector(vec![1.0, 0.0]).unwrap(); // similarity = 1.0 (rank 1)
    store.add_vector(vec![0.8, 0.6]).unwrap(); // similarity = 0.8 (rank 2)
    store.add_vector(vec![0.6, 0.8]).unwrap(); // similarity = 0.6 (rank 3) 
    store.add_vector(vec![0.0, 1.0]).unwrap(); // similarity = 0.0 (rank 4)

    let query = vec![1.0, 0.0];
    let results = store
        .query(query, Metric::Cosine)
        .take(4)
        .collect()
        .unwrap();

    // Results should be in descending order of similarity
    let similarities: Vec<f32> = results[0].iter().map(|(_, sim)| *sim).collect();

    // Verify correct ranking
    assert!(
        (similarities[0] - 1.0).abs() < 1e-6,
        "Highest similarity should be 1.0, got {}",
        similarities[0]
    );
    assert!(
        (similarities[1] - 0.8).abs() < 1e-6,
        "Second highest should be 0.8, got {}",
        similarities[1]
    );
    assert!(
        (similarities[2] - 0.6).abs() < 1e-6,
        "Third highest should be 0.6, got {}",
        similarities[2]
    );
    assert!(
        similarities[3].abs() < 1e-6,
        "Fourth should be 0.0, got {}",
        similarities[3]
    );

    // Verify descending order
    for i in 0..similarities.len() - 1 {
        assert!(
            similarities[i] >= similarities[i + 1],
            "Similarities should be in descending order: {} >= {}",
            similarities[i],
            similarities[i + 1]
        );
    }
}

#[test]
fn test_euclidean_ranking_correctness() {
    let mut store = VecStore::new(2);

    // Add vectors with known distances to [0.0, 0.0]
    store.add_vector(vec![0.0, 0.0]).unwrap(); // distance² = 0 (rank 1)
    store.add_vector(vec![1.0, 0.0]).unwrap(); // distance² = 1 (rank 2)
    store.add_vector(vec![0.0, 1.0]).unwrap(); // distance² = 1 (rank 2)
    store.add_vector(vec![1.0, 1.0]).unwrap(); // distance² = 2 (rank 4)
    store.add_vector(vec![2.0, 0.0]).unwrap(); // distance² = 4 (rank 5)
    store.add_vector(vec![3.0, 4.0]).unwrap(); // distance² = 25 (rank 6)

    let query = vec![0.0, 0.0];
    let results = store
        .query(query, Metric::Euclidean)
        .take_closest(6)
        .collect()
        .unwrap();

    // Results should be in ascending order of distance
    let distances: Vec<f32> = results[0].iter().map(|(_, dist)| *dist).collect();

    // Verify correct ranking (squared distances)
    assert!(distances[0].abs() < 1e-6, "Shortest distance should be 0");
    assert!(
        (distances[1] - 1.0).abs() < 1e-6 || (distances[2] - 1.0).abs() < 1e-6,
        "Next shortest should be 1"
    );
    assert!(
        (distances[2] - 1.0).abs() < 1e-6 || (distances[1] - 1.0).abs() < 1e-6,
        "Next shortest should be 1"
    );
    assert!(
        (distances[3] - 2.0).abs() < 1e-6,
        "Fourth shortest should be 2"
    );
    assert!(
        (distances[4] - 4.0).abs() < 1e-6,
        "Fifth shortest should be 4"
    );
    assert!((distances[5] - 25.0).abs() < 1e-6, "Longest should be 25");

    // Verify ascending order
    for i in 0..distances.len() - 1 {
        assert!(
            distances[i] <= distances[i + 1],
            "Distances should be in ascending order: {} <= {}",
            distances[i],
            distances[i + 1]
        );
    }
}

#[test]
fn test_filter_threshold_correctness() {
    let mut store = VecStore::new(2);

    // Add vectors with precise known similarities
    store.add_vector(vec![1.0, 0.0]).unwrap(); // similarity = 1.0
    store.add_vector(vec![0.8, 0.6]).unwrap(); // similarity = 0.8  
    store.add_vector(vec![0.6, 0.8]).unwrap(); // similarity = 0.6
    store.add_vector(vec![0.0, 1.0]).unwrap(); // similarity = 0.0
    store.add_vector(vec![-0.6, 0.8]).unwrap(); // similarity = -0.6

    let query = vec![1.0, 0.0];

    // Test various filter thresholds
    let above_07 = store
        .query(query.clone(), Metric::Cosine)
        .filter(0.7, Cmp::Gt)
        .take(10)
        .collect()
        .unwrap();

    // Should only include vectors with similarity > 0.7 (1.0 and 0.8)
    assert_eq!(above_07[0].len(), 2);
    for (_, sim) in &above_07[0] {
        assert!(
            *sim > 0.7,
            "All results should have similarity > 0.7, got {}",
            sim
        );
    }

    let above_equal_06 = store
        .query(query.clone(), Metric::Cosine)
        .filter(0.6, Cmp::Geq)
        .take(10)
        .collect()
        .unwrap();

    // Should include vectors with similarity >= 0.6 (1.0, 0.8, 0.6)
    assert_eq!(above_equal_06[0].len(), 3);
    for (_, sim) in &above_equal_06[0] {
        assert!(
            *sim >= 0.6,
            "All results should have similarity >= 0.6, got {}",
            sim
        );
    }

    let below_05 = store
        .query(query, Metric::Cosine)
        .filter(0.5, Cmp::Lt)
        .take(10)
        .collect()
        .unwrap();

    // Should include vectors with similarity < 0.5 (0.0, -0.6)
    assert_eq!(below_05[0].len(), 2);
    for (_, sim) in &below_05[0] {
        assert!(
            *sim < 0.5,
            "All results should have similarity < 0.5, got {}",
            sim
        );
    }
}

#[test]
fn test_batch_query_correctness() {
    let mut store = VecStore::new(2);

    // Add test vectors
    store.add_vector(vec![1.0, 0.0]).unwrap(); // index 0
    store.add_vector(vec![0.0, 1.0]).unwrap(); // index 1
    store.add_vector(vec![-1.0, 0.0]).unwrap(); // index 2

    let queries = vec![
        vec![1.0, 0.0], // Should find index 0 as most similar
        vec![0.0, 1.0], // Should find index 1 as most similar
    ];

    let results = store
        .query(queries, Metric::Cosine)
        .take(1) // Only get the most similar for each
        .collect()
        .unwrap();

    assert_eq!(results.len(), 2);

    // First query should find vector at index 0 with similarity 1.0
    assert_eq!(results[0].len(), 1);
    assert_eq!(results[0][0].0, 0); // index 0
    assert!((results[0][0].1 - 1.0).abs() < 1e-6); // similarity 1.0

    // Second query should find vector at index 1 with similarity 1.0
    assert_eq!(results[1].len(), 1);
    assert_eq!(results[1][0].0, 1); // index 1
    assert!((results[1][0].1 - 1.0).abs() < 1e-6); // similarity 1.0
}

// ============================================================================
// API Design Tests (showcasing the new lazy evaluation)
// ============================================================================

#[test]
fn test_api_design_showcase() -> Result<(), String> {
    let mut store = VecStore::new(3);

    // Add some test data
    for i in 0..100 {
        let vector = vec![
            (i as f32) / 100.0,
            ((i * 2) as f32) / 100.0,
            ((i * 3) as f32) / 100.0,
        ];
        store.add_vector(vector).map_err(|e| e.to_string())?;
    }

    let query = vec![0.5, 0.5, 0.5];

    // Demonstrate the clean API without unwraps in the chain
    let results = store
        .query(query, Metric::Cosine)
        .filter(0.8, Cmp::Gt) // Only high similarities
        .take_closest(10) // Get 10 closest
        .collect()?; // Only error handling point

    assert_eq!(results.len(), 1); // Single query

    // All results should have similarity > 0.8
    for (_, similarity) in &results[0] {
        assert!(*similarity > 0.8);
    }

    Ok(())
}

#[test]
fn test_error_in_chain_stops_execution() {
    let store = VecStore::new(3);

    // This should fail immediately when query is processed, but chain continues building
    let query = vec![1.0, 0.0]; // Wrong dimension

    // The entire chain should build successfully
    let plan = store
        .query(query, Metric::Cosine)
        .filter(0.5, Cmp::Gt)
        .take(10)
        .take_closest(5); // This overrides the previous take

    // Error only surfaces when we try to collect
    let result = plan.collect();
    assert!(result.is_err());

    let error_msg = result.unwrap_err();
    assert!(error_msg.contains("Query vector length 2 does not match expected dimension 3"));
}

// ============================================================================
// Tests for uncovered functionality
// ============================================================================

#[test]
fn test_vec_query_plan_new() {
    let plan = VecQueryPlan::new();
    let result = plan.collect();
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .contains("Query vectors or their norms are not set")
    );
}

#[test]
fn test_error_propagation_in_filter() {
    // Start with a fresh plan that has an error state
    let plan = VecQueryPlan::new();
    // Add an error by trying to filter without proper setup
    let plan_with_error = plan.filter(0.5, Cmp::Gt);

    // The error should still be present and prevent further operations
    let result = plan_with_error.collect();
    assert!(result.is_err());
}

#[test]
fn test_error_propagation_in_take_methods() {
    // Test all take methods propagate errors correctly when called on invalid plans
    let plan = VecQueryPlan::new();
    let plan1 = plan.take(5);
    assert!(plan1.collect().is_err());

    let plan = VecQueryPlan::new();
    let plan2 = plan.take_global(5);
    assert!(plan2.collect().is_err());

    let plan = VecQueryPlan::new();
    let plan3 = plan.take_closest(5);
    assert!(plan3.collect().is_err());

    let plan = VecQueryPlan::new();
    let plan4 = plan.take_closest_global(5);
    assert!(plan4.collect().is_err());

    let plan = VecQueryPlan::new();
    let plan5 = plan.take_farthest(5);
    assert!(plan5.collect().is_err());

    let plan = VecQueryPlan::new();
    let plan6 = plan.take_farthest_global(5);
    assert!(plan6.collect().is_err());
}

#[test]
fn test_empty_query_vectors_in_batch() {
    let store = VecStore::new(3);
    let empty_queries: Vec<Vec<f32>> = vec![];
    let plan = store.query(empty_queries, Metric::Cosine);

    let result = plan.collect();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("No queries provided"));
}

#[test]
fn test_filter_with_all_comparison_operators() {
    let mut store = VecStore::new(2);
    store
        .add_vectors(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
            vec![0.8, 0.6],
        ])
        .unwrap();

    let query = vec![1.0, 0.0];

    // Test Lt filter
    let results = store
        .query(query.clone(), Metric::Cosine)
        .filter(0.9, Cmp::Lt)
        .take(10)
        .collect()
        .unwrap();
    assert!(!results[0].is_empty());

    // Test Gt filter
    let results = store
        .query(query.clone(), Metric::Cosine)
        .filter(0.1, Cmp::Gt)
        .take(10)
        .collect()
        .unwrap();
    assert!(!results[0].is_empty());

    // Test Leq filter
    let results = store
        .query(query.clone(), Metric::Cosine)
        .filter(1.0, Cmp::Leq)
        .take(10)
        .collect()
        .unwrap();
    assert!(!results[0].is_empty());

    // Test Geq filter
    let results = store
        .query(query.clone(), Metric::Cosine)
        .filter(0.0, Cmp::Geq)
        .take(10)
        .collect()
        .unwrap();
    assert!(!results[0].is_empty());

    // Test Eq filter (this should be rare but needs coverage)
    let results = store
        .query(query, Metric::Cosine)
        .filter(1.0, Cmp::Eq)
        .take(10)
        .collect()
        .unwrap();
    // Should find the exact match (first vector)
    assert!(!results[0].is_empty());
}

#[test]
fn test_add_vector_with_zero_norm() {
    let mut store = VecStore::new(3);

    // Add a zero vector (this will create an infinite inverse norm)
    let result = store.add_vector(vec![0.0, 0.0, 0.0]);
    // This should succeed but create inf inverse norm
    assert!(result.is_ok());

    // Query should still work (though results might be weird with inf values)
    let results = store
        .query(vec![1.0, 0.0, 0.0], Metric::Cosine)
        .take(1)
        .collect();
    assert!(results.is_ok());
}

#[test]
fn test_query_with_zero_norm_query_vector() {
    let mut store = VecStore::new(3);
    store.add_vector(vec![1.0, 0.0, 0.0]).unwrap();

    // Query with zero vector (will create inf inverse norm)
    let results = store
        .query(vec![0.0, 0.0, 0.0], Metric::Cosine)
        .take(1)
        .collect();
    // This should succeed but may have inf/nan values
    assert!(results.is_ok());
}

#[test]
fn test_global_scope_functionality() {
    let mut store = VecStore::new(2);
    store
        .add_vectors(vec![
            vec![1.0, 0.0], // index 0
            vec![0.0, 1.0], // index 1
            vec![0.8, 0.6], // index 2
            vec![0.6, 0.8], // index 3
        ])
        .unwrap();

    // Test multiple queries with global scope
    let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    let results = store
        .query(queries, Metric::Cosine)
        .take_global(2)
        .collect()
        .unwrap();

    // With global scope, we should get a single result vector containing
    // the top 2 results across all queries
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].len(), 2);
}

#[test]
fn test_filter_and_merge_with_no_filtering() {
    let mut store = VecStore::new(2);
    store
        .add_vectors(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]])
        .unwrap();

    // Query without any filter - should cover the None case in filter_and_merge_results
    let results = store
        .query(vec![1.0, 0.0], Metric::Cosine)
        .take(2)
        .collect()
        .unwrap();

    assert_eq!(results[0].len(), 2);
}

#[test]
fn test_dimension_mismatch_during_add_vectors() {
    let mut store = VecStore::new(3);

    let vectors = vec![
        vec![1.0, 0.0, 0.0], // correct dimension
        vec![1.0, 0.0],      // wrong dimension
    ];

    let result = store.add_vectors(vectors);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .contains("Input vector length 2 does not match expected dimension 3")
    );
}

#[test]
fn test_take_closest_and_farthest_methods() {
    let mut store = VecStore::new(2);
    store
        .add_vectors(vec![
            vec![1.0, 0.0], // Close to query
            vec![0.0, 1.0], // Far from query
            vec![0.9, 0.1], // Very close to query
        ])
        .unwrap();

    let query = vec![1.0, 0.0];

    // Test take_closest - should prioritize smaller distances
    let results = store
        .query(query.clone(), Metric::Euclidean)
        .take_closest(2)
        .collect()
        .unwrap();
    assert_eq!(results[0].len(), 2);

    // Test take_farthest - should prioritize larger distances
    let results = store
        .query(query.clone(), Metric::Euclidean)
        .take_farthest(2)
        .collect()
        .unwrap();
    assert_eq!(results[0].len(), 2);

    // Test take_closest_global
    let queries = vec![query.clone(), vec![0.0, 1.0]];
    let results = store
        .query(queries, Metric::Euclidean)
        .take_closest_global(1)
        .collect()
        .unwrap();
    assert_eq!(results.len(), 1); // Global scope
    assert_eq!(results[0].len(), 1);

    // Test take_farthest_global
    let queries = vec![query, vec![0.0, 1.0]];
    let results = store
        .query(queries, Metric::Euclidean)
        .take_farthest_global(1)
        .collect()
        .unwrap();
    assert_eq!(results.len(), 1); // Global scope
    assert_eq!(results[0].len(), 1);
}

#[test]
fn test_query_batch_conversions() {
    let mut store = VecStore::new(3);
    store.add_vector(vec![1.0, 0.0, 0.0]).unwrap();

    // Test From<Vec<f32>> implementation by passing single vector
    let single_query = vec![1.0, 0.0, 0.0];
    let results = store
        .query(single_query, Metric::Cosine)
        .take(1)
        .collect()
        .unwrap();
    assert_eq!(results.len(), 1);

    // Test From<Vec<Vec<f32>>> implementation by passing multiple vectors
    let multi_queries = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let results = store
        .query(multi_queries, Metric::Cosine)
        .take(1)
        .collect()
        .unwrap();
    assert_eq!(results.len(), 2); // Two queries, so two result sets
}

#[test]
fn test_error_states_in_chained_operations() {
    let mut store = VecStore::new(3);
    store.add_vector(vec![1.0, 0.0, 0.0]).unwrap();

    // Create a plan that will have an error (mismatched dimensions)
    let query = vec![1.0, 0.0]; // Wrong dimension
    let plan = store.query(query, Metric::Cosine);

    // Chain operations - they should all propagate the error
    let final_plan = plan
        .filter(0.5, Cmp::Gt)
        .take(5)
        .take_global(3)
        .take_closest(2)
        .take_farthest(1);

    let result = final_plan.collect();
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .contains("does not match expected dimension")
    );
}

#[test]
fn test_filtering_edge_cases() {
    let mut store = VecStore::new(2);
    store
        .add_vectors(vec![
            vec![1.0, 0.0],  // cos sim = 1.0 with [1,0] query
            vec![0.0, 1.0],  // cos sim = 0.0 with [1,0] query
            vec![-1.0, 0.0], // cos sim = -1.0 with [1,0] query
        ])
        .unwrap();

    let query = vec![1.0, 0.0];

    // Test filtering that results in empty results
    let results = store
        .query(query.clone(), Metric::Cosine)
        .filter(1.5, Cmp::Gt) // Nothing should be > 1.5
        .take(10)
        .collect()
        .unwrap();
    assert!(results[0].is_empty());

    // Test filtering with exact equality
    let results = store
        .query(query, Metric::Cosine)
        .filter(1.0, Cmp::Eq) // Should find exactly the [1,0] vector
        .take(10)
        .collect()
        .unwrap();
    assert_eq!(results[0].len(), 1);
}
