use otters::expr::col;
use otters::prelude::*;

// Helper to build a standard MetaStore with three chunks (chunk_size = 3)
// val (Int32) distribution:
//   Chunk0: 1, 2, NULL
//   Chunk1: 10, 11, 12
//   Chunk2: NULL, NULL, NULL
// ts (DateTime) distribution:
//   Chunk0: 2024-01-01T00:00:00Z, NULL, 2024-06-01T00:00:00Z
//   Chunk1: 2026-01-01T00:00:00Z, 2026-06-01T00:00:00Z, 2024-12-31T23:59:59Z
//   Chunk2: NULL, NULL, NULL
// grade (String) distribution:
//   Chunk0: "A", "B", NULL
//   Chunk1: "C", "A", "A"
//   Chunk2: NULL, NULL, NULL
fn build_store() -> MetaStore {
    let vectors: Vec<Vec<f32>> = (0..9).map(|_| vec![1.0, 0.0]).collect(); // dummy vectors

    let val = Column::new("val", DataType::Int32)
        .from(vec![
            Some(1),
            Some(2),
            None,
            Some(10),
            Some(11),
            Some(12),
            None,
            None,
            None,
        ])
        .unwrap();

    let ts = Column::new("ts", DataType::DateTime)
        .from(vec![
            Some("2024-01-01T00:00:00Z"),
            None,
            Some("2024-06-01T00:00:00Z"),
            Some("2026-01-01T00:00:00Z"),
            Some("2026-06-01T00:00:00Z"),
            Some("2024-12-31T23:59:59Z"),
            None,
            None,
            None,
        ])
        .unwrap();

    let grade = Column::new("grade", DataType::String)
        .from(vec![
            Some("A"),
            Some("B"),
            None,
            Some("C"),
            Some("A"),
            Some("A"),
            None,
            None,
            None,
        ])
        .unwrap();

    MetaStore::from_columns(vec![val, ts, grade])
        .with_vectors(vectors)
        .with_chunk_size(3)
        .build()
        .unwrap()
}

#[test]
fn zonemap_prunes_numeric_with_nulls() {
    let store = build_store();

    // val > 5 should prune chunk0 (max=2) and chunk2 (all null) leaving only chunk1
    let results = store
        .query(vec![1.0, 0.0], Metric::DotProduct)
        .meta_filter(col("val").gt(5))
        .take(9)
        .collect()
        .unwrap();

    // Expect only rows 3,4,5 (values 10,11,12)
    let indices: std::collections::HashSet<usize> = results.indices.iter().cloned().collect();
    assert_eq!(indices, [3usize, 4usize, 5usize].into_iter().collect());

    let stats = store.last_query_stats().unwrap();
    assert_eq!(stats.total_chunks, 3);
    assert_eq!(stats.evaluated_chunks, 1, "Only middle chunk should remain");
    assert_eq!(stats.pruned_chunks, 2);
}

#[test]
fn zonemap_boundary_conditions() {
    let store = build_store();

    // val >= 2 should keep chunk0 (range 1..2) + chunk1, prune chunk2
    let _ = store
        .query(vec![1.0, 0.0], Metric::Cosine)
        .meta_filter(col("val").gte(2))
        .take(9)
        .collect()
        .unwrap();
    let stats = store.last_query_stats().unwrap();
    assert_eq!(stats.total_chunks, 3);
    assert_eq!(stats.pruned_chunks, 1, "Only all-null chunk pruned");

    // val > 2 should drop chunk0 now (max=2) and the all-null chunk2 => only chunk1
    let _ = store
        .query(vec![1.0, 0.0], Metric::Cosine)
        .meta_filter(col("val").gt(2))
        .take(9)
        .collect()
        .unwrap();
    let stats2 = store.last_query_stats().unwrap();
    assert_eq!(stats2.evaluated_chunks, 1);
    assert_eq!(stats2.pruned_chunks, 2);
}

#[test]
fn zonemap_all_null_chunk_pruned_for_equality() {
    let store = build_store();
    // Filter on grade == "A"; chunk2 has only nulls => should be pruned.
    let _ = store
        .query(vec![1.0, 0.0], Metric::Cosine)
        .meta_filter(col("grade").eq("A"))
        .take(9)
        .collect()
        .unwrap();
    let stats = store.last_query_stats().unwrap();
    assert_eq!(stats.total_chunks, 3);
    assert!(stats.pruned_chunks >= 1);
}

#[test]
fn zonemap_and_clause_numeric_datetime() {
    let store = build_store();
    // (val > 5) AND (ts < 2025-01-01) should yield only row 5 (val=12, ts=2024-12-31.. )
    let results = store
        .query(vec![1.0, 0.0], Metric::DotProduct)
        .meta_filter(col("val").gt(5).and(col("ts").lt("2025-01-01T00:00:00Z")))
        .take(9)
        .collect()
        .unwrap();

    assert_eq!(
        results.len(),
        1,
        "Exactly one row should satisfy both predicates"
    );
    assert_eq!(results.indices[0], 5);

    let stats = store.last_query_stats().unwrap();
    assert_eq!(stats.total_chunks, 3);
    // Chunk0 pruned by val>5; chunk2 pruned (all null). Only chunk1 evaluated.
    assert_eq!(stats.evaluated_chunks, 1);
    assert_eq!(stats.pruned_chunks, 2);
}

#[test]
fn zonemap_ne_comparator_with_null_only_chunk() {
    let store = build_store();
    // val != 1 — semantically could match many rows, but the all-null chunk must still be pruned.
    let _ = store
        .query(vec![1.0, 0.0], Metric::Cosine)
        .meta_filter(col("val").neq(1))
        .take(9)
        .collect()
        .unwrap();
    let stats = store.last_query_stats().unwrap();
    assert_eq!(stats.total_chunks, 3);
    assert!(
        stats.pruned_chunks >= 1,
        "Null-only chunk should not survive Neq pruning"
    );
}
