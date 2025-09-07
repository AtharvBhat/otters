use otters::expr::col;
use otters::prelude::*;

#[test]
fn meta_basic_pruning_and_stats() {
    let vectors = vec![
        vec![1.0, 0.0, 0.0], // 0
        vec![0.0, 1.0, 0.0], // 1
        vec![0.5, 0.5, 0.0], // 2
        vec![0.0, 0.0, 1.0], // 3
    ];
    let age = Column::new("age", DataType::Int32)
        .from(vec![Some(10), Some(20), Some(30), None])
        .unwrap();
    let grade = Column::new("grade", DataType::String)
        .from(vec![Some("A"), Some("B"), Some("A"), Some("C")])
        .unwrap();
    let meta = MetaStore::from_columns(vec![("age".into(), age), ("grade".into(), grade)])
        .with_vectors(vectors)
        .with_chunk_size(2)
        .build()
        .unwrap();

    let results = meta
        .query(vec![1.0, 0.0, 0.0], Metric::Cosine)
        .meta_filter(col("age").gt(15).and(col("grade").eq("A")))
        .unwrap()
        .with_stats()
        .take(4)
        .collect()
        .unwrap();

    // Only idx 2 matches age>15 and grade==A
    assert_eq!(results.len(), 1);
    let set: std::collections::HashSet<usize> = results[0].iter().map(|(i, _)| *i).collect();
    assert!(set.contains(&2));
    assert_eq!(set.len(), 1);

    let stats = meta.last_query_stats().expect("stats present");
    assert_eq!(stats.total_chunks, 2);
    assert!(stats.evaluated_chunks >= 1);
    assert!(stats.results_before_postfilter >= stats.results_after_postfilter);
}

#[test]
fn meta_string_eq_prunes_chunks() {
    // 6 rows, chunk size 3. First chunk has no "A", second chunk has "A".
    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![0.5, 0.5, 0.0],
    ];
    let ages = Column::new("age", DataType::Int32)
        .from(vec![Some(10), Some(11), Some(12), Some(20), Some(21), Some(22)])
        .unwrap();
    let grades = Column::new("grade", DataType::String)
        .from(vec![Some("B"), Some("C"), Some("B+"), Some("A"), Some("A"), Some("C")])
        .unwrap();
    let meta = MetaStore::from_columns(vec![("age".into(), ages), ("grade".into(), grades)])
        .with_vectors(vectors)
        .with_chunk_size(3)
        .build()
        .unwrap();

    // meta filter only on grade=="A" should prune first chunk
    let _ = meta
        .query(vec![1.0, 0.0, 0.0], Metric::Cosine)
        .meta_filter(col("grade").eq("A"))
        .unwrap()
        .with_stats()
        .take(6)
        .collect()
        .unwrap();
    let stats = meta.last_query_stats().unwrap();
    assert_eq!(stats.total_chunks, 2);
    assert!(stats.pruned_chunks >= 1);
}

#[test]
fn meta_datetime_range_filter() {
    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let ts = Column::new("ts", DataType::DateTime)
        .from(vec![
            Some("2023-01-01T00:00:00Z"),
            Some("2023-06-01T00:00:00Z"),
            Some("2024-01-01T00:00:00Z"),
        ])
        .unwrap();
    let meta = MetaStore::from_columns(vec![("ts".into(), ts)])
        .with_vectors(vectors)
        .with_chunk_size(2)
        .build()
        .unwrap();

    // Keep only rows in 2023
    let results = meta
        .query(vec![1.0, 0.0], Metric::DotProduct)
        .meta_filter(col("ts").gte("2023-01-01T00:00:00Z").and(col("ts").lt("2024-01-01T00:00:00Z")))
        .unwrap()
        .with_stats()
        .take(3)
        .collect()
        .unwrap();
    let set: std::collections::HashSet<usize> = results[0].iter().map(|(i, _)| *i).collect();
    assert_eq!(set, [0usize, 1usize].into_iter().collect());
}

#[test]
fn meta_global_scope_merge_and_vec_threshold() {
    let vectors = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 0.0],
    ];
    let grade = Column::new("grade", DataType::String)
        .from(vec![Some("A"), Some("B"), Some("A"), Some("A")])
        .unwrap();
    let meta = MetaStore::from_columns(vec![("grade".into(), grade)])
        .with_vectors(vectors)
        .with_chunk_size(2)
        .build()
        .unwrap();

    let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let results = meta
        .query_batch(queries, Metric::DotProduct)
        .meta_filter(col("grade").eq("A"))
        .unwrap()
        .vec_filter(0.5, Cmp::Gt)
        .with_stats()
        .take_global(2)
        .collect()
        .unwrap();

    // Global scope returns single list
    assert_eq!(results.len(), 1);
    assert!(results[0].len() <= 2);

    let stats = meta.last_query_stats().unwrap();
    // With k=2 and two chunks, prefilter results can't exceed 4 (best case)
    assert!(stats.results_before_postfilter <= 4);
}

#[test]
fn meta_build_mismatched_column_len_errors() {
    let vectors = vec![vec![1.0], vec![2.0]];
    let bad_col = Column::new("age", DataType::Int32).from(vec![Some(1)]).unwrap();
    let result = MetaStore::from_columns(vec![("age".into(), bad_col)])
        .with_vectors(vectors)
        .with_chunk_size(2)
        .build();
    assert!(result.is_err());
}

#[test]
fn meta_stats_without_meta_filter() {
    let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let meta = MetaStore::from_columns(Vec::new())
        .with_vectors(vectors)
        .with_chunk_size(2)
        .build()
        .unwrap();

    let _ = meta
        .query(vec![1.0, 0.0], Metric::Cosine)
        .with_stats()
        .take(3)
        .collect()
        .unwrap();
    let stats = meta.last_query_stats().unwrap();
    assert_eq!(stats.results_before_postfilter, stats.results_after_postfilter);
}
