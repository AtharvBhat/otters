use otters::expr::col;
use otters::prelude::*;

fn build_store() -> OttersStore {
    let vectors = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.8, 0.2, 0.0],
        vec![0.6, 0.4, 0.0],
    ];

    let mut vector_builder = Column::new_vector("embedding", 3);
    for vec in &vectors {
        vector_builder = vector_builder.append(Some(vec.as_slice()));
    }
    let embeddings = vector_builder.collect().unwrap();

    let mut age_builder = Column::new_int32("age");
    age_builder = age_builder.append(Some(25));
    age_builder = age_builder.append(Some(35));
    age_builder = age_builder.append(Some(45));
    let ages = age_builder.collect().unwrap();

    OttersStore::new(["embedding", "age"], [embeddings, ages])
        .with_embedding_column("embedding")
        .build()
        .unwrap()
}

#[test]
fn defaults_to_desc_for_cosine_with_take() {
    let store = build_store();
    let output = store
        .query()
        .with_query_vec(vec![1.0, 0.0, 0.0])
        .metric(QueryMetric::Cosine)
        .take(2)
        .collect()
        .unwrap();

    assert_eq!(output.batch.num_rows(), 2);
    let row_ids = output
        .batch
        .column_by_name(store.row_id_column_name())
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap();
    assert_eq!(row_ids.value(0), 0);
    assert_eq!(row_ids.value(1), 1);

    let scores = output
        .batch
        .column_by_name("score")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .unwrap();
    assert!(scores.value(0) >= scores.value(1));
    assert!(output.warning.is_none());
}

#[test]
fn defaults_to_asc_for_euclidean_with_take() {
    let store = build_store();
    let output = store
        .query()
        .with_query_vec(vec![1.0, 0.0, 0.0])
        .metric(QueryMetric::Euclidean)
        .take(2)
        .collect()
        .unwrap();

    let row_ids = output
        .batch
        .column_by_name(store.row_id_column_name())
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap();
    assert_eq!(row_ids.value(0), 0);
    assert_eq!(row_ids.value(1), 1);
    assert!(output.warning.is_none());
}

#[test]
fn respects_explicit_order_override() {
    let store = build_store();
    let output = store
        .query()
        .with_query_vec(vec![1.0, 0.0, 0.0])
        .metric(QueryMetric::Euclidean)
        .order_by_desc()
        .take(2)
        .collect()
        .unwrap();

    let row_ids = output
        .batch
        .column_by_name(store.row_id_column_name())
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap();
    assert_eq!(row_ids.value(0), 2);
    assert!(output.warning.is_none());
}

#[test]
fn supports_metadata_and_metric_filters() {
    let store = build_store();

    let output = store
        .query()
        .with_query_vec(vec![1.0, 0.0, 0.0])
        .filter(col("embedding").cosine().gt(0.7) & col("age").gte(40))
        .order_by_desc()
        .collect()
        .unwrap();

    assert_eq!(output.batch.num_rows(), 1);
    let row_ids = output
        .batch
        .column_by_name(store.row_id_column_name())
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap();
    assert_eq!(row_ids.value(0), 2);
    assert!(output.warning.is_none());
}

#[test]
fn emits_warning_when_skipping_null_embeddings() {
    let embeddings = Column::new_vector("embedding", 3)
        .append([
            Some([1.0f32, 0.0, 0.0].as_slice()),
            None,
            Some([0.5f32, 0.5, 0.0].as_slice()),
        ])
        .collect()
        .unwrap();

    let ages = Column::new_int32("age")
        .append([Some(25), Some(35), Some(45)])
        .collect()
        .unwrap();

    let store = OttersStore::new(["embedding", "age"], [embeddings, ages])
        .with_embedding_column("embedding")
        .build()
        .unwrap();

    let output = store
        .query()
        .with_query_vec(vec![1.0, 0.0, 0.0])
        .take(3)
        .collect()
        .unwrap();

    assert_eq!(output.batch.num_rows(), 2);
    assert!(output.warning.is_some());
}

#[test]
fn metric_conflict_is_reported() {
    let store = build_store();
    let result = store
        .query()
        .with_query_vec(vec![1.0, 0.0, 0.0])
        .metric(QueryMetric::Euclidean)
        .filter(col("embedding").cosine().gt(0.1))
        .collect();

    assert!(matches!(
        result,
        Err(otters::error::OttersError::Query(
            otters::error::QueryError::MetricConflict { .. }
        ))
    ));
}

#[test]
fn mixed_metric_and_metadata_or_clause_is_rejected() {
    let store = build_store();
    let result = store
        .query()
        .with_query_vec(vec![1.0, 0.0, 0.0])
        .filter(col("age").gt(10) | col("embedding").cosine().gt(0.1))
        .collect();

    assert!(matches!(
        result,
        Err(otters::error::OttersError::Expr(
            otters::expr::ExprError::MixedMetricMetadataClause
        ))
    ));
}
