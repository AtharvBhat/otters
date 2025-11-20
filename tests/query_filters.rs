use otters::expr::col;
use otters::prelude::*;

fn build_store() -> OttersStore {
    let embeddings = Column::new_vector("embedding", 2)
        .append([
            Some([1.0f32, 0.0].as_slice()),
            Some([0.0f32, 1.0].as_slice()),
            Some([0.5f32, 0.5].as_slice()),
        ])
        .collect()
        .unwrap();

    let ints = Column::new_int32("age")
        .append([Some(21), Some(30), Some(40)])
        .collect()
        .unwrap();

    let floats = Column::new_float32("score_f")
        .append([Some(1.5f32), Some(2.5), Some(3.5)])
        .collect()
        .unwrap();

    let strings = Column::new_string("name")
        .append([Some("alice"), Some("bob"), Some("carol")])
        .collect()
        .unwrap();

    let millis = Column::new_timestamp("created_at")
        .append([
            Some(1_700_000_000_000i64),
            Some(1_700_000_100_000),
            Some(1_700_000_200_000),
        ])
        .collect()
        .unwrap();

    OttersStore::new(
        ["embedding", "age", "score_f", "name", "created_at"],
        [embeddings, ints, floats, strings, millis],
    )
    .with_embedding_column("embedding")
    .build()
    .unwrap()
}

#[test]
fn metadata_filters_across_types_work() {
    let store = build_store();
    let output = store
        .query()
        .with_query_vec(vec![1.0, 0.0])
        .filter(
            col("age").gte(30)
                & col("score_f").lt(3.0)
                & col("name").neq("alice")
                & col("created_at").gt("2023-11-14T00:00:00Z"),
        )
        .collect()
        .unwrap();

    assert_eq!(output.batch.num_rows(), 1);
    let names = output
        .batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap();
    assert_eq!(names.value(0), "bob");
    assert!(output.warning.is_none());
}

#[test]
fn metric_plan_threshold_filters_after_scoring() {
    let store = build_store();
    let output = store
        .query()
        .with_query_vec(vec![1.0, 0.0])
        .filter(col("embedding").cosine().gt(0.9))
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
    assert_eq!(row_ids.len(), 1);
}
