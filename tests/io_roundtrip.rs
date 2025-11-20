use arrow::array::StringArray;
use arrow::record_batch::RecordBatch;
use otters::col::Column;
use otters::prelude::*;
use std::path::PathBuf;
use tempfile::tempdir;

fn sample_batch() -> RecordBatch {
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

    let strings = Column::new_string("name")
        .append([Some("alice"), Some("bob"), Some("carol")])
        .collect()
        .unwrap();

    OttersStore::new(["embedding", "age", "name"], [embeddings, ints, strings])
        .with_embedding_column("embedding")
        .build()
        .unwrap()
        .to_recordbatch()
        .unwrap()
}

#[test]
fn parquet_round_trip_preserves_columns() {
    let dir = tempdir().unwrap();
    let path: PathBuf = dir.path().join("store.parquet");

    let store = OttersStore::from_recordbatch(sample_batch())
        .with_embedding_column("embedding")
        .build()
        .unwrap();
    store.write_parquet(&path).unwrap();

    let loaded = OttersStore::from_parquet(path.to_str().unwrap())
        .unwrap()
        .with_embedding_column("embedding")
        .build()
        .unwrap();

    let batch = loaded.batch().unwrap();
    assert_eq!(batch.num_rows(), 3);
    assert!(batch.schema().field_with_name("row_id").is_ok());
    assert!(
        batch
            .schema()
            .field_with_name(loaded.inv_norm_column_name())
            .is_ok()
    );

    let names = batch
        .column(batch.schema().index_of("name").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(names.value(1), "bob");
}
