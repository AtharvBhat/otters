use otters::error::{OttersError, StoreError};
use otters::prelude::*;

fn sample_embeddings() -> Column {
    let vectors = [
        vec![1.0, 0.0, 0.0],
        vec![0.8, 0.2, 0.0],
        vec![0.6, 0.4, 0.0],
    ];
    let mut builder = Column::new_vector("embedding", 3);
    for vec in &vectors {
        builder = builder.append(Some(vec.as_slice()));
    }
    builder
}

#[test]
fn existing_inv_norms_must_be_non_null() {
    let embeddings = sample_embeddings().collect().unwrap();
    let inv_norms = Column::new_float32("embedding_inv_norms")
        .append([Some(1.0f32), None, Some(1.0f32)])
        .collect()
        .unwrap();

    let result = OttersStore::new(
        ["embedding", "embedding_inv_norms"],
        [embeddings, inv_norms],
    )
    .with_embedding_column("embedding")
    .build();

    assert!(
        matches!(
            result,
            Err(OttersError::Store(StoreError::ColumnTypeMismatch { .. }))
        ),
        "expected a validation error when inverse norms contain nulls"
    );
}

#[test]
fn try_dim_errors_when_not_built() {
    let embeddings = sample_embeddings().collect().unwrap();
    let store = OttersStore::new(["embedding"], [embeddings]).with_embedding_column("embedding");
    let err = store.try_dim().unwrap_err();
    assert!(matches!(err, OttersError::Store(StoreError::NotBuilt)));
}

#[test]
fn row_id_integrity_validation_catches_mismatch() {
    let embeddings = sample_embeddings().collect().unwrap();
    let row_ids = Column::new_int64("row_id")
        .append([Some(0i64), Some(2), Some(1)])
        .collect()
        .unwrap();

    let result = OttersStore::new(["embedding", "row_id"], [embeddings, row_ids])
        .with_embedding_column("embedding")
        .build();

    assert!(matches!(
        result,
        Err(OttersError::Store(StoreError::RowIdIntegrity { .. }))
    ));
}
