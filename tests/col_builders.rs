use otters::col::Column;

#[test]
fn vector_builder_rejects_wrong_length() {
    let builder = Column::new_vector("embedding", 3).append(Some([1.0f32, 2.0].as_slice()));
    let err = builder.collect().expect_err("expected length mismatch");
    assert!(
        err.to_string()
            .contains("Vector length does not match fixed dimension")
    );
}

#[test]
fn vector_builder_preserves_null_entries() {
    let column = Column::new_vector("embedding", 2)
        .append([
            Some([1.0f32, 0.0].as_slice()),
            None,
            Some([0.5f32, 0.5].as_slice()),
        ])
        .collect()
        .unwrap();

    assert_eq!(column.len(), 3);
    assert_eq!(column.null_count(), 1);
    assert_eq!(column.vector_at(0).unwrap(), vec![1.0, 0.0]);
    assert!(column.is_null(1));
}

#[test]
fn vector_at_out_of_bounds_returns_none() {
    let column = Column::new_vector("embedding", 2)
        .append([Some([1.0f32, 0.0].as_slice())])
        .collect()
        .unwrap();
    assert!(column.vector_at(5).is_none());
}
