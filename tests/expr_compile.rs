use otters::expr::{DataType, ExprError, MetricExpr, col, cosine};
use std::collections::HashMap;

fn schema(map: &[(&str, DataType)]) -> HashMap<String, DataType> {
    map.iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect::<HashMap<_, _>>()
}

#[test]
fn rejects_string_gt() {
    let expr = col("name").gt("carol");
    let err = expr
        .compile(&schema(&[("name", DataType::String)]))
        .unwrap_err();
    assert!(matches!(err, ExprError::UnsupportedStringOp(c) if c == "name"));
}

#[test]
fn rejects_float_literal_on_int_column() {
    let expr = col("age").gt(3.14);
    let err = expr
        .compile(&schema(&[("age", DataType::Int64)]))
        .unwrap_err();
    assert!(matches!(err, ExprError::TypeMismatch(c, _, _) if c == "age"));
}

#[test]
fn rejects_metric_literal_string() {
    let expr = col("embedding").cosine().gt("foo");
    let err = expr
        .compile(&schema(&[("embedding", DataType::Float32)]))
        .unwrap_err();
    assert!(matches!(err, ExprError::UnsupportedMetricLiteral));
}

#[test]
fn mixed_metric_metadata_clause_is_invalid() {
    let expr = col("age").gt(10) | otters::expr::cosine().gt(0.5);
    let err = expr
        .compile(&schema(&[("age", DataType::Int32)]))
        .unwrap_err();
    assert!(matches!(err, ExprError::MixedMetricMetadataClause));
}

#[test]
fn metric_plan_extracted_correctly() {
    use otters::expr::CompiledFilter;
    let expr = cosine().gt(0.5) & col("age").gt(20);
    let compiled = expr.compile(&schema(&[
        ("embedding", DataType::Float32),
        ("age", DataType::Int32),
    ]));
    assert!(compiled.is_ok());

    let cf: CompiledFilter = compiled.unwrap();
    let plan = cf.metric_plan();
    assert_eq!(plan.len(), 1);
    assert_eq!(plan[0].len(), 1);
    assert_eq!(plan[0][0].metric, MetricExpr::Cosine);
}
