use std::collections::HashMap;

use otters::expr::{
    CmpOp, ColumnFilter, CompiledFilter, Expr, ExprError, NumericLiteral, col, lit,
};
use otters::types::DataType;

fn schema() -> HashMap<String, DataType> {
    use DataType::*;
    let mut m = HashMap::new();
    m.insert("age".to_string(), Int64);
    m.insert("score".to_string(), Float64);
    m.insert("name".to_string(), DataType::String);
    m.insert("ts".to_string(), DateTime);
    m
}

#[test]
fn numeric_gt_simple() {
    let e = col("age").gt(25);
    let cf = e.compile(&schema()).unwrap();
    let expected = CompiledFilter {
        clauses: vec![vec![ColumnFilter::Numeric {
            column: "age".to_string(),
            cmp: CmpOp::Gt,
            rhs: NumericLiteral::I64(25),
        }]],
    };
    assert_eq!(cf.clauses, expected.clauses);
}

#[test]
fn literal_on_left_is_invalid() {
    // Manually build Literal < Column: 25 < age => invalid under strict mode
    let e = Expr::Cmp {
        left: Box::new(lit(25)),
        right: Box::new(col("age")),
        op: CmpOp::Lt,
    };
    let err = e.compile(&schema()).unwrap_err();
    assert!(matches!(err, ExprError::InvalidComparison));
}

#[test]
fn string_eq_allowed() {
    let e = col("name").eq("alice");
    let cf = e.compile(&schema()).unwrap();
    let expected = CompiledFilter {
        clauses: vec![vec![ColumnFilter::String {
            column: "name".to_string(),
            cmp: CmpOp::Eq,
            rhs: "alice".to_string(),
        }]],
    };
    assert_eq!(cf.clauses, expected.clauses);
}

#[test]
fn string_or_multiple_equalities() {
    // name == Alice OR name == Bob should be one clause with two leaves
    let e = col("name").eq("Alice") | col("name").eq("Bob");
    let cf = e.compile(&schema()).unwrap();
    let expected = CompiledFilter {
        clauses: vec![vec![
            ColumnFilter::String {
                column: "name".to_string(),
                cmp: CmpOp::Eq,
                rhs: "Alice".to_string(),
            },
            ColumnFilter::String {
                column: "name".to_string(),
                cmp: CmpOp::Eq,
                rhs: "Bob".to_string(),
            },
        ]],
    };
    assert_eq!(cf.clauses, expected.clauses);
}

#[test]
fn string_unsupported_op_err() {
    let e = Expr::Cmp {
        left: Box::new(col("name")),
        right: Box::new(lit("bob")),
        op: CmpOp::Gt,
    };
    let err = e.compile(&schema()).unwrap_err();
    assert!(matches!(err, ExprError::UnsupportedStringOp(col) if col == "name"));
}

#[test]
fn type_mismatch_errs() {
    // string literal on int column
    let e = col("age").eq("x");
    let err = e.compile(&schema()).unwrap_err();
    assert!(matches!(err, ExprError::TypeMismatch(col, DataType::Int64, "string") if col == "age"));

    // float literal on int column
    let e2 = col("age").gt(25.5_f64);
    let err2 = e2.compile(&schema()).unwrap_err();
    assert!(matches!(err2, ExprError::TypeMismatch(col, DataType::Int64, "float") if col == "age"));
}

#[test]
fn float_column_widen_int_literal() {
    let e = col("score").gte(80); // int literal -> widened to f64
    let cf = e.compile(&schema()).unwrap();
    let expected = CompiledFilter {
        clauses: vec![vec![ColumnFilter::Numeric {
            column: "score".to_string(),
            cmp: CmpOp::Gte,
            rhs: NumericLiteral::F64(80.0),
        }]],
    };
    assert_eq!(cf.clauses, expected.clauses);
}

#[test]
fn float_column_float_literal() {
    let e = col("score").gt(80.5_f64);
    let cf = e.compile(&schema()).unwrap();
    let expected = CompiledFilter {
        clauses: vec![vec![ColumnFilter::Numeric {
            column: "score".to_string(),
            cmp: CmpOp::Gt,
            rhs: NumericLiteral::F64(80.5),
        }]],
    };
    assert_eq!(cf.clauses, expected.clauses);
}

#[test]
fn and_yields_two_clauses() {
    let e = col("age").gt(25) & col("score").gte(80.0);
    let cf = e.compile(&schema()).unwrap();
    assert_eq!(cf.clauses.len(), 2);
    assert!(matches!(cf.clauses[0][0], ColumnFilter::Numeric { .. }));
    assert!(matches!(cf.clauses[1][0], ColumnFilter::Numeric { .. }));
}

#[test]
fn or_yields_one_clause_with_two_leaves() {
    let e = col("age").gt(25) | col("age").lt(18);
    let cf = e.compile(&schema()).unwrap();
    assert_eq!(cf.clauses.len(), 1);
    assert_eq!(cf.clauses[0].len(), 2);
}

#[test]
fn complex_cnf_distribution() {
    // A & (B | C) => [[A], [B, C]]
    let a = col("age").gt(25);
    let b = col("score").gte(80.0);
    let c = col("age").lt(18);
    let e = a & (b | c);
    let cf = e.compile(&schema()).unwrap();
    assert_eq!(cf.clauses.len(), 2);
    // One clause should have a single leaf, the other two leaves
    let sizes: Vec<usize> = cf.clauses.iter().map(|cl| cl.len()).collect();
    assert!(sizes.contains(&1) && sizes.contains(&2));
}

#[test]
fn unknown_column_error() {
    let e = col("missing").eq(1);
    let err = e.compile(&schema()).unwrap_err();
    assert!(matches!(err, ExprError::UnknownColumn(c) if c == "missing"));
}

#[test]
fn datetime_string_literal_compiles() {
    // RFC3339 string allowed and parsed to millis
    use chrono::{DateTime as ChronoDateTime, Utc};
    let s = "2023-01-02T03:04:05Z";
    let expected_ms = ChronoDateTime::parse_from_rfc3339(s)
        .unwrap()
        .with_timezone(&Utc)
        .timestamp_millis();

    let e = col("ts").gte(s);
    let cf = e.compile(&schema()).unwrap();
    let expected = CompiledFilter {
        clauses: vec![vec![ColumnFilter::Numeric {
            column: "ts".to_string(),
            cmp: CmpOp::Gte,
            rhs: NumericLiteral::I64(expected_ms),
        }]],
    };
    assert_eq!(cf.clauses, expected.clauses);
}

#[test]
fn datetime_non_string_literal_err() {
    let e = col("ts").eq(1700000000000_i64);
    let err = e.compile(&schema()).unwrap_err();
    assert!(
        matches!(err, ExprError::TypeMismatch(col, DataType::DateTime, "datetime string") if col == "ts")
    );
}

#[test]
fn tautology_in_or_clause_is_removed() {
    // (name == bob OR name != bob) AND age > 5 => just age > 5
    let e = (col("name").eq("bob") | col("name").neq("bob")) & col("age").gt(5);
    let cf = e.compile(&schema()).unwrap();
    assert_eq!(cf.clauses.len(), 1);
    assert!(matches!(cf.clauses[0][0], ColumnFilter::Numeric { .. }));
}
