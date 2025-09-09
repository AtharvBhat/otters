//! Expression DSL for metadata filtering
//!
//! Otters exposes a small, ergonomic expression DSL for describing metadata
//! predicates that can be pushed down into pruning and row-level filtering.
//! Build expressions with `col("name")` and combine them with `&` (AND) and
//! `|` (OR). Expressions are type-checked and compiled against your schema
//! before being evaluated.
//!
//! Examples
//! --------
//! ```rust
//! use otters::expr::col;
//!
//! // price <= 40 AND version >= 2
//! let e1 = col("price").lte(40.0) & col("version").gte(2);
//!
//! // (age < 18 OR age > 65) AND name != "alice"
//! let e2 = (col("age").lt(18) | col("age").gt(65)) & col("name").neq("alice");
//!
//! // string equality OR equality
//! let e3 = col("grade").eq("A") | col("grade").eq("B");
//! ```
//!
//! Datatypes and operators
//! -----------------------
//! - String: Eq / Neq only
//! - Int32 / Int64: Eq / Neq / Lt / Lte / Gt / Gte with integer literals
//! - Float32 / Float64: same operators with float or integer literals
//! - DateTime: same operators with a parseable datetime string
//!   (RFC3339/ISO8601, `YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`)
//!
//! Compiling
//! ---------
//! Call `Expr::compile(&schema)` to type-check the expression against your
//! column types and obtain a `CompiledFilter` plan used internally by the
//! engine for fast pruning.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use crate::type_utils::DataType;

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    I64(i64),
    F64(f64),
    Str(String),
}

impl From<i32> for Literal {
    fn from(v: i32) -> Self {
        Literal::I64(v as i64)
    }
}
impl From<i64> for Literal {
    fn from(v: i64) -> Self {
        Literal::I64(v)
    }
}
impl From<f32> for Literal {
    fn from(v: f32) -> Self {
        Literal::F64(v as f64)
    }
}
impl From<f64> for Literal {
    fn from(v: f64) -> Self {
        Literal::F64(v)
    }
}
impl From<&str> for Literal {
    fn from(v: &str) -> Self {
        Literal::Str(v.to_string())
    }
}
impl From<String> for Literal {
    fn from(v: String) -> Self {
        Literal::Str(v)
    }
}

/// Comparison operator used in expression leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(String),
    Literal(Literal),
    Cmp {
        left: Box<Expr>,
        right: Box<Expr>,
        op: CmpOp,
    },
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    // No Not as it can be done with Neq in Cmp
    // And just complicates the expression tree
}

/// Builder for a column reference (polars-like DSL).
pub fn col(name: &str) -> Expr {
    Expr::Column(name.to_string())
}
/// Builder for a literal value.
pub fn lit<T: Into<Literal>>(v: T) -> Expr {
    Expr::Literal(v.into())
}

impl Expr {
    // Comparison builders
    /// Column == value
    pub fn eq<T: Into<Literal>>(self, v: T) -> Expr {
        Expr::Cmp {
            left: Box::new(self),
            right: Box::new(lit(v)),
            op: CmpOp::Eq,
        }
    }
    /// Column != value
    pub fn neq<T: Into<Literal>>(self, v: T) -> Expr {
        Expr::Cmp {
            left: Box::new(self),
            right: Box::new(lit(v)),
            op: CmpOp::Neq,
        }
    }
    /// Column < value
    pub fn lt<T: Into<Literal>>(self, v: T) -> Expr {
        Expr::Cmp {
            left: Box::new(self),
            right: Box::new(lit(v)),
            op: CmpOp::Lt,
        }
    }
    /// Column <= value
    pub fn lte<T: Into<Literal>>(self, v: T) -> Expr {
        Expr::Cmp {
            left: Box::new(self),
            right: Box::new(lit(v)),
            op: CmpOp::Lte,
        }
    }
    /// Column > value
    pub fn gt<T: Into<Literal>>(self, v: T) -> Expr {
        Expr::Cmp {
            left: Box::new(self),
            right: Box::new(lit(v)),
            op: CmpOp::Gt,
        }
    }
    /// Column >= value
    pub fn gte<T: Into<Literal>>(self, v: T) -> Expr {
        Expr::Cmp {
            left: Box::new(self),
            right: Box::new(lit(v)),
            op: CmpOp::Gte,
        }
    }

    /// Logical AND
    pub fn and(self, other: Expr) -> Expr {
        Expr::And(Box::new(self), Box::new(other))
    }
    /// Logical OR
    pub fn or(self, other: Expr) -> Expr {
        Expr::Or(Box::new(self), Box::new(other))
    }
}

// Allow use of &, | operators
impl std::ops::BitAnd for Expr {
    type Output = Expr;
    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(rhs)
    }
}
impl std::ops::BitOr for Expr {
    type Output = Expr;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.or(rhs)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NumericLiteral {
    I64(i64),
    F64(f64),
}

/// A compiled, typed column filter used at evaluation time.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnFilter {
    Numeric {
        column: String,
        cmp: CmpOp,
        rhs: NumericLiteral,
    },
    String {
        column: String,
        cmp: CmpOp,
        rhs: String,
    },
}

/// Filter plan representation used by the expression compiler.
///
/// Invariant:
/// - The outer Vec represents an AND over clauses (every clause must pass).
/// - Each inner Vec represents a single clause, which is an OR over ColumnFilter items.
///
/// Example: `[[A, B], [C]]` means `(A OR B) AND (C)`.
pub type Plan = Vec<Vec<ColumnFilter>>;

/// Compiled expression that is ready to be evaluated.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledFilter {
    pub clauses: Plan,
}

/// Errors returned while compiling expressions to a filter plan.
#[derive(Debug, PartialEq)]
pub enum ExprError {
    UnknownColumn(String),
    TypeMismatch(String, DataType, &'static str),
    UnsupportedStringOp(String),
    InvalidComparison,
    InvalidExpression,
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprError::UnknownColumn(c) => write!(f, "Unknown column '{c}'"),
            ExprError::TypeMismatch(c, dt, got) => {
                write!(
                    f,
                    "Type mismatch for column '{c}': expected {dt:?}, got literal {got}"
                )
            }
            ExprError::UnsupportedStringOp(c) => {
                write!(f, "Unsupported comparator for string column '{c}'")
            }
            ExprError::InvalidComparison => write!(
                f,
                "Invalid expression shape for comparison (expect column vs literal)"
            ),
            ExprError::InvalidExpression => write!(
                f,
                "Invalid expression (unexpected literal or column without comparator)"
            ),
        }
    }
}

impl Error for ExprError {}

// Local datetime parsing (mirrors logic in col.rs)
// Accepts: RFC3339/ISO8601, YYYY-MM-DD, YYYY-MM-DD HH:MM:SS
fn parse_datetime_literal_millis(s: &str) -> Option<i64> {
    // Use chrono just like col.rs
    use chrono::{DateTime as ChronoDateTime, NaiveDate, NaiveDateTime, Utc};

    if let Ok(dt) = ChronoDateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc).timestamp_millis());
    }
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d")
        && let Some(dt) = date.and_hms_opt(0, 0, 0)
    {
        return Some(dt.and_utc().timestamp_millis());
    }
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.and_utc().timestamp_millis());
    }
    None
}

impl Expr {
    /// Type-check and lower the expression against a `schema` into a `CompiledFilter`.
    ///
    /// The schema maps column names to `DataType` and is used to validate
    /// comparator compatibility and coerce literals (e.g., ints to floats).
    pub fn compile(&self, schema: &HashMap<String, DataType>) -> Result<CompiledFilter, ExprError> {
        // Lower the expression to a filter plan
        // and then normalize the clauses
        let plan = lower_to_plan(self, schema)?;
        Ok(CompiledFilter {
            clauses: normalize_plan(plan),
        })
    }
}

/// Normalize a plan by:
/// - Dropping clauses that are tautologies like `(col == v) OR (col != v)`
fn normalize_plan(mut plan: Plan) -> Plan {
    let mut out: Plan = Vec::with_capacity(plan.len());

    for clause in plan.drain(..) {
        let mut is_tautology = false;

        // Check for simple tautology: same col/value with Eq and Neq in the same clause
        for lf in &clause {
            match lf {
                ColumnFilter::Numeric { column, cmp, rhs } if *cmp == CmpOp::Eq => {
                    let conflict = clause.iter().any(|x| matches!(
                        x,
                        ColumnFilter::Numeric { column: c2, cmp: CmpOp::Neq, rhs: v2 } if c2 == column && v2 == rhs
                    ));
                    if conflict {
                        is_tautology = true;
                        break;
                    }
                }
                ColumnFilter::String { column, cmp, rhs } if *cmp == CmpOp::Eq => {
                    let conflict = clause.iter().any(|x| matches!(
                        x,
                        ColumnFilter::String { column: c2, cmp: CmpOp::Neq, rhs: v2 } if c2 == column && v2 == rhs
                    ));
                    if conflict {
                        is_tautology = true;
                        break;
                    }
                }
                _ => {}
            }
        }

        if is_tautology {
            continue;
        }

        // Keep clause as-is (no deduping of filters; no clause-level dedup)
        out.push(clause);
    }
    out
}

/// Lower an expression into a filter plan (AND of clauses with OR-inside).
///
/// Rules:
/// - Cmp leaf => one clause with one filter: `[[filter]]`
/// - And(a, b) => concatenate clause lists: `and_concat_clauses(lower(a), lower(b))`
/// - Or(a, b)  => distribute OR over AND: `or_distribute_clauses(lower(a), lower(b))`
///
/// Validation:
/// - Uses the provided `schema` to type-check leaves via `compile_cmp_leaf`.
/// - Returns `ExprError` for invalid shapes (e.g., literal on the left) or type mismatches.
fn lower_to_plan(expr: &Expr, schema: &HashMap<String, DataType>) -> Result<Plan, ExprError> {
    match expr {
        Expr::And(a, b) => {
            let left = lower_to_plan(a, schema)?;
            let right = lower_to_plan(b, schema)?;
            Ok(and_concat_clauses(left, right))
        }
        Expr::Or(a, b) => {
            let left = lower_to_plan(a, schema)?;
            let right = lower_to_plan(b, schema)?;
            Ok(or_distribute_clauses(left, right))
        }
        Expr::Cmp { left, right, op } => {
            compile_cmp_leaf(left, right, *op, schema).map(|f| vec![vec![f]])
        }
        Expr::Column(_) | Expr::Literal(_) => Err(ExprError::InvalidExpression),
    }
}

/// Compile a single comparison leaf into a `ColumnFilter`.
///
/// Requirements:
/// - Shape must be `Column op Literal`; otherwise `InvalidComparison`.
/// - Column must exist in `schema`; otherwise `UnknownColumn`.
///
/// Type rules per column data type:
/// - String: only `Eq`/`Neq`; literal must be a string; other ops => `UnsupportedStringOp`.
/// - Int32/Int64: literal must be `i64`; floats/strings => `TypeMismatch`.
/// - Float32/Float64: literal may be `f64` or `i64` (widened to f64); strings => `TypeMismatch`.
/// - DateTime: literal must be a parseable datetime string; stored as i64 millis; others => `TypeMismatch`.
fn compile_cmp_leaf(
    left: &Expr,
    right: &Expr,
    op: CmpOp,
    schema: &HashMap<String, DataType>,
) -> Result<ColumnFilter, ExprError> {
    let (col_name, lit) = match (left, right) {
        (Expr::Column(name), Expr::Literal(l)) => (name.clone(), l.clone()),
        _ => return Err(ExprError::InvalidComparison),
    };

    let dtype = schema
        .get(&col_name)
        .ok_or_else(|| ExprError::UnknownColumn(col_name.clone()))?;

    match dtype {
        DataType::String => {
            // Only Eq / Neq allowed
            let cmp = match op {
                CmpOp::Eq => CmpOp::Eq,
                CmpOp::Neq => CmpOp::Neq,
                _ => return Err(ExprError::UnsupportedStringOp(col_name)),
            };
            let rhs = match lit {
                Literal::Str(s) => s,
                Literal::I64(_) | Literal::F64(_) => {
                    return Err(ExprError::TypeMismatch(col_name, *dtype, "string"));
                }
            };
            Ok(ColumnFilter::String {
                column: col_name,
                cmp,
                rhs,
            })
        }
        DataType::Int32 | DataType::Int64 => {
            // Numeric integral literal only
            let rhs = match lit {
                Literal::I64(v) => NumericLiteral::I64(v),
                Literal::F64(_) => return Err(ExprError::TypeMismatch(col_name, *dtype, "float")),
                Literal::Str(_) => return Err(ExprError::TypeMismatch(col_name, *dtype, "string")),
            };
            Ok(ColumnFilter::Numeric {
                column: col_name,
                cmp: op,
                rhs,
            })
        }
        DataType::DateTime => {
            // Accept only datetime-parseable string literals; store as i64 millis
            let millis = match lit {
                Literal::Str(s) => match parse_datetime_literal_millis(&s) {
                    Some(ms) => ms,
                    None => {
                        return Err(ExprError::TypeMismatch(col_name, *dtype, "datetime string"));
                    }
                },
                Literal::I64(_) | Literal::F64(_) => {
                    return Err(ExprError::TypeMismatch(col_name, *dtype, "datetime string"));
                }
            };
            Ok(ColumnFilter::Numeric {
                column: col_name,
                cmp: op,
                rhs: NumericLiteral::I64(millis),
            })
        }
        DataType::Float32 | DataType::Float64 => {
            // Numeric float literal (allow ints by widening)
            let rhs = match lit {
                Literal::I64(v) => NumericLiteral::F64(v as f64),
                Literal::F64(v) => NumericLiteral::F64(v),
                Literal::Str(_) => return Err(ExprError::TypeMismatch(col_name, *dtype, "string")),
            };
            Ok(ColumnFilter::Numeric {
                column: col_name,
                cmp: op,
                rhs,
            })
        }
    }
}

/// AND-combine two plans by concatenating their clause lists (outer Vecs).
///
/// Rationale: a plan is an AND of clauses; AND keeps all existing clauses from both sides.
/// Edge cases: if one side is empty, return the other.
///
/// Example: `[[A], [B]] AND [[C]]` => `[[A], [B], [C]]`.
fn and_concat_clauses(mut a: Plan, b: Plan) -> Plan {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    a.extend(b);
    a
}

/// OR-combine two plans by distributing OR over AND (cross-product of clauses).
///
/// For every clause ca in `a` and every clause cb in `b`, produce a new clause `ca ∪ cb`.
/// Each output clause is an OR of the original filters; the set of all such clauses is ANDed.
/// Edge cases: if one side is empty, return the other.
///
/// Examples:
/// - `[[A]]  OR [[B]]`            => `[[A, B]]`
/// - `[[A1],[A2]] OR [[B1],[B2]]` => `[[A1,B1],[A1,B2],[A2,B1],[A2,B2]]`.
fn or_distribute_clauses(a: Plan, b: Plan) -> Plan {
    if a.is_empty() {
        return b;
    }
    if b.is_empty() {
        return a;
    }
    a.iter()
        .flat_map(|ca| {
            b.iter().map(move |cb| {
                let mut merged = Vec::with_capacity(ca.len() + cb.len());
                merged.extend_from_slice(ca);
                merged.extend_from_slice(cb);
                merged
            })
        })
        .collect()
}
