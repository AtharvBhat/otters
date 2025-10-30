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

use crate::datetime::try_parse_datetime_millis;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Logical data types understood by the expression compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int32,
    Int64,
    Float32,
    Float64,
    String,
    DateTime,
}

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

/// Identifier for the metric (vector similarity) used in query expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricExpr {
    Cosine,
    DotProduct,
    Euclidean,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(String),
    Literal(Literal),
    Metric(MetricExpr),
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

/// Builder for a metric pseudo-column (cosine similarity).
pub fn cosine() -> Expr {
    Expr::Metric(MetricExpr::Cosine)
}

/// Builder for a metric pseudo-column (dot product).
pub fn dot_product() -> Expr {
    Expr::Metric(MetricExpr::DotProduct)
}

/// Builder for a metric pseudo-column (squared euclidean distance).
pub fn euclidean() -> Expr {
    Expr::Metric(MetricExpr::Euclidean)
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

#[derive(Debug, Clone, PartialEq)]
pub struct MetricFilter {
    pub metric: MetricExpr,
    pub cmp: CmpOp,
    pub threshold: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FilterItem {
    Metadata(ColumnFilter),
    Metric(MetricFilter),
}

/// Filter plan representation used by the expression compiler.
///
/// Invariant:
/// - The outer Vec represents an AND over clauses (every clause must pass).
/// - Each inner Vec represents a single clause, which is an OR over ColumnFilter items.
///
/// Example: `[[A, B], [C]]` means `(A OR B) AND (C)`.
pub type Plan = Vec<Vec<FilterItem>>;

/// Metadata-only filter plan (AND-of-OR clauses over metadata predicates).
pub type MetadataPlan = Vec<Vec<ColumnFilter>>;
/// Metric-only filter plan.
pub type MetricPlan = Vec<Vec<MetricFilter>>;

/// Compiled expression that is ready to be evaluated.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledFilter {
    pub clauses: Plan,
    pub metadata_clauses: MetadataPlan,
}

/// Errors returned while compiling expressions to a filter plan.
#[derive(Debug, PartialEq)]
pub enum ExprError {
    UnknownColumn(String),
    TypeMismatch(String, DataType, &'static str),
    UnsupportedStringOp(String),
    InvalidComparison,
    InvalidExpression,
    InvalidMetricComparison,
    UnsupportedMetricLiteral,
    MixedMetricMetadataClause,
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
            ExprError::InvalidMetricComparison => write!(
                f,
                "Invalid expression shape for metric comparison (expect metric vs literal)"
            ),
            ExprError::UnsupportedMetricLiteral => {
                write!(f, "Metric comparisons require numeric literals")
            }
            ExprError::MixedMetricMetadataClause => {
                write!(f, "Metric and metadata predicates cannot be OR-ed together")
            }
        }
    }
}

impl Error for ExprError {}

impl Expr {
    /// Type-check and lower the expression against a `schema` into a `CompiledFilter`.
    ///
    /// The schema maps column names to `DataType` and is used to validate
    /// comparator compatibility and coerce literals (e.g., ints to floats).
    pub fn compile(&self, schema: &HashMap<String, DataType>) -> Result<CompiledFilter, ExprError> {
        // Lower the expression to a filter plan
        // and then normalize the metadata clauses
        let plan = lower_to_plan(self, schema)?;
        validate_plan(&plan)?;
        let metadata_plan = extract_metadata_plan(&plan);
        let metadata_clauses = normalize_metadata_plan(metadata_plan);
        Ok(CompiledFilter {
            clauses: plan,
            metadata_clauses,
        })
    }
}

impl CompiledFilter {
    /// Extract the metric-only plan (AND-of-OR clauses) for post-scoring filtering.
    pub fn metric_plan(&self) -> MetricPlan {
        extract_metric_plan(&self.clauses)
    }

    /// Returns true if the filter contains any metric predicates.
    pub fn has_metric_filters(&self) -> bool {
        self.clauses
            .iter()
            .flatten()
            .any(|item| matches!(item, FilterItem::Metric(_)))
    }
}

fn validate_plan(plan: &Plan) -> Result<(), ExprError> {
    for clause in plan {
        if clause_contains_both_kinds(clause) {
            return Err(ExprError::MixedMetricMetadataClause);
        }
    }
    Ok(())
}

fn extract_metadata_plan(plan: &Plan) -> MetadataPlan {
    collect_plan_items(plan, |item| match item {
        FilterItem::Metadata(cf) => Some(cf.clone()),
        FilterItem::Metric(_) => None,
    })
}

fn extract_metric_plan(plan: &Plan) -> MetricPlan {
    collect_plan_items(plan, |item| match item {
        FilterItem::Metric(mf) => Some(mf.clone()),
        FilterItem::Metadata(_) => None,
    })
}

fn normalize_metadata_plan(plan: MetadataPlan) -> MetadataPlan {
    plan.into_iter()
        .filter(|clause| !metadata_clause_is_tautology(clause))
        .collect()
}

fn clause_contains_both_kinds(clause: &[FilterItem]) -> bool {
    let mut has_metadata = false;
    let mut has_metric = false;
    for item in clause {
        match item {
            FilterItem::Metadata(_) => has_metadata = true,
            FilterItem::Metric(_) => has_metric = true,
        }
        if has_metadata && has_metric {
            return true;
        }
    }
    false
}

fn collect_plan_items<T, F>(plan: &Plan, mut map: F) -> Vec<Vec<T>>
where
    T: Clone,
    F: FnMut(&FilterItem) -> Option<T>,
{
    let mut out: Vec<Vec<T>> = Vec::new();
    for clause in plan {
        let collected: Vec<T> = clause.iter().filter_map(&mut map).collect();
        if !collected.is_empty() {
            out.push(collected);
        }
    }
    out
}

fn metadata_clause_is_tautology(clause: &[ColumnFilter]) -> bool {
    clause.iter().any(|filter| match filter {
        ColumnFilter::Numeric { column, cmp: CmpOp::Eq, rhs } => clause.iter().any(|other| matches!(
            other,
            ColumnFilter::Numeric { column: c2, cmp: CmpOp::Neq, rhs: v2 } if c2 == column && v2 == rhs
        )),
        ColumnFilter::String { column, cmp: CmpOp::Eq, rhs } => clause.iter().any(|other| matches!(
            other,
            ColumnFilter::String { column: c2, cmp: CmpOp::Neq, rhs: v2 } if c2 == column && v2 == rhs
        )),
        _ => false,
    })
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
        Expr::Column(_) | Expr::Literal(_) | Expr::Metric(_) => Err(ExprError::InvalidExpression),
    }
}

/// Compile a single comparison leaf into a filter item (metadata column or metric).
///
/// Requirements:
/// - Metadata: shape must be `Column op Literal`; column must exist in `schema`.
/// - Metric: shape must be `Metric op Literal`; literal must be numeric.
fn compile_cmp_leaf(
    left: &Expr,
    right: &Expr,
    op: CmpOp,
    schema: &HashMap<String, DataType>,
) -> Result<FilterItem, ExprError> {
    match (left, right) {
        (Expr::Column(name), Expr::Literal(lit)) => {
            compile_column_cmp(name, lit.clone(), op, schema).map(FilterItem::Metadata)
        }
        (Expr::Metric(metric), Expr::Literal(lit)) => {
            compile_metric_cmp(*metric, lit.clone(), op).map(FilterItem::Metric)
        }
        (Expr::Column(_), _) => Err(ExprError::InvalidComparison),
        (Expr::Metric(_), _) => Err(ExprError::InvalidMetricComparison),
        _ => Err(ExprError::InvalidComparison),
    }
}

fn compile_metric_cmp(
    metric: MetricExpr,
    lit: Literal,
    op: CmpOp,
) -> Result<MetricFilter, ExprError> {
    let threshold = match lit {
        Literal::I64(v) => v as f32,
        Literal::F64(v) => v as f32,
        Literal::Str(_) => return Err(ExprError::UnsupportedMetricLiteral),
    };

    if !threshold.is_finite() {
        return Err(ExprError::UnsupportedMetricLiteral);
    }

    Ok(MetricFilter {
        metric,
        cmp: op,
        threshold,
    })
}

/// Compile a metadata column comparison.
///
/// Type rules per column data type:
/// - String: only `Eq`/`Neq`; literal must be a string; other ops => `UnsupportedStringOp`.
/// - Int32/Int64: literal must be `i64`; floats/strings => `TypeMismatch`.
/// - Float32/Float64: literal may be `f64` or `i64` (widened to f64); strings => `TypeMismatch`.
/// - DateTime: literal must be a parseable datetime string; stored as i64 millis; others => `TypeMismatch`.
fn compile_column_cmp(
    col_name: &str,
    lit: Literal,
    op: CmpOp,
    schema: &HashMap<String, DataType>,
) -> Result<ColumnFilter, ExprError> {
    let dtype = schema
        .get(col_name)
        .ok_or_else(|| ExprError::UnknownColumn(col_name.to_string()))?;
    let column_name = col_name.to_string();

    match dtype {
        DataType::String => {
            // Only Eq / Neq allowed
            let cmp = match op {
                CmpOp::Eq => CmpOp::Eq,
                CmpOp::Neq => CmpOp::Neq,
                _ => return Err(ExprError::UnsupportedStringOp(column_name)),
            };
            let rhs = match lit {
                Literal::Str(s) => s,
                Literal::I64(_) | Literal::F64(_) => {
                    return Err(ExprError::TypeMismatch(
                        column_name.clone(),
                        *dtype,
                        "string",
                    ));
                }
            };
            Ok(ColumnFilter::String {
                column: column_name,
                cmp,
                rhs,
            })
        }
        DataType::Int32 | DataType::Int64 => {
            // Numeric integral literal only
            let rhs = match lit {
                Literal::I64(v) => NumericLiteral::I64(v),
                Literal::F64(_) => {
                    return Err(ExprError::TypeMismatch(
                        column_name.clone(),
                        *dtype,
                        "float",
                    ));
                }
                Literal::Str(_) => {
                    return Err(ExprError::TypeMismatch(
                        column_name.clone(),
                        *dtype,
                        "string",
                    ));
                }
            };
            Ok(ColumnFilter::Numeric {
                column: column_name,
                cmp: op,
                rhs,
            })
        }
        DataType::DateTime => {
            // Accept only datetime-parseable string literals; store as i64 millis
            let millis = match lit {
                Literal::Str(s) => match try_parse_datetime_millis(&s) {
                    Some(ms) => ms,
                    None => {
                        return Err(ExprError::TypeMismatch(
                            column_name.clone(),
                            *dtype,
                            "datetime string",
                        ));
                    }
                },
                Literal::I64(_) | Literal::F64(_) => {
                    return Err(ExprError::TypeMismatch(
                        column_name.clone(),
                        *dtype,
                        "datetime string",
                    ));
                }
            };
            Ok(ColumnFilter::Numeric {
                column: column_name,
                cmp: op,
                rhs: NumericLiteral::I64(millis),
            })
        }
        DataType::Float32 | DataType::Float64 => {
            // Numeric float literal (allow ints by widening)
            let rhs = match lit {
                Literal::I64(v) => NumericLiteral::F64(v as f64),
                Literal::F64(v) => NumericLiteral::F64(v),
                Literal::Str(_) => {
                    return Err(ExprError::TypeMismatch(
                        column_name.clone(),
                        *dtype,
                        "string",
                    ));
                }
            };
            Ok(ColumnFilter::Numeric {
                column: column_name,
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
