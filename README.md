# otters 🦦

Minimal, exact vector search with metadata filtering. Think "Polars for vector search."

A learning project exploring Rust performance patterns, SIMD, and composable APIs.

- Brute-force SIMD search (no indexing)
- Composable query API with filtering and top-k selection
- Expression compiler for metadata prefiltering (WIP)

**Exact, minimal, in-memory vector search with metadata filtering capabilities.**

Think "Polars for vector search" — a fast, ergonomic toolkit for vector search and metadata filtering, not a full vector database.

## 🎯 Project Goals
- **Exact search**: No indexing or approximate algorithms. SIMD-accelerated exact search only.
- **Metadata filtering**: Zone-map style prefiltering with a minimal expression compiler.
- **Ergonomic API**: Clean, composable interface inspired by data processing libraries like Polars.
- **Minimal footprint**: Small, focused codebase.

## ✅ What's Implemented

### Vector Search (`VecStore`)
- **Storage**: Add vectors to an in-memory store with configurable dimensions
- **Metrics**: Cosine similarity and Euclidean distance with SIMD acceleration  
- **Filtering**: Threshold-based filtering (`filter(threshold, comparison)`)
- **Selection**: Top-k selection with various strategies:
  - `take(k)` - top-k results for single queries
  - `take_min(k)` / `take_max(k)` - smallest/largest k values
  - `take_global(k)` - global top-k across batch queries
- **Batch queries**: `VecQueryPlan` builder for processing multiple query vectors ( Batch querying isnt optimal rn :( )

### Expression System
- **Typed DSL**: Build expressions with `col("name").gt(value)` syntax
- **Schema validation**: Type checking against column schemas (Int32/64, Float32/64, String, DateTime)
- **Compilation**: Expressions compile to `Plan = Vec<Vec<ColumnFilter>>` (AND-of-ORs structure)

### Metadata Column Storage (Demo/Testing)
- **Typed columns**: Support for integers, floats, strings, and datetime values
- **Data loading**: `push()` individual values or `from()` bulk data
- **Display utilities**: `head()` and `head_n()` for inspection

## Usage

```rust
use otters::prelude::*;

// Basic search
let mut store = VecStore::new(128);
store.add_vectors(my_vectors)?;

let results = store
    .query(query_vec, Metric::Cosine)
    .filter(0.5, Cmp::Gt)
    .take(10)
    .collect()?;

// Reusable query plans
let search_plan = VecQueryPlan::new()
    .with_metric(Metric::Euclidean)
    .filter(100.0, Cmp::Lt)
    .take(5);

// Use with different vectors/stores
let results1 = search_plan
    .with_query_vectors(vec![query1, query2])
    .with_vector_store(&store1)
    .collect()?;

let results2 = search_plan
    .with_query_vectors(user_query)
    .with_vector_store(&store2)  
    .collect()?;
```

## What's Next
- Metadata filtering integration with vector search
- Zone maps for fast prefiltering  
- Python bindings (PyO3)

## Examples
- Demo: `src/main.rs`
