# otters 🦦

[![Crates.io](https://img.shields.io/crates/v/otters-rs.svg)](https://crates.io/crates/otters-rs)
[![Docs.rs](https://docs.rs/otters-rs/badge.svg)](https://docs.rs/otters-rs)
[![CI](https://github.com/AtharvBhat/otters/actions/workflows/rust.yml/badge.svg)](https://github.com/AtharvBhat/otters/actions/workflows/rust.yml)

Otters is a minimal, exact vector search library with expressive metadata filtering. Think “Polars for vector search.”

Otters targets smaller to mid-size datasets (up to ~10M vectors) where:
- You want exact results, not approximate indices.
- You care about not just vector search but also rich metadata filtering.
- All in memory without needing a full database.

The design leans on chunked zonemaps (min/max/null counts + light Bloom filters) to prune work early, then runs tight SIMD loops for scoring on the surviving chunks.

## Quick Start

```rust,ignore
use otters::prelude::*;

// Basic vector search
let mut store = VecStore::new(128);
store.add_vectors(my_vectors)?;

let results = store
    .query(query_vec, Metric::Cosine)
    .filter(0.8, Cmp::Gt) // only results with similarity > 0.8
    .take(10)
    .collect()?;

// Build a MetaStore for metadata + vector pruning
let columns = vec![
    Column::new("item", DataType::String).from(item_vals)?,
    Column::new("price", DataType::Float64).from(price_vals)?,
];

let meta = MetaStore::from_columns(columns)
    .with_vectors(my_vectors)
    .with_chunk_size(1024)
    // .with_bloom_bits(4096) to explicitly size bloom filter to 4096 bits
    .build()?;

// Metadata + vector query with stats
use otters::expr::col;
let top5 = meta
    .query(query_vec, Metric::Cosine)
    .meta_filter(col("item").eq("rust") & col("price").gt(100.0))
    .vec_filter(0.8, Cmp::Gt)
    .take(5)
    .collect()?;

meta.print_last_stats();
```

## Example:

Otters implements `Display` for result sets and prints MetaStore heads and stats as ASCII tables. Here’s a compact, deterministic example:

```rust,ignore
use otters::prelude::*;

// Small item catalog (8 rows, 4 dims)
let vectors = vec![
    vec![1.0, 0.0, 0.0, 0.0], // 0
    vec![0.0, 1.0, 0.0, 0.0], // 1
    vec![1.0, 1.0, 0.0, 0.0], // 2
    vec![0.0, 0.0, 1.0, 0.0], // 3
    vec![0.8, 0.2, 0.0, 0.0], // 4
    vec![0.0, 0.0, 0.0, 1.0], // 5
    vec![0.6, 0.6, 0.0, 0.0], // 6
    vec![0.0, 0.5, 0.5, 0.0], // 7
];

let names = Column::new("name", DataType::String).from(vec![
    Some("widget"), Some("gizmo"), Some("adapter"), Some("battery"),
    Some("charger"), Some("cable"), Some("dock"), Some("earbuds"),
])?;
let prices = Column::new("price", DataType::Float64)
    .from(vec![Some(19.99), Some(49.00), Some(12.50), Some(8.99), Some(29.99), Some(5.99), Some(39.50), Some(59.99)])?;
let mfg = Column::new("mfg", DataType::DateTime).from(vec![
    Some("2024-01-05"), Some("2024-01-10"), Some("2024-02-15"), Some("2024-03-01"),
    Some("2024-03-20"), Some("2024-04-05"), Some("2024-05-01"), Some("2024-05-12"),
])?;
let exp = Column::new("exp", DataType::DateTime).from(vec![
    Some("2025-01-05"), Some("2024-12-31"), Some("2024-10-01"), Some("2024-06-01"),
    Some("2025-06-01"), Some("2024-08-01"), Some("2025-01-01"), Some("2024-12-01"),
])?;
let version = Column::new("version", DataType::Int32)
    .from(vec![Some(1), Some(2), Some(2), Some(1), Some(3), Some(1), Some(2), Some(3)])?;

let meta = MetaStore::from_columns(vec![names, prices, mfg, exp, version])
    .with_vectors(vectors)
    .with_chunk_size(4)
    .build()?;

// Head (first 5 rows) as ASCII table
meta.head();

// Query similar items, price <= 40, version >= 2, fresh
let results = meta
    .query(vec![1.0, 0.0, 0.0, 0.0], Metric::Cosine)
    .meta_filter(
        col("price").lte(40.0) 
        & col("version").gte(2) 
        & col("mfg").gte("2024-01-01") 
    & col("exp").gte("2024-06-01"))
    .take(5)
    .collect()?;

// Pretty-print results with metadata columns
println!("{}", results);
meta.print_last_query_stats();
```

Sample output:

```text
MetaStore Head • rows=8 • chunks=2 • chunk_size=4
+-------+-------------------------+-------------------------+---------+---------+---------+
| index | exp                     | mfg                     | name    | price   | version |
+-------+-------------------------+-------------------------+---------+---------+---------+
| 0     | 2025-01-05 00:00:00 UTC | 2024-01-05 00:00:00 UTC | widget  | 19.9900 | 1       |
| 1     | 2024-12-31 00:00:00 UTC | 2024-01-10 00:00:00 UTC | gizmo   | 49.0000 | 2       |
| 2     | 2024-10-01 00:00:00 UTC | 2024-02-15 00:00:00 UTC | adapter | 12.5000 | 2       |
| 3     | 2024-06-01 00:00:00 UTC | 2024-03-01 00:00:00 UTC | battery | 8.9900  | 1       |
| 4     | 2025-06-01 00:00:00 UTC | 2024-03-20 00:00:00 UTC | charger | 29.9900 | 3       |
+-------+-------------------------+-------------------------+---------+---------+---------+

Query Results
+-------+----------+-------------------------+-------------------------+---------+---------+---------+
| index | score    | exp                     | mfg                     | name    | price   | version |
+-------+----------+-------------------------+-------------------------+---------+---------+---------+
| 4     | 0.970142 | 2025-06-01 00:00:00 UTC | 2024-03-20 00:00:00 UTC | charger | 29.9900 | 3       |
| 2     | 0.707107 | 2024-10-01 00:00:00 UTC | 2024-02-15 00:00:00 UTC | adapter | 12.5000 | 2       |
| 6     | 0.707107 | 2025-01-01 00:00:00 UTC | 2024-05-01 00:00:00 UTC | dock    | 39.5000 | 2       |
+-------+----------+-------------------------+-------------------------+---------+---------+---------+

Last Query Stats
+------------------+-------+
| metric           | value |
+------------------+-------+
| total_chunks     | 2     |
| pruned_chunks    | 0     |
| evaluated_chunks | 2     |
| vectors_compared | 8     |
| prune_ms         | 0.002 |
| score_ms         | 0.031 |
| merge_ms         | 0.000 |
| total_ms         | 0.032 |
+------------------+-------+

```

Note on pruning: this example intentionally hand-tunes per-chunk metadata distributions (e.g., prices, versions, dates) and uses a small chunk size to make pruning visible in the stats. Real-world datasets are often not clustered by filter columns, so pruning may be weaker unless you pre-sort or naturally ingest data in a way that groups similar values within chunks. Choosing an appropriate chunk size and sorting on common filter columns can significantly improve pruning effectiveness. I plan to add features to reorder data for better pruning in future releases.

## Architecture

- VecStore: row‑major f32 vectors with SIMD kernels for scoring. Supports cosine, dot product, and squared euclidean.
- MetaStore: wraps vectors in fixed‑size chunks and builds per‑chunk zonemaps:
  - Numeric: min, max, and non‑null counts for fast range pruning.
  - String: small Bloom filter per chunk for equality pruning.
- Query plan: combines an expression tree for metadata (AND/OR across leaves) with vector scoring and optional row masks

## Zonemaps & Bloom Filters

- Numeric pruning: compare the predicate against per‑chunk min/max (respecting null‑only chunks) to skip entire chunks.
- String pruning: per‑chunk Bloom filters enable `col("s").eq("value")` to drop chunks that can’t possibly contain the value (false positives may pass through; no false negatives).

### Bloomfilter configuration

You can size Bloom filters in either of two ways on the builder:
- `with_bloom_fpr(fpr)`: target false‑positive rate (0 < fpr < 1). Default is 0.01.
- `with_bloom_bits(bits)`: set the total number of bits allocated for the filter.

Under the hood, string zonemaps use `fastbloom` and construct filters with either
`BloomFilter::with_false_pos(fpr)` or `BloomFilter::with_num_bits(bits)`.

## Chunk Size Trade‑offs

Chunking affects both pruning power and compute overhead:
- Smaller chunks: better pruning (tighter ranges, smaller blooms), but more chunk bookkeeping.
- Larger chunks: fewer boundaries to manage, but coarser ranges and weaker pruning.

Guidance:
- Sorting your data by common filter columns before ingest can improve pruning effectiveness.
- Start with 512–2048 depending on data distribution and the selectivity of predicates.

## Expression API (metadata)
- Rich Expressions allow you to filter by metadata.

```rust
use otters::prelude::*;

// Examples
let e1 = col("age").gt(25) & col("score").gte(80.0);
let e2 = (col("age").lt(18) | col("age").gt(65)) & col("name").neq("alice");
let e3 = col("grade").eq("A") | col("grade").eq("B");
```

## Status and stability

This project is early-stage. Expect frequent breaking changes
- Current pre-release: 0.1.0-alpha1.
- MSRV: 1.88.

## Roadmap
- Test with real datasets
- Persistence (save/load MetaStore to/from disk)
- Mutability (add/remove rows after build)
- Quantization for vectors
- More Metrics (Manhattan, Hamming, Jaccard)
- More features to handle string columns (e.g. `contains`, `starts_with`, `ends_with` or fuzzy matching)
- More Metadata Types and Filters
- Ability to reorder metadata for better pruning ( Something like Z-ordering )
- Integration with Parquet/Arrow formats
- Python bindings
