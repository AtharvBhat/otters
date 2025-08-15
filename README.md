# otters 🦦

Minimal, exact vector search with metadata filtering. Think "Polars for vector search."

I needed a simple vector search tool for smaller-scale projects, but most vector databases felt like overkill and complex to deploy and maintain for what I actually needed. I also wanted to dive deeper into Rust, so I decided to build exactly the tool I was looking for.

I love Polars and its ergonomic API, so I wanted to adapt a similar approach for vector search—something that feels natural and expressive. Otters is inspired heavily by Polars.

Good metadata filtering is hard to combine with vector indexing like HNSW. You can either have:
- A vector index with bad interoperability with metadata filtering, or  
- Good metadata filtering and column indexing with bad vector search that hurts memory locality

So I decided not to index at all. Instead, I believe that using clever techniques like zonemaps for metadata filtering and leveraging memory locality with vectorized SIMD instructions should make it possible to achieve very fast vector search and metadata filtering for most use cases.

Otters is designed for smaller datasets (~10M vectors) where you want fast, exact vector search with metadata filtering, all running efficiently in memory and no need for the complexity of full vector databases or approximate search indices.

Otters is meant to be a focused library with an ergonomic, expression based query API that makes vector search feel as natural as working with Polars data frames.

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

// You can also prebuild a query plan and use it with a vector store
let closest_5_query = VecQueryPlan::new()
    .with_query_vectors(vec![get_random_vec(dim), get_random_vec(dim)])
    .with_metric(Metric::Euclidean)
    .filter(400.0, Cmp::Lt)
    .take_global(5);

let closest_5 = closest_5_query.with_vector_store(&store).collect()?;

// You can also prebuild a query plan without query vectors
// and provide them later when available
let farthest_5_query = VecQueryPlan::new()
    .with_vector_store(&store)
    .with_metric(Metric::Cosine)
    .take_min(5);

// Add query vectors and execute when ready
let farthest_5 = farthest_5_query
    .with_query_vectors(get_random_vec(dim))
    .collect()?;
```

## What's Next
- Metadata filtering integration with vector search
- Python bindings (PyO3)

## Examples
- Demo: `src/main.rs`
