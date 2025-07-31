# Otters: Vector Search Engine Design Document

## Overview

Otters is a high-performance vector search engine with a Polars-like lazy evaluation API. It targets the "missing middle" - applications with up to ~1M vectors that need better performance than naive search but don't require the complexity of a full vector database.

The API is inspired by Polars' lazy evaluation model: build a query plan with method chaining, then execute with `collect()`. This enables query optimization and plan reusability while keeping the API intuitive.

## Core Design Principles

1. **Simplicity First**: Pure brute-force search with SIMD optimization - no approximate algorithms
2. **Lazy Evaluation**: Operations build a query plan that executes only on `collect()`
3. **Cache-Efficient**: Vectors are stored contiguously for optimal memory access patterns
4. **Extensible**: API designed to compose well with metadata filtering
5. **Type-Safe**: Single queries return `Vec<(usize, f32)>`, batch queries return `Vec<Vec<(usize, f32)>>`
6. **Language Agnostic**: Core Rust implementation with Python bindings
7. **Focused Scope**: Optimized for up to ~1M vectors - beyond that, use a full vector database

### Lazy Query Planning

Similar to Polars and Spark, Otters uses lazy evaluation to optimize query execution:

```rust
// This doesn't execute anything - just builds a query plan
let plan = metadata_index
    .m_filter(col("price") > 10 & col("category").eq("electronics"))
    .query(query_vec, SearchType::Cosine)
    .v_filter(score > 0.8)
    .top(5);

// Only when collect() is called does execution happen
let results = plan.collect();
```

Benefits of lazy evaluation:
- **Query optimization**: The planner can analyze the entire query before execution
- **Efficient execution**: Can determine optimal B-tree traversal path
- **Composability**: Plans can be built dynamically and reused
- **Future optimizations**: Could add query caching, parallel execution strategies, etc.

Why lazy evaluation matters for vector search:
- **Metadata filtering is expensive**: B-tree traversal can be optimized if you know all filters upfront
- **Memory allocation**: Knowing the full query helps pre-allocate result buffers
- **Parallel strategy**: Can decide whether to parallelize based on estimated result size
- **Common query patterns**: Many applications search the same metadata filters repeatedly

## Current Architecture (MVP)

### Vector Storage

- Vectors stored in `RowAlignedVecs` structure
- Contiguous memory layout for SIMD operations
- Pre-computed inverse norms for cosine similarity
- Single implementation handles both single and batch queries (batch size = 1 for single)

### API Design

```rust
// Single query
vectors
    .search(query_vec, SearchType::Cosine)
    .top(5)
    .collect()

// Batch queries
vectors
    .search(query_vecs, SearchType::Cosine)
    .top(5)
    .collect()

// Alternative: global top-k across all queries in batch
vectors
    .search(query_vecs, SearchType::Cosine)
    .top_global(5)
    .collect()
```

### Query Plan Structure

- Each method returns a new query plan object (not results!)
- `search()` creates a query plan but doesn't execute
- `m_filter()` returns a plan with metadata filters added
- `v_filter()` returns a plan with score filters added  
- `top(k)` returns a plan with k-selection added (automatically chooses max/min based on search type)
- `collect()` executes the plan and returns actual results
- Plans are immutable - each method returns a new plan

Example:
```rust
let plan1 = vectors.search(query, SearchType::Cosine);  // SearchPlan
let plan2 = plan1.top(10);                              // SearchPlan with top-k
let results = plan2.collect();                          // Vec<(usize, f32)> - actual results!

// Or chained:
let results = vectors
    .search(query, SearchType::Cosine)  // Returns SearchPlan
    .top(10)                            // Returns SearchPlan  
    .collect();                         // Returns Vec<(usize, f32)>
```

### Query Plan Reusability

Plans can be built once and executed multiple times with different query vectors:

```rust
// Build a reusable plan for electronics under $100
let electronics_plan = index
    .m_filter(col("category").eq("electronics") & col("price") < 100)
    .top(20);

// Execute with different query vectors
let results1 = electronics_plan.query(laptop_embedding, SearchType::Cosine).collect();
let results2 = electronics_plan.query(phone_embedding, SearchType::Cosine).collect();

// Or execute the same plan periodically
for query in incoming_queries {
    let results = electronics_plan.query(query.embedding, SearchType::Cosine).collect();
    // Process results...
}
```

### Search Types

- **Cosine Similarity**: Normalized dot product (higher scores = more similar)
- **Euclidean Distance**: L2 distance squared (lower scores = more similar)

## Future Architecture (With Metadata)

### Metadata Index Structure

```
B-tree Index (on metadata columns)
    ├── Node: category="electronics"
    │   ├── Node: price<100
    │   │   └── Leaf: RowAlignedVecs instance (vectors with these properties)
    │   └── Node: price>=100
    │       └── Leaf: RowAlignedVecs instance
    └── Node: category="books"
        └── Leaf: RowAlignedVecs instance
```

### Key Design Decisions

1. **Each B-tree leaf owns its own `RowAlignedVecs` instance**
   - Perfect memory locality for vectors with similar metadata
   - Natural parallelism (each leaf can be searched independently)
   - No need for views or slices
   - Simple brute-force search is fast enough with good data locality

2. **Vectors and metadata are coupled at creation**
   - MetadataIndex takes both vectors and metadata together
   - Metadata structured like Polars DataFrame with named columns
   - Enables expression-based filtering API

3. **Separate engines with clear responsibilities**
   - Vector engine: Similarity search only
   - Metadata engine: Indexing, filtering, and orchestration

4. **Two types of filtering**
   - `m_filter`: Metadata filtering during index traversal (pre-filtering)
   - `v_filter`: Score-based filtering after vector search (post-filtering)

### Extended API Design

```rust
// Creating metadata index - vectors and metadata coupled together
let metadata_index = MetadataIndex::new(vectors, metadata);

// Entry point via metadata index when filtering needed
metadata_index
    .m_filter(col("price") > 10 & col("category").eq("electronics"))
    .query(query_vec, SearchType::Cosine)
    .v_filter(score > 0.8)
    .top(5)
    .collect()

// Direct vector search still available when no metadata filtering needed
vectors
    .search(query_vec, SearchType::Cosine)
    .top(5)
    .collect()
```

### Execution Flow

#### Planning Phase (Lazy)
When methods are chained, no computation happens - only a query plan is built:

1. **m_filter**: Adds metadata predicates to the plan
2. **query**: Adds vector search operation with query vector and search type
3. **v_filter**: Adds score filtering predicate
4. **top**: Adds k-selection operation

The plan tracks:
- Metadata filter expression tree
- Query vector(s) and search type
- Score thresholds
- Result limits

#### Execution Phase (collect)
When `collect()` is called, the plan executes:

1. **Analyze metadata filters**: Determine which B-tree nodes to visit
2. **Traverse B-tree**: Find all leaf nodes matching metadata predicates
3. **Parallel vector search**: Search relevant `RowAlignedVecs` instances in parallel
4. **Apply score filter**: Keep only results meeting score threshold
5. **Merge and sort**: Combine results from all leaves
6. **Select top-k**: Return final results

Example execution:
```rust
// Planning: O(1) - just building the plan
let plan = index
    .m_filter(col("category").eq("shoes") & col("in_stock").eq(true))
    .query(embedding, SearchType::Cosine)
    .top(10);

// Can inspect plan (future feature)
println!("Plan will search approximately {} vectors", plan.estimate_vectors());

// Execution: O(n) where n = vectors in matching leaves
let results = plan.collect();  // <- All computation happens here
```

### Benefits of This Design

1. **Cache Efficiency**: Vectors likely to be searched together are stored together
2. **Parallelism**: Each leaf can be searched independently
3. **Clean Separation**: Vector search doesn't need to know about metadata
4. **Simplicity**: No complex graph structures or index building - just fast SIMD search
5. **Language Accessibility**: Fast Rust core with Python bindings for ML practitioners
6. **Lazy Evaluation Benefits**:
   - Query optimization before execution
   - Plan reusability for common query patterns  
   - Future optimizations possible (caching, better parallelism strategies)
   - Clear separation between plan building (cheap) and execution (expensive)
7. **Optimization Opportunities**: 
   - Early termination based on score thresholds
   - Parallel execution across leaves
   - CPU-only design eliminates GPU transfer overhead

## Implementation Phases

### Phase 1 (Current MVP)
- Single unified SIMD-optimized search function
- Lazy evaluation with query plans
- Support for single and batch queries
- Top-k selection with automatic max/min based on search type

### Phase 2 (Metadata Integration)
- Metadata index with B-tree structure
- Polars-like expression API for metadata filtering
- Integration between metadata and vector search
- Parallel execution across index leaves

### Phase 3 (Python Bindings)
- PyO3-based Python bindings
- Zero-copy numpy array support (Rust owns data after insertion)
- Pythonic API design
- Integration with common ML frameworks

## Future Enhancements

### Python Bindings

- Expose the Rust API through PyO3
- Rust owns all vector data after insertion (like Polars)
- Automatic f32 conversion from numpy arrays
- Pythonic API that feels natural while maintaining performance

```python
# Example Python API
import otters

# Create index
vectors = otters.RowAlignedVecs(dim=768)
vectors.add_vectors(embeddings)  # numpy array - data is copied and owned by Rust

# Search - lazy evaluation
plan = vectors.search(query_embedding, search_type="cosine").top(5)
# Nothing computed yet!
results = plan.collect()  # <- Execution happens here

# With metadata (future)
# metadata is like a polars DataFrame with named columns
index = otters.MetadataIndex(embeddings, metadata)  # vectors and metadata coupled together

# Build complex query plan
plan = (index
    .m_filter(otters.col("price") > 10 & otters.col("category").eq("electronics"))
    .query(query_embedding, search_type="cosine")
    .top(5))

# Still nothing executed!
print(f"Plan will search ~{plan.estimate_vectors()} vectors")  # Future feature

# Execute the plan
results = plan.collect()  # <- All computation happens here
```

## Open Questions for Future

1. How to handle vector updates that change metadata (rebalancing)?
2. How to handle multi-modal search (text + image vectors)?
3. Should metadata indices support compound keys?
4. Optimal B-tree node size for cache efficiency?
5. Should we support incremental index updates or require rebuild?

### Future Query Plan Optimizations

With lazy evaluation, several optimizations become possible:

1. **Predicate pushdown**: Analyze filters to minimize B-tree traversal
2. **Result size estimation**: Predict result sizes for better memory allocation
3. **Adaptive parallelism**: Choose parallel vs sequential execution based on estimated work
4. **Query plan caching**: Cache frequently used query patterns
5. **Short-circuit evaluation**: Stop searching once enough high-scoring results are found

## Target Use Cases

Otters is designed for the "missing middle" - applications that need better than naive search but don't require a full vector database:

### When to use Otters:
- **Dataset size**: Up to ~1M vectors
- **Use cases**: Semantic search, recommendation systems, similarity matching
- **Requirements**: Exact results, low latency, metadata filtering
- **Performance**: 40ms for 1M vectors on CPU - no GPU needed
- **Examples**: E-commerce product search, document retrieval for a team, image similarity for a photo app

### When to use a full vector database:
- **Dataset size**: 10M+ vectors
- **Requirements**: Approximate search is acceptable, distributed scale
- **Examples**: Web-scale search, billion-scale recommendation systems

## Example Use Cases

### E-commerce Product Search
```rust
// Find similar products in a price range
product_index
    .m_filter(col("price").between(50, 200) & col("category").eq("shoes"))
    .query(image_embedding, SearchType::Cosine)
    .top(10)
    .collect()
```

### Document Retrieval
```rust
// Find recent documents similar to query
document_index
    .m_filter(col("date") > "2024-01-01")
    .m_filter(col("department").eq("engineering"))
    .query(text_embedding, SearchType::Cosine)
    .v_filter(score > 0.7)
    .top(20)
    .collect()
```

### Multi-tenant Search
```rust
// Search within a specific tenant's data
index
    .m_filter(col("tenant_id").eq(tenant_id))
    .query(query_vec, SearchType::Euclidean)
    .top(5)
    .collect()
```

### Demonstrating Lazy Evaluation
```rust
// Build expensive query plan once
let premium_electronics = index
    .m_filter(
        col("category").eq("electronics") & 
        col("price") > 1000 & 
        col("brand").is_in(["Apple", "Sony", "Samsung"])
    )
    .v_filter(score > 0.85)
    .top(20);

// Reuse for multiple queries - plan analysis happens once
for customer_query in customer_queries.iter() {
    // Only vector search is executed, metadata filtering is pre-computed
    let recommendations = premium_electronics
        .query(&customer_query.embedding, SearchType::Cosine)
        .collect();
    
    send_recommendations(customer_query.id, recommendations);
}
```

## Performance Characteristics

### Brute Force (RowAlignedVecs)
- **Search complexity**: O(n) where n = vectors in leaf
- **Benefits**: Exact results, simple implementation, predictable performance
- **SIMD optimized**: Processes 8 floats at a time
- **Cache efficient**: Contiguous memory layout
- **Target scale**: Up to ~1M vectors total
- **CPU-only**: 40ms for 1M vectors (512-dim) - GPU transfer overhead would exceed compute gains

### With Metadata Indexing
- **Pre-filtering**: O(log m) B-tree traversal where m = metadata entries
- **Parallelism**: Search multiple leaves simultaneously
- **Memory**: Each leaf maintains its own contiguous vector storage
- **Cache locality**: Vectors frequently searched together are stored together

### Python Bindings
- **Data ownership**: Rust owns all data after insertion (like Polars)
- **Type conversion**: Automatic conversion to f32
- **GIL released**: During search operations for true parallelism
- **Overhead**: Minimal - most time spent in Rust code

## Complete Vision

Otters creates a vector search engine that:
1. **Solves the missing middle**: Perfect for applications with up to ~1M vectors
2. **Keeps it simple**: Pure brute-force search with SIMD optimization
3. **Integrates metadata**: Pre-filtering for cache-efficient search
4. **Supports Python**: First-class bindings for ML practitioners
5. **Stays focused**: Not trying to replace full vector databases
6. **CPU-only by design**: Fast enough that GPU overhead isn't worth it

The design philosophy is to do one thing extremely well: provide blazing-fast exact vector search for the 99% of use cases that don't need massive scale. By combining smart data layout (metadata-based partitioning) with SIMD optimization and parallel execution, Otters can outperform more complex solutions in its target domain.