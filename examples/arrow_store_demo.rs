//! Demo of Arrow-based unified store for vectors and metadata
//!
//! Run with: cargo run --example arrow_store_demo

use otters::expr::cosine;
use otters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Otters Arrow Store Demo ===\n");

    // Create some sample vectors (embeddings)
    let dim = 4;
    let vectors = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.5, 0.5, 0.0, 0.0],
        vec![0.0, 0.5, 0.5, 0.0],
    ];

    println!(
        "Creating store with {} vectors of dimension {}...",
        vectors.len(),
        dim
    );

    // Build metadata columns in bulk
    let mut names_builder = Column::new_string("name");
    names_builder.append([
        Some("Alice"),
        Some("Bob"),
        Some("Charlie"),
        Some("Diana"),
        Some("Eve"),
    ]);
    let names = names_builder.collect()?;

    let mut ages_builder = Column::new_int32("age");
    ages_builder.append([Some(25), Some(30), Some(35), Some(28), Some(32)]);
    let ages = ages_builder.collect()?;

    let mut vector_builder = Column::new_vector("embedding", dim);
    for vec in &vectors {
        vector_builder.append(Some(vec.as_slice()));
    }
    let embeddings = vector_builder.collect()?;

    // Build the Arrow store
    let store = OttersStore::new(["name", "age", "embedding"], [names, ages, embeddings])
        .with_embedding_column("embedding")
        .build()?;

    println!("Store created successfully!\n");

    // Display store info
    println!("Store Information:");
    println!("  Rows: {}", store.len());
    println!("  Vector dimension: {}", store.dim());
    println!("  Columns: {:?}", store.column_names());
    println!("  Metadata columns: {:?}\n", store.metadata_columns());

    // Display schema
    println!("Schema:");
    println!("{}", store.schema());
    println!();

    println!("Table snapshot:\n{store}");

    // Demonstrate adapting an existing Arrow RecordBatch into a new OttersStore
    println!("Reconstructing store from existing RecordBatch...");
    let batch = store.to_recordbatch().expect("store already built");
    let restored = OttersStore::from_recordbatch(batch)
        .with_embedding_column(store.vector_column_name())
        .build()?;
    println!("Restored store schema:\n{}", restored.schema());
    println!();

    // Get specific vector
    println!("Vector at index 0:");
    let vectors_col = store.vectors();
    if let Some(vec) = vectors_col.vector_at(0) {
        println!("  {vec:?}");
    }
    println!();

    // Access pre-computed inverse norms
    if let Some(inv_norms) = store.column(store.inv_norm_column_name()) {
        println!("Pre-computed inverse norms for cosine similarity:\n{inv_norms}");
        println!();
    }

    // Build a query: cosine similarity > 0.75 and age >= 28, take top 3 matches
    let query_vector = vec![1.0, 0.0, 0.0, 0.0];
    let results = store
        .query(query_vector)
        .filter(cosine().gt(0.7) & col("age").gte(28))
        .take(3)
        .collect()?;

    println!("Top cosine matches (cosine >= 0.7 and age >= 28):\n{results}");

    if let Some(stats) = store.get_last_query_stats() {
        println!("\nLast query timings:");
        println!("{stats}");
    }

    println!("\n=== Demo Complete ===");

    Ok(())
}
