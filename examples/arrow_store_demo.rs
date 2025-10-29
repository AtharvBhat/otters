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
    let mut names_builder = ColumnBuilder::new_string("name");
    names_builder.append([
        Some("Alice"),
        Some("Bob"),
        Some("Charlie"),
        Some("Diana"),
        Some("Eve"),
    ])?;
    let names = names_builder.collect();

    let mut ages_builder = ColumnBuilder::new_int32("age");
    ages_builder.append([Some(25), Some(30), Some(35), Some(28), Some(32)])?;
    let ages = ages_builder.collect();

    // Build the Arrow store
    let store = OttersStore::builder(dim)
        .with_vectors(vectors)
        .with_metadata_column("name", names)
        .with_metadata_column("age", ages)
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
    let schema = store.schema();
    for field in schema.fields() {
        println!("  - {} ({:?})", field.name(), field.data_type());
    }
    println!();

    println!("Table snapshot:\n{store}");

    // Get specific vector
    println!("Vector at index 0:");
    let vectors_col = store.vectors();
    if let Some(vec) = vectors_col.vector_at(0) {
        println!("  {vec:?}");
    }
    println!();

    // Access pre-computed inverse norms
    println!("Pre-computed inverse norms for cosine similarity:");
    for (i, inv_norm) in store.inv_norms_array().iter().enumerate().take(5) {
        match inv_norm {
            Some(value) => println!("  [{i}]: {value:.6}"),
            None => println!("  [{i}]: NULL"),
        }
    }
    println!();

    // Build a query: cosine similarity > 0.75 and age >= 28, take top 3 matches
    let query_vector = vec![1.0, 0.0, 0.0, 0.0];
    let results = store
        .query(query_vector)
        .filter(cosine().gt(0.75) & col("age").gte(28))
        .take(3)
        .collect()?;

    println!("Top cosine matches (cosine >= 0.75 and age >= 28):\n{results}");

    println!("\n=== Demo Complete ===");

    Ok(())
}
