//! Demo of Arrow-based unified store for vectors and metadata
//!
//! Run with: cargo run --example arrow_store_demo

use arrow::util::pretty::print_batches;
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

    // Build metadata columns
    let mut names_builder = ColumnBuilder::new_string("name");
    names_builder.append_string(Some("Alice"))?;
    names_builder.append_string(Some("Bob"))?;
    names_builder.append_string(Some("Charlie"))?;
    names_builder.append_string(Some("Diana"))?;
    names_builder.append_string(Some("Eve"))?;
    let names = names_builder.collect();

    let mut ages_builder = ColumnBuilder::new_int32("age");
    ages_builder.append_i32(Some(25))?;
    ages_builder.append_i32(Some(30))?;
    ages_builder.append_i32(Some(35))?;
    ages_builder.append_i32(Some(28))?;
    ages_builder.append_i32(Some(32))?;
    let ages = ages_builder.collect();

    let mut scores_builder = ColumnBuilder::new_float64("score");
    scores_builder.append_f64(Some(0.95))?;
    scores_builder.append_f64(Some(0.87))?;
    scores_builder.append_f64(Some(0.92))?;
    scores_builder.append_f64(Some(0.88))?;
    scores_builder.append_f64(Some(0.91))?;
    let scores = scores_builder.collect();

    // Build the Arrow store
    let store = ArrowStore::builder(dim)
        .with_vectors(vectors)?
        .with_metadata_column("name", names)
        .with_metadata_column("age", ages)
        .with_metadata_column("score", scores)
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

    // Pretty print the entire RecordBatch as a table
    println!("RecordBatch Contents:");
    println!("{}", "=".repeat(80));
    print_batches(&[store.batch().clone()]).unwrap();
    println!("{}", "=".repeat(80));
    println!();

    // Get specific vector
    println!("Vector at index 0:");
    let vectors_col = store.vectors();
    if let Some(vec) = vectors_col.vector_at(0) {
        println!("  {:?}", vec);
    }
    println!();

    // Access pre-computed inverse norms
    println!("Pre-computed inverse norms for cosine similarity:");
    for (i, inv_norm) in store.inv_norms().iter().enumerate().take(5) {
        println!("  [{}]: {:.6}", i, inv_norm);
    }
    println!();

    println!("=== Demo Complete ===");

    Ok(())
}
