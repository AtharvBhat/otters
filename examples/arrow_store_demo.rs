//! Demo of Arrow-based unified store for vectors and metadata
//!
//! Run with: cargo run --example arrow_store_demo

use arrow::array::{Float32Array, Int64Array};
use arrow::util::pretty::print_batches;
use otters::expr::cosine;
use otters::prelude::*;
use otters::type_utils::DataType;
use std::collections::HashMap;

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
    let store = OttersStore::builder(dim)
        .with_vectors(vectors)
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
    for (i, inv_norm) in store.inv_norms_array().iter().enumerate().take(5) {
        match inv_norm {
            Some(value) => println!("  [{}]: {:.6}", i, value),
            None => println!("  [{}]: NULL", i),
        }
    }
    println!();

    // Build a query: cosine similarity > 0.75 and age >= 28, take top 3 matches
    let query_vector = vec![1.0, 0.0, 0.0, 0.0];
    let expr = cosine().gt(0.75) & col("age").gte(28);
    let mut schema = HashMap::new();
    schema.insert("age".to_string(), DataType::Int32);
    let compiled = expr.compile(&schema).unwrap();

    let results = store
        .query(query_vector)
        .filter(compiled)
        .take(3)
        .collect()?;

    println!("Top cosine matches (score >= 0.75 and age >= 28):");
    let score_col = results
        .batch
        .column_by_name("_score")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    let row_ids = results
        .batch
        .column_by_name("_row_id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let ages = results
        .batch
        .column_by_name("age")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap();
    let names = results
        .batch
        .column_by_name("name")
        .unwrap()
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap();

    for i in 0..results.batch.num_rows() {
        let score = score_col.value(i);
        let id = row_ids.value(i);
        let age = ages.value(i);
        let name = names.value(i);
        println!("  row {id}: {name} (age {age}) -> score {score:.4}");
    }

    println!("\n=== Demo Complete ===");

    Ok(())
}
