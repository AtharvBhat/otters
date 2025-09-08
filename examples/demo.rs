use otters::prelude::*;
use rand::random_range;

fn get_random_vec(dim: usize) -> Vec<f32> {
    let vec: Vec<f32> = (0..dim).map(|_| random_range(-1.0..1.0)).collect();
    vec
}

fn get_random_vectors(num_vecs: usize, dim: usize) -> Vec<Vec<f32>> {
    let vec: Vec<Vec<f32>> = (0..num_vecs).map(|_| get_random_vec(dim)).collect();
    vec
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_size: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "1000".to_string())
        .parse()
        .unwrap_or(1000);
    let dim: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "100".to_string())
        .parse()
        .unwrap_or(100);

    // Demo: MetaStore with chunked zone-map pruning + vector search
    // Build metadata columns equal in length to the vectors (online store items)

    // name: item_{i}
    let name_vals: Vec<Option<String>> = (0..n_size).map(|i| Some(format!("item_{i}"))).collect();

    // Use small chunk size and per-chunk distributions to demonstrate pruning clearly
    let prune_demo_chunk: usize = 128; // produces ~n_size / 128 chunks

    // price: expensive in even-numbered chunks, cheap in odd-numbered chunks
    let price_vals: Vec<Option<f64>> = (0..n_size)
        .map(|i| {
            let g = i / prune_demo_chunk;
            if g % 2 == 0 {
                Some(80.0 + (i % 20) as f64)
            } else {
                Some(10.0 + (i % 20) as f64)
            }
        })
        .collect();

    // manufacture date (mfg): earlier in even chunks, later in odd chunks
    let mfg_vals: Vec<Option<String>> = (0..n_size)
        .map(|i| {
            let g = i / prune_demo_chunk;
            if g % 2 == 0 {
                Some("2024-01-01".to_string())
            } else {
                Some("2024-07-01".to_string())
            }
        })
        .collect();

    // expiration date (exp): 2024-12-31 in even chunks, 2025-12-31 in odd chunks
    let exp_vals: Vec<Option<String>> = (0..n_size)
        .map(|i| {
            let g = i / prune_demo_chunk;
            if g % 2 == 0 {
                Some("2024-12-31".to_string())
            } else {
                Some("2025-12-31".to_string())
            }
        })
        .collect();

    // version: older in even chunks, newer in odd chunks
    let version_vals: Vec<Option<i32>> = (0..n_size)
        .map(|i| {
            let g = i / prune_demo_chunk;
            if g % 2 == 0 { Some(1) } else { Some(3) }
        })
        .collect();

    let columns = vec![
        Column::new("name", DataType::String).from(name_vals)?,
        Column::new("price", DataType::Float64).from(price_vals)?,
        Column::new("mfg", DataType::DateTime).from(mfg_vals)?,
        Column::new("exp", DataType::DateTime).from(exp_vals)?,
        Column::new("version", DataType::Int32).from(version_vals)?,
    ];

    let meta = MetaStore::from_columns(columns)
        .with_vectors(get_random_vectors(n_size, dim))
        .with_chunk_size(prune_demo_chunk)
        .build()?;
    println!("=== MetaStore built ===");
    meta.print_build_stats();

    // Show MetaStore head as ASCII table
    println!("\n=== MetaStore Head (ASCII table) ===");
    meta.head();

    // Note for users: this example hand-tunes per-chunk metadata to make pruning obvious.
    // Real-world datasets may not prune as strongly unless rows are clustered/sorted by
    // common filter columns. Consider pre-sorting or choosing chunk sizes accordingly.
    println!(
        "Note: example data is hand-tuned per chunk to clearly show pruning; real datasets may prune less unless clustered by filter columns.\n"
    );

    let meta_results = meta
        .query(get_random_vec(dim), Metric::Cosine)
        .meta_filter(
            // Prunes all even-numbered chunks (price high, exp in 2024, version 1)
            col("price").lt(50.0) & col("version").gte(2) & col("exp").gte("2025-01-01"),
    )
        .vec_filter(0.1, Cmp::Gt)
        .take(5)
        .collect()?;

    println!("\n=== Meta query top 5 (ASCII table) ===");
    // Pretty print results with metadata entries for the returned indices
    println!("{meta_results}");

    meta.print_last_query_stats();

    // Demonstrate accessing result contents: print heads of result columns
    println!("\n=== Access result columns (head) ===");
    if let Some(col) = meta_results.column("name") {
        col.head();
    }
    if let Some(col) = meta_results.column("price") {
        col.head();
    }
    if let Some(col) = meta_results.column("version") {
        col.head();
    }

    Ok(())
}
