//! Simple throughput demo: treat one train shard as a test set and measure
//! query performance against the remaining DBPedia shards.
//!
//! Usage:
//! ```text
//! cargo run --example dbpedia_query_demo [dataset_dir] [top_k]
//! ```
//! Defaults assume the dataset lives at
//! `/home/atharvbhat/dbpedia-entities-openai-1M` and `top_k = 5`.

use glob::glob;
use otters::prelude::*;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::env;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_DATASET_DIR: &str = "/home/atharvbhat/dbpedia-entities-openai-1M";
const DEFAULT_TOP_K: usize = 5;
const EMBEDDING_COLUMN: &str = "openai";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DATASET_DIR));
    let top_k = env::args()
        .nth(2)
        .as_deref()
        .map(|raw| raw.parse::<usize>())
        .transpose()
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?
        .unwrap_or(DEFAULT_TOP_K);

    if top_k == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "top_k must be greater than zero",
        )
        .into());
    }

    println!("=== DBPedia Parquet Throughput Demo ===");
    println!("Dataset directory: {}", dataset_dir.display());
    println!("top_k: {top_k}\n");

    let mut shards = find_train_shards(&dataset_dir)?;
    if shards.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "Expected at least two Parquet shards in {}, found {}",
                dataset_dir.display(),
                shards.len()
            ),
        )
        .into());
    }

    shards.sort();
    let train_shard = shards[0].clone();
    let test_shard = shards[1].clone();

    println!("Using {} as the train split.", train_shard.display());
    println!("Using {} as the test split.", test_shard.display());
    if shards.len() > 2 {
        println!(
            "Skipping remaining {} shards for this run.\n",
            shards.len() - 2
        );
    } else {
        println!();
    }

    let train_store = load_store(&[train_shard.clone()])?;
    println!(
        "Train store ready: {} rows, dim {}\n",
        train_store.len(),
        train_store.dim()
    );

    let test_store = load_store(&[test_shard.clone()])?;
    println!("Test store rows: {}\n", test_store.len());

    if test_store.len() == 0 {
        println!("Test shard had no rows, exiting.");
        return Ok(());
    }

    let vectors = test_store.vectors();
    let test_ids = test_store.column("_id");

    let mut total_queries = 0usize;
    let mut accumulated_latency = 0f64;
    let mut min_latency = f64::MAX;
    let mut max_latency = 0f64;
    let mut last_latency_ms: Option<f64> = None;
    let mut last_query_id: Option<String> = None;
    let mut last_match_count: Option<usize> = None;
    let overall_start = Instant::now();

    for row in 0..100 {
        let Some(vector) = vectors.vector_at(row) else {
            continue;
        };

        let id = test_ids
            .as_ref()
            .and_then(|col| string_value(col, row))
            .unwrap_or_else(|| "<unknown>".to_string());

        let query_start = Instant::now();
        let result = train_store
            .query(vector)
            .take(top_k)
            .collect()
            .map_err(|err| io::Error::new(io::ErrorKind::Other, err))?;
        let latency = query_start.elapsed().as_secs_f64();

        accumulated_latency += latency;
        if latency < min_latency {
            min_latency = latency;
        }
        if latency > max_latency {
            max_latency = latency;
        }
        last_latency_ms = Some(latency * 1_000.0);
        last_query_id = Some(id.clone());
        last_match_count = Some(result.batch.num_rows());
        total_queries += 1;

        if total_queries <= 3 {
            println!(
                "Query #{total_queries} (id={id}) returned {} matches in {:.2} ms",
                result.batch.num_rows(),
                latency * 1_000.0
            );
        }
    }

    let wall_time = overall_start.elapsed().as_secs_f64();
    println!("\n=== Summary ===");
    println!("Total queries run : {total_queries}");
    println!("Total wall time   : {:.3} s", wall_time);

    if total_queries > 0 {
        println!(
            "Average per-query : {:.3} ms",
            (accumulated_latency / total_queries as f64) * 1_000.0
        );
        println!(
            "Queries per second: {:.2}",
            total_queries as f64 / wall_time
        );
        println!("Fastest query     : {:.2} ms", min_latency * 1_000.0);
        println!("Slowest query     : {:.2} ms", max_latency * 1_000.0);
    } else {
        println!("No embeddings found in the test shard.");
    }

    if let (Some(id), Some(latency_ms), Some(matches)) =
        (last_query_id.as_ref(), last_latency_ms, last_match_count)
    {
        println!(
            "Last query #{total_queries} (id={id}) latency: {:.2} ms, matches: {matches}",
            latency_ms
        );
    }

    if let Some(stats) = train_store.get_last_query_stats() {
        println!("\n--- Last Query Stats ---");
        println!("{stats}");
    }

    println!("\n=== Demo complete ===");
    Ok(())
}

fn find_train_shards(base: &Path) -> io::Result<Vec<PathBuf>> {
    let pattern = base.join("data").join("train-*.parquet");
    let pattern = pattern.to_string_lossy().into_owned();
    let mut shards = Vec::new();

    for entry in glob(&pattern).map_err(|err| io::Error::new(io::ErrorKind::Other, err.msg))? {
        match entry {
            Ok(path) => shards.push(path),
            Err(err) => {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("Failed to read glob entry: {err}"),
                ));
            }
        }
    }

    Ok(shards)
}

fn load_store(paths: &[PathBuf]) -> Result<OttersStore, Box<dyn std::error::Error>> {
    let mut batches = Vec::new();

    for path in paths {
        println!("Loading {}", path.display());
        let file = File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|err| {
                io::Error::new(io::ErrorKind::Other, format!("{}: {err}", path.display()))
            })?
            .build()
            .map_err(|err| {
                io::Error::new(io::ErrorKind::Other, format!("{}: {err}", path.display()))
            })?;

        for batch in reader {
            let batch = batch.map_err(|err| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("{}: failed to read batch: {err}", path.display()),
                )
            })?;
            batches.push(batch);
        }
    }

    OttersStore::from_recordbatches(batches)
        .with_embedding_column(EMBEDDING_COLUMN)
        .build()
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err).into())
}

fn string_value(column: &OttersColumn, index: usize) -> Option<String> {
    if column.is_null(index) {
        None
    } else {
        column
            .string_values()
            .map(|values| values.value(index).to_owned())
    }
}
