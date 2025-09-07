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

fn demonstrate_column_api() -> Result<(), Box<dyn std::error::Error>> {
    Column::new("ages", DataType::Int32)
        .from(vec![Some(42), None, Some(15)])?
        .head();
    println!();

    println!("2. Float values with method chaining:");
    Column::new("prices", DataType::Float64)
        .from(vec![19.99, 29.95, 39.90])?
        .head();
    println!();

    let mut str_col = Column::new("names", DataType::String);
    str_col.push("Alice")?;
    str_col.push(None::<&str>)?;
    str_col.from(vec!["Bob", "Charlie"])?.head();
    println!();

    // Example 4: DateTime with custom format
    println!("4. DateTime with custom format:");
    Column::new("events", DataType::DateTime)
        .with_datetime_fmt("%m/%d/%Y")
        .from(vec![
            Some("01/15/2024"),
            Some("02/20/2024"),
            None,
            Some("12/25/2023"),
        ])?
        .head();
    println!();

    // Example 5: Show more rows
    println!("5. Display 10 rows:");
    Column::new("numbers", DataType::Int32)
        .from((1..=12).map(Some).collect::<Vec<_>>())?
        .head_n(10);
    println!();

    let col =
        Column::new("data", DataType::Float64).from(vec![Some(1.1), Some(2.2), Some(3.3), None])?;
    println!("Column: {}", col.name());
    let values = col.values();
    println!("   Data type: {:?}", values.data_type());
    println!("   Length: {}", values.len());
    println!("   Null values: {}", col.null_mask());

    // Pattern match to access the actual data vectors
    match values {
        otters::col::ColumnValues::Float64(data) => {
            println!("   Raw vector: {data:?}");
        }
        _ => println!("   Unexpected data type"),
    }

    col.head();
    println!();

    Ok(())
}

fn demonstrate_expr_api() -> Result<(), Box<dyn std::error::Error>> {
    use otters::expr::col;
    use otters::type_utils::DataType;
    use std::collections::HashMap;

    // Build a simple schema
    let mut schema: HashMap<String, DataType> = HashMap::new();
    schema.insert("age".to_string(), DataType::Int64);
    schema.insert("score".to_string(), DataType::Float64);
    schema.insert("name".to_string(), DataType::String);
    schema.insert("ts".to_string(), DataType::DateTime);

    println!("\n=== Expr API demo ===");

    // 1) age > 25 AND score >= 80.0
    let e1 = col("age").gt(25) & col("score").gte(80.0);
    let cf1 = e1.compile(&schema)?;
    println!("Expr1: {:?}\nPlan1: {:?}\n", e1, cf1.clauses);

    // 2) (age > 25 OR age < 18) AND name != "alice"
    let e3 = (col("age").gt(25) | col("age").lt(18)) & col("name").neq("alice");
    let cf3 = e3.compile(&schema)?;
    println!("Expr2: {:?}\nPlan2: {:?}\n", e3, cf3.clauses);

    // 3) OR of multiple string equalities
    let e4 = col("name").eq("Alice") | col("name").eq("Bob");
    let cf4 = e4.compile(&schema)?;
    println!(
        "Expr3 (name == Alice OR name == Bob): {:?}\nPlan3: {:?}\n",
        e4, cf4.clauses
    );

    // 4) DateTime comparisons using datetime strings
    let e5 = col("ts").gte("2023-01-02T03:04:05Z") & col("ts").lt("2023-12-31 23:59:59");
    let cf5 = e5.compile(&schema)?;
    println!(
        "Expr4 (DateTime strings): {:?}\nPlan4: {:?}\n",
        e5, cf5.clauses
    );

    Ok(())
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

    let v = get_random_vectors(n_size, dim);

    let mut store = VecStore::new(dim);
    store.add_vectors(v.clone())?;

    let test_vec = get_random_vec(dim);

    // Test cosine similarity search
    let start_time = std::time::Instant::now();
    let top5_similarities = store
        .query(test_vec.clone(), Metric::Cosine)
        .filter(0.1, Cmp::Gt)
        .take(5)
        .collect()?;

    println!(
        "Top 5 cosine similarities: {:?} \n elapsed time: {:?}\n",
        top5_similarities[0],
        start_time.elapsed()
    );

    // Test euclidean distance search
    let start_time = std::time::Instant::now();

    // You can also prebuild a query plan and then use it with a vector store
    let closest_5_query = VecQueryPlan::new()
        .with_query_vectors(vec![get_random_vec(dim), get_random_vec(dim)])
        .with_metric(Metric::Euclidean)
        .filter(400.0, Cmp::Lt)
        .take_global(5);

    let closest_5 = closest_5_query.with_vector_store(&store).collect()?;

    println!(
        "Top 5 closest vectors (BATCH): {:?} \n elapsed time: {:?}\n",
        closest_5,
        start_time.elapsed()
    );

    let start_time = std::time::Instant::now();

    // You can also prebuild a query plan without any query vectors
    // and then use it with an incoming query vector when you have one.
    let farthest_5_query = VecQueryPlan::new()
        .with_vector_store(&store)
        .with_metric(Metric::Cosine)
        .take_min(5);

    // Use your query vectors and execute when you are ready
    let farthest_5 = farthest_5_query
        .with_query_vectors(get_random_vec(dim))
        .collect()?;

    println!(
        "Top 5 Farthest vectors: {:?} \n elapsed time: {:?}\n",
        farthest_5,
        start_time.elapsed()
    );

    // test Column API
    demonstrate_column_api()?;

    // demo Expr API
    demonstrate_expr_api()?;

    // Demo: MetaStore with chunked zone-map pruning + vector search
    // Build metadata columns equal in length to the vectors
    use otters::expr::col;

    let age_vals: Vec<Option<i32>> = (0..n_size)
        .map(|i| {
            if i % 11 == 0 {
                None
            } else {
                Some((i % 60) as i32)
            }
        })
        .collect();
    let grades = ["A", "B+", "C", "A-", "B", "C+"];
    let grade_vals: Vec<Option<String>> = (0..n_size)
        .map(|i| {
            if i % 13 == 0 {
                None
            } else {
                Some(grades[i % grades.len()].to_string())
            }
        })
        .collect();
    let name_vals: Vec<Option<String>> = (0..n_size).map(|i| Some(format!("user_{i}"))).collect();

    let columns = vec![
        (
            "age".to_string(),
            Column::new("age", DataType::Int32).from(age_vals)?,
        ),
        (
            "grade".to_string(),
            Column::new("grade", DataType::String).from(grade_vals)?,
        ),
        (
            "name".to_string(),
            Column::new("name", DataType::String).from(name_vals)?,
        ),
    ];

    let meta = MetaStore::from_columns(columns)
        .with_vectors(v)
        .with_chunk_size(1024)
        .build()?;

    let start_time = std::time::Instant::now();
    let meta_results = meta
        .query(test_vec.clone(), Metric::Cosine)
        .meta_filter(col("age").gt(10) & col("grade").eq("A"))?
        .vec_filter(0.1, Cmp::Gt)
        .with_stats()
        .take(5)
        .collect()?;

    println!(
        "Meta query top 5: {:?} \n elapsed time: {:?}",
        meta_results.first().unwrap_or(&Vec::new()),
        start_time.elapsed()
    );

    meta.print_last_stats();

    // Explore performance across multiple chunk sizes
    let test_chunk_sizes = [256usize, 512, 1024, 2048, 4096];
    println!("\n=== Chunk Size Sweep (serial + parallel) ===");
    for &cs in &test_chunk_sizes {
        // Build fresh metadata columns and vectors for each chunk size
        let age_vals: Vec<Option<i32>> = (0..n_size)
            .map(|i| {
                if i % 11 == 0 {
                    None
                } else {
                    Some((i % 60) as i32)
                }
            })
            .collect();
        let grades = ["A", "B+", "C", "A-", "B", "C+"];
        let grade_vals: Vec<Option<String>> = (0..n_size)
            .map(|i| {
                if i % 13 == 0 {
                    None
                } else {
                    Some(grades[i % grades.len()].to_string())
                }
            })
            .collect();
        let name_vals: Vec<Option<String>> =
            (0..n_size).map(|i| Some(format!("user_{i}"))).collect();

        let columns = vec![
            (
                "age".to_string(),
                Column::new("age", DataType::Int32).from(age_vals)?,
            ),
            (
                "grade".to_string(),
                Column::new("grade", DataType::String).from(grade_vals)?,
            ),
            (
                "name".to_string(),
                Column::new("name", DataType::String).from(name_vals)?,
            ),
        ];

        // Fresh vectors so we don't clone large buffers each iteration
        let v_cs = get_random_vectors(n_size, dim);
        let meta_cs = MetaStore::from_columns(columns)
            .with_vectors(v_cs)
            .with_chunk_size(cs)
            .build()?;

        // Serial
        let start = std::time::Instant::now();
        let _ = meta_cs
            .query(test_vec.clone(), Metric::Cosine)
            .meta_filter(col("age").gt(10).and(col("grade").eq("A")))?
            .vec_filter(0.1, Cmp::Gt)
            .with_stats()
            .take(5)
            .collect()?;
        println!("[serial] chunk_size={} elapsed={:?}", cs, start.elapsed());
        meta_cs.print_last_stats();

        // Parallel
        let start = std::time::Instant::now();
        let _ = meta_cs
            .query(test_vec.clone(), Metric::Cosine)
            .meta_filter(col("age").gt(10).and(col("grade").eq("A")))?
            .vec_filter(0.1, Cmp::Gt)
            .with_stats()
            .take(5)
            .collect()?;
        println!("[parallel] chunk_size={} elapsed={:?}", cs, start.elapsed());
        meta_cs.print_last_stats();
    }

    // Parallel version for comparison
    let start_time = std::time::Instant::now();
    let _ = meta
        .query(test_vec.clone(), Metric::Cosine)
        .meta_filter(col("age").gt(10) & col("grade").eq("A"))?
        .vec_filter(0.1, Cmp::Gt)
        .with_stats()
        .take(5)
        .collect()?;
    println!("Parallel meta elapsed time: {:?}", start_time.elapsed());
    meta.print_last_stats();
    Ok(())
}
