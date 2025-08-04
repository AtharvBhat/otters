use otters::col::{Column, DataType};
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
    let _col = Column::new("ages", DataType::Int32)
        .from(vec![Some(42), None, Some(15)])?
        .head();
    println!();

    println!("2. Float values with method chaining:");
    let _col = Column::new("prices", DataType::Float64)
        .from(vec![19.99, 29.95, 39.90])?
        .head();
    println!();

    let mut str_col = Column::new("names", DataType::String);
    str_col.push("Alice")?;
    str_col.push(None::<&str>)?;
    let _col = str_col.from(vec!["Bob", "Charlie"])?.head();
    println!();

    // Example 4: DateTime with custom format
    println!("4. DateTime with custom format:");
    let _col = Column::new("events", DataType::DateTime)
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
    let _col = Column::new("numbers", DataType::Int32)
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
            println!("   Raw vector: {:?}", data);
        }
        _ => println!("   Unexpected data type"),
    }

    let _col = col.head();
    println!();

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
        "Top 5 cosine similarities: {:?} \n elapsed time: {:?}",
        top5_similarities[0],
        start_time.elapsed()
    );

    // Test euclidean distance search
    let start_time = std::time::Instant::now();
    let closest_5 = store
        .query(test_vec, Metric::Euclidean)
        .filter(300.0, Cmp::Lt)
        .take(5)
        .collect()?;

    println!(
        "Top 5 closest vectors: {:?} \n elapsed time: {:?}",
        closest_5[0],
        start_time.elapsed()
    );

    // test Column API
    demonstrate_column_api()?;

    Ok(())
}
