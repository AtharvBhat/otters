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

    Ok(())
}
