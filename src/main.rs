use rand::random_range;

mod vec;

fn get_random_vec(dim: usize) -> Vec<f32> {
    let vec: Vec<f32> = (0..dim).map(|_| random_range(-1.0..1.0)).collect();
    vec
}

fn get_random_vectors(num_vecs: usize, dim: usize) -> Vec<Vec<f32>> {
    let vec: Vec<Vec<f32>> = (0..num_vecs).map(|_| get_random_vec(dim)).collect();
    vec
}

fn main() {
    let n_size: usize = std::env::args()
        .nth(1)
        .expect("Need Number of vectors")
        .parse()
        .unwrap();
    let dim: usize = std::env::args()
        .nth(2)
        .expect("Need dimension of vectors")
        .parse()
        .unwrap();

    let v = get_random_vectors(n_size, dim);

    let mut row_aligned_vecs = vec::RowAlignedVecs::new(dim);
    let mut column_aligned_vecs = vec::ColumnAlignedVecs::new(dim);

    row_aligned_vecs.add_vectors(v.clone()).unwrap();
    column_aligned_vecs.add_vectors(v.clone()).unwrap();

    let test_vec = get_random_vec(dim);

    let start_time = std::time::Instant::now();

    let top5_similarities = row_aligned_vecs.search_vec_cosine(&test_vec).take_max(5);

    println!(
        "Row Aligned Top 5 similarities: {:?} \n elapsed time: {:?}",
        top5_similarities,
        start_time.elapsed()
    );

    let start_time = std::time::Instant::now();

    let top5_similarities = column_aligned_vecs.search_vec_cosine(&test_vec).take_max(5);

    println!(
        "Column Aligned Top 5 similarities: {:?} \n elapsed time: {:?}",
        top5_similarities,
        start_time.elapsed()
    );
}
