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
    let n_size:usize = std::env::args().nth(1).expect("Need Number of vectors").parse().unwrap();
    let dim:usize = std::env::args().nth(2).expect("Need dimension of vectors").parse().unwrap();

    let mut row_aligned_vecs = vec::RowAlignedVecs::new(dim);
    let v = get_random_vectors(n_size, dim);
    row_aligned_vecs.add_vectors(v).unwrap();

    let test_vec = get_random_vec(dim);

    // top 5 by similarity
    let mut cosine_sims = row_aligned_vecs
        .compare_all(&test_vec, vec::ComparisonMethods::Euclidean);

    cosine_sims.sort_by(|a, b| a.total_cmp(b));
    let top5:Vec<f32> = cosine_sims.into_iter().take(5).collect();
    println!("{top5:?}")
}
