//! Metadata-enhanced vector search with chunked zone-map pruning.
//!
//! The `MetaStore` couples a `VecStore` per chunk with auxiliary per-column
//! zonemap statistics (min/max/null counts and light bloom filters) to prune
//! irrelevant chunks before performing expensive vector similarity scoring.
//! Queries can apply both metadata expressions and vector similarity filters
//! with optional per-query or global top-k consolidation.
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{
    col::{self, Column},
    expr::{CmpOp, ColumnFilter, CompiledFilter, Expr, NumericLiteral},
    meta_compute::{ZoneStat, build_zone_stat_for_range},
    type_utils::{
        DataType, apply_rows_mask_f32, apply_rows_mask_f64, apply_rows_mask_i32,
        apply_rows_mask_i64,
    },
    vec::{Cmp as VecCmp, Metric, TakeScope, TakeType, VecStore},
};
use bitvec::bitvec;
use bitvec::prelude::BitVec;
use rayon::prelude::*;

#[derive(Debug)]
struct MetaChunk {
    base_offset: usize,
    len: usize,
    vec_store: VecStore,
    stats: HashMap<String, ZoneStat>,
}

#[derive(Debug)]
pub struct MetaStore {
    schema: HashMap<String, DataType>,
    columns: HashMap<String, Column>,
    chunk_size: usize,
    chunks: Vec<MetaChunk>,
    last_stats: std::cell::RefCell<Option<MetaQueryStats>>,
    build_stats: Option<MetaBuildStats>,
    // Tightly packed per-chunk ranges to enable SIMD-friendly zonemap filtering.
    // Generic form reduces boilerplate between numeric types.
    packed_ranges_f64: HashMap<String, PackedRanges<f64>>,
    packed_ranges_i64: HashMap<String, PackedRanges<i64>>,
    packed_ranges_f32: HashMap<String, PackedRanges<f32>>,
    packed_ranges_i32: HashMap<String, PackedRanges<i32>>,
}

#[derive(Debug)]
pub struct MetaStoreBuilder {
    schema: HashMap<String, DataType>,
    columns: HashMap<String, Column>,
    vectors: Option<Vec<Vec<f32>>>,
    chunk_size: usize,
}

#[derive(Debug, Default, Clone)]
struct PackedRanges<T> {
    min: Vec<T>,
    max: Vec<T>,
    non_null: Vec<usize>,
}

impl MetaStoreBuilder {
    pub fn with_vectors(mut self, vectors: Vec<Vec<f32>>) -> Self {
        self.vectors = Some(vectors);
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    // Backward-compatible alias
    pub fn with_chunks(self, chunk_size: usize) -> Self {
        self.with_chunk_size(chunk_size)
    }

    // Supply a single fully-built column for a name from the schema
    pub fn with_column(mut self, name: &str, column: Column) -> Result<Self, String> {
        let expected = *self
            .schema
            .get(name)
            .ok_or_else(|| format!("unknown column '{name}' not present in schema"))?;
        if expected != column.dtype() {
            return Err(format!(
                "dtype mismatch for column '{}': schema {:?}, got {:?}",
                name,
                expected,
                column.dtype()
            ));
        }
        self.columns.insert(name.to_string(), column);
        Ok(self)
    }

    // Supply multiple fully-built columns; validates against schema
    pub fn with_columns(mut self, columns: Vec<(String, Column)>) -> Result<Self, String> {
        for (name, col) in columns {
            let expected = *self
                .schema
                .get(&name)
                .ok_or_else(|| format!("unknown column '{name}' not present in schema"))?;
            if expected != col.dtype() {
                return Err(format!(
                    "dtype mismatch for column '{}': schema {:?}, got {:?}",
                    name,
                    expected,
                    col.dtype()
                ));
            }
            self.columns.insert(name, col);
        }
        Ok(self)
    }

    pub fn build(self) -> Result<MetaStore, String> {
        // Validate vectors/columns lengths
        let vectors = self
            .vectors
            .ok_or_else(|| "vectors must be provided to build MetaStore".to_string())?;

        let n_rows = vectors.len();

        // Enforce: every column in schema must be present and have the same length as vectors
        for name in self.schema.keys() {
            let col = self
                .columns
                .get(name)
                .ok_or_else(|| format!("missing column '{name}' in builder columns"))?;
            if col.len() != n_rows {
                return Err(format!(
                    "column '{}' length {} does not match vectors length {}",
                    name,
                    col.len(),
                    n_rows
                ));
            }
        }

        // Derive vector dimension
        let dim = if n_rows > 0 { vectors[0].len() } else { 0 };
        if dim == 0 && n_rows > 0 {
            return Err("vector dimension cannot be zero".to_string());
        }
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(format!(
                    "vector at index {} has dim {}, expected {}",
                    i,
                    v.len(),
                    dim
                ));
            }
        }

        // Timings
        let build_start = Instant::now();
        let mut vectors_ingest_duration = Duration::default();
        let mut zonemap_build_duration = Duration::default();

        // Prepare packed per-chunk ranges for SIMD-friendly filtering
        let mut packed_ranges_f64: HashMap<String, PackedRanges<f64>> = HashMap::new();
        let mut packed_ranges_i64: HashMap<String, PackedRanges<i64>> = HashMap::new();
        let mut packed_ranges_f32: HashMap<String, PackedRanges<f32>> = HashMap::new();
        let mut packed_ranges_i32: HashMap<String, PackedRanges<i32>> = HashMap::new();

        // Build chunks without cloning vectors: consume the owned vectors in order
        let mut chunks: Vec<MetaChunk> = Vec::new();
        let mut base_offset = 0usize;
        let mut iter = vectors.into_iter();
        loop {
            let mut chunk_vecs: Vec<Vec<f32>> =
                Vec::with_capacity(self.chunk_size.min(n_rows - base_offset));
            for _ in 0..self.chunk_size {
                if let Some(v) = iter.next() {
                    chunk_vecs.push(v); // move, no clone
                } else {
                    break;
                }
            }
            if chunk_vecs.is_empty() {
                break;
            }

            let mut vs = VecStore::new(dim);
            let ingest_start = Instant::now();
            vs.add_vectors_owned(chunk_vecs)?; // move vectors into the store without cloning
            vectors_ingest_duration += ingest_start.elapsed();

            let end = base_offset + vs.len(); // length of this chunk

            // Build stats for this chunk
            let mut stats: HashMap<String, ZoneStat> = HashMap::new();
            let zstart = Instant::now();
            for (name, dtype) in &self.schema {
                let col = self
                    .columns
                    .get(name)
                    .ok_or_else(|| format!("missing column '{name}' in builder columns"))?;
                let stat = build_zone_stat_for_range(col, *dtype, base_offset, end)?;
                stats.insert(name.clone(), stat);
            }
            zonemap_build_duration += zstart.elapsed();

            // Append packed per-chunk ranges
            for (name, dtype) in &self.schema {
                match (dtype, stats.get(name).unwrap()) {
                    (DataType::Float32, ZoneStat::Float { min, max, non_null }) => {
                        // Use only f32 packed ranges for f32 columns
                        let e32 = packed_ranges_f32.entry(name.clone()).or_default();
                        e32.min.push(*min as f32);
                        e32.max.push(*max as f32);
                        e32.non_null.push(*non_null);
                    }
                    (DataType::Float64, ZoneStat::Float { min, max, non_null }) => {
                        let entry = packed_ranges_f64.entry(name.clone()).or_default();
                        entry.min.push(*min);
                        entry.max.push(*max);
                        entry.non_null.push(*non_null);
                    }
                    (DataType::Int32, ZoneStat::Int { min, max, non_null }) => {
                        // Use only i32 packed ranges for i32 columns
                        let e32 = packed_ranges_i32.entry(name.clone()).or_default();
                        e32.min.push(*min as i32);
                        e32.max.push(*max as i32);
                        e32.non_null.push(*non_null);
                    }
                    (DataType::Int64, ZoneStat::Int { min, max, non_null }) => {
                        let entry = packed_ranges_i64.entry(name.clone()).or_default();
                        entry.min.push(*min);
                        entry.max.push(*max);
                        entry.non_null.push(*non_null);
                    }
                    (DataType::DateTime, ZoneStat::DateTime { min, max, non_null }) => {
                        let entry = packed_ranges_i64.entry(name.clone()).or_default();
                        entry.min.push(*min);
                        entry.max.push(*max);
                        entry.non_null.push(*non_null);
                    }
                    _ => {}
                }
            }

            chunks.push(MetaChunk {
                base_offset,
                len: end - base_offset,
                vec_store: vs,
                stats,
            });

            base_offset = end;
            if base_offset >= n_rows {
                break;
            }
        }

        let build_total_duration = build_start.elapsed();
        let n_chunks = chunks.len();

        Ok(MetaStore {
            schema: self.schema,
            columns: self.columns,
            chunk_size: self.chunk_size,
            chunks,
            last_stats: std::cell::RefCell::new(None),
            build_stats: Some(MetaBuildStats {
                n_rows,
                dim,
                n_chunks,
                vectors_ingest_duration,
                zonemap_build_duration,
                build_total_duration,
            }),
            packed_ranges_f64,
            packed_ranges_i64,
            packed_ranges_f32,
            packed_ranges_i32,
        })
    }
}

// build_zone_stat_for_range moved to meta_compute.rs

impl MetaStore {
    /// Lightweight constructor for empty columns; primarily for demos.
    pub fn new(schema: &[(String, DataType)]) -> Self {
        let schema_map = HashMap::from_iter(schema.iter().cloned());
        let columns = HashMap::from_iter(
            schema
                .iter()
                .map(|(name, dtype)| (name.clone(), col::Column::new(name, *dtype))),
        );
        Self {
            schema: schema_map,
            columns,
            chunk_size: 1024,
            chunks: Vec::new(),
            last_stats: std::cell::RefCell::new(None),
            build_stats: None,
            packed_ranges_f64: HashMap::new(),
            packed_ranges_i64: HashMap::new(),
            packed_ranges_f32: HashMap::new(),
            packed_ranges_i32: HashMap::new(),
        }
    }

    /// Start a builder using a provided schema and fully-populated columns.
    /// Prefer `from_columns`.
    pub fn builder_from_columns(columns: Vec<(String, Column)>) -> MetaStoreBuilder {
        Self::from_columns(columns)
    }

    /// Start a builder using a provided schema and fully-populated columns.
    pub fn from_columns(columns: Vec<(String, Column)>) -> MetaStoreBuilder {
        let mut schema = HashMap::new();
        let mut col_map = HashMap::new();
        for (name, col) in columns {
            schema.insert(name.clone(), col.dtype());
            col_map.insert(name, col);
        }
        MetaStoreBuilder {
            schema,
            columns: col_map,
            vectors: None,
            chunk_size: 1024,
        }
    }

    /// Start a builder by schema; you should supply filled columns via `with_columns`.
    pub fn from_schema(schema: &[(String, DataType)]) -> MetaStoreBuilder {
        let mut schema_map = HashMap::new();
        let mut columns = HashMap::new();
        for (name, dt) in schema {
            schema_map.insert(name.clone(), *dt);
            columns.insert(name.clone(), Column::new(name, *dt));
        }
        MetaStoreBuilder {
            schema: schema_map,
            columns,
            vectors: None,
            chunk_size: 1024,
        }
    }

    pub fn head(&self) {
        println!("MetaStore:");
        println!("Schema:");
        for (name, dtype) in &self.schema {
            println!(" - {name}: {dtype:?}");
        }
        for col in self.columns.values() {
            col.head();
        }
        println!(
            "Chunks: {} (chunk_size: {})",
            self.chunks.len(),
            self.chunk_size
        );
    }

    pub fn last_query_stats(&self) -> Option<MetaQueryStats> {
        self.last_stats.borrow().clone()
    }

    // Build a per-chunk mask for the compiled plan using packed ranges and SIMD, mirroring
    // the pruning logic used during query planning. True indicates the chunk may satisfy the plan.
    fn build_chunk_mask_for_plan(&self, compiled: &CompiledFilter) -> BitVec {
        let n_chunks = self.chunks.len();
        if n_chunks == 0 {
            return BitVec::new();
        }
        compiled
            .clauses
            .iter()
            .fold(bitvec![1; n_chunks], |mut acc, clause| {
                let mut clause_mask = bitvec![0; n_chunks];
                clause.iter().for_each(|leaf| match leaf {
                    ColumnFilter::Numeric { column, cmp, rhs } => {
                        self.apply_numeric_leaf_chunk_mask(&mut clause_mask, column, *cmp, rhs);
                    }
                    ColumnFilter::String { column, cmp, rhs } => {
                        self.apply_string_leaf_chunk_mask(&mut clause_mask, column, *cmp, rhs);
                    }
                });
                acc &= &clause_mask;
                acc
            })
    }

    // Leaf helpers for chunk preselection
    fn apply_numeric_leaf_chunk_mask(
        &self,
        mask: &mut BitVec,
        column: &str,
        cmp: CmpOp,
        rhs: &NumericLiteral,
    ) {
        let n_chunks = mask.len();
        match rhs {
            NumericLiteral::F64(rv) => match self.schema.get(column) {
                Some(DataType::Float32) => {
                    if let Some(packed) = self.packed_ranges_f32.get(column) {
                        crate::type_utils::apply_chunk_mask_ranges_f32_bits(
                            &packed.min,
                            &packed.max,
                            &packed.non_null,
                            n_chunks,
                            cmp,
                            *rv as f32,
                            mask,
                        );
                    }
                }
                Some(DataType::Float64) => {
                    if let Some(packed) = self.packed_ranges_f64.get(column) {
                        crate::type_utils::apply_chunk_mask_ranges_f64_bits(
                            &packed.min,
                            &packed.max,
                            &packed.non_null,
                            n_chunks,
                            cmp,
                            *rv,
                            mask,
                        );
                    }
                }
                _ => {
                    if let Some(packed) = self.packed_ranges_i64.get(column) {
                        crate::type_utils::apply_chunk_mask_ranges_i64_bits(
                            &packed.min,
                            &packed.max,
                            &packed.non_null,
                            n_chunks,
                            cmp,
                            *rv as i64,
                            mask,
                        );
                    }
                }
            },
            NumericLiteral::I64(rv) => match self.schema.get(column) {
                Some(DataType::Int32) => {
                    if let Some(packed) = self.packed_ranges_i32.get(column) {
                        crate::type_utils::apply_chunk_mask_ranges_i32_bits(
                            &packed.min,
                            &packed.max,
                            &packed.non_null,
                            n_chunks,
                            cmp,
                            *rv as i32,
                            mask,
                        );
                    }
                }
                Some(DataType::Int64) | Some(DataType::DateTime) => {
                    if let Some(packed) = self.packed_ranges_i64.get(column) {
                        crate::type_utils::apply_chunk_mask_ranges_i64_bits(
                            &packed.min,
                            &packed.max,
                            &packed.non_null,
                            n_chunks,
                            cmp,
                            *rv,
                            mask,
                        );
                    } else if let Some(packed) = self.packed_ranges_f64.get(column) {
                        crate::type_utils::apply_chunk_mask_ranges_f64_bits(
                            &packed.min,
                            &packed.max,
                            &packed.non_null,
                            n_chunks,
                            cmp,
                            *rv as f64,
                            mask,
                        );
                    }
                }
                _ => {}
            },
        }
    }

    fn apply_string_leaf_chunk_mask(&self, mask: &mut BitVec, column: &str, cmp: CmpOp, rhs: &str) {
        for (i, chunk) in self.chunks.iter().enumerate() {
            if let Some(ZoneStat::String { bloom, non_null }) = chunk.stats.get(column) {
                if *non_null == 0 {
                    continue;
                }
                match cmp {
                    CmpOp::Eq => {
                        if bloom.contains(rhs.as_bytes()) {
                            mask.set(i, true);
                        }
                    }
                    CmpOp::Neq => {
                        mask.set(i, true);
                    }
                    _ => {}
                }
            } else {
                mask.set(i, true); // conservatively keep when unknown
            }
        }
    }

    pub fn print_last_stats(&self) {
        println!("-- MetaStore Build Stats --");
        match &self.build_stats {
            Some(b) => {
                println!(
                    "Rows: {}, Dimensions: {}, Chunks: {}",
                    b.n_rows, b.dim, b.n_chunks
                );
                println!(
                    "Vector Ingest: {:.3} ms | Zone Map Build: {:.3} ms | Build Total: {:.3} ms",
                    b.vectors_ingest_duration.as_secs_f64() * 1000.0,
                    b.zonemap_build_duration.as_secs_f64() * 1000.0,
                    b.build_total_duration.as_secs_f64() * 1000.0
                );
            }
            None => println!("(no build stats)"),
        }

        println!("-- Last Meta Query Stats --");
        match self.last_query_stats() {
            Some(s) => {
                println!(
                    "Chunks: total={} | pruned={} | evaluated={}",
                    s.total_chunks, s.pruned_chunks, s.evaluated_chunks
                );
                println!(
                    "Vector Comparisons: {} | Candidates before meta filter: {} | After meta filter: {}",
                    s.vectors_compared, s.results_before_postfilter, s.results_after_postfilter
                );
                println!(
                    "Timing: prune={:.3} ms | score={:.3} ms | merge={:.3} ms | total={:.3} ms",
                    s.prune_duration.as_secs_f64() * 1000.0,
                    s.score_duration.as_secs_f64() * 1000.0,
                    s.merge_duration.as_secs_f64() * 1000.0,
                    s.total_duration.as_secs_f64() * 1000.0
                );
            }
            None => println!("(no query stats)"),
        }
    }

    /// Begin a query plan over this MetaStore with a single query vector
    pub fn query(&self, query: Vec<f32>, metric: Metric) -> MetaQueryPlan {
        MetaQueryPlan::new(self, vec![query], metric)
    }

    /// Begin a query plan with multiple query vectors
    pub fn query_batch(&self, queries: Vec<Vec<f32>>, metric: Metric) -> MetaQueryPlan {
        MetaQueryPlan::new(self, queries, metric)
    }
}

// compute helpers moved to meta_compute.rs

#[derive(Debug)]
pub struct MetaQueryPlan<'a> {
    store: &'a MetaStore,
    queries: Vec<Vec<f32>>,
    metric: Metric,
    meta_filter: Option<CompiledFilter>,
    vec_filter: Option<(f32, VecCmp)>,
    take_type: Option<TakeType>,
    take_count: Option<usize>,
    take_scope: TakeScope,
    capture_stats: bool,
}

// Internal aggregation bucket for per-chunk processing
#[derive(Default)]
struct ChunkAgg {
    results: Vec<Vec<(usize, f32)>>,
    before: usize,
    after: usize,
    compared: usize,
}

// Process a single chunk: run per-chunk VecStore query, apply optional row-level meta filter,
// and collect results and counters. This function is thread-safe and does not capture &self.
#[allow(clippy::too_many_arguments)]
fn process_chunk(
    chunk: &MetaChunk,
    take_scope: &TakeScope,
    metric: &Metric,
    queries: &[Vec<f32>],
    vec_filter: Option<(f32, VecCmp)>,
    meta_filter: Option<&CompiledFilter>,
    columns: &HashMap<String, Column>,
    k: usize,
    n_queries: usize,
) -> ChunkAgg {
    let mut agg = ChunkAgg {
        results: match *take_scope {
            TakeScope::Local => vec![Vec::new(); n_queries],
            TakeScope::Global => vec![Vec::new()],
        },
        before: 0,
        after: 0,
        compared: chunk.len * n_queries,
    };

    // Build SIMD row mask if meta filter present
    let row_mask_opt: Option<BitVec> =
        meta_filter.map(|cf| build_row_mask_for_chunk(cf, columns, chunk.base_offset, chunk.len));

    let metric2 = *metric;
    let mut plan = chunk.vec_store.query(queries.to_owned(), metric2);
    if let Some((thr, cmp)) = &vec_filter {
        plan = plan.filter(*thr, cmp.clone());
    }
    if let Some(rm) = row_mask_opt.clone() {
        plan = plan.with_row_mask(rm);
    }
    plan = match *take_scope {
        TakeScope::Local => plan.take(k),
        TakeScope::Global => plan.take(k),
    };

    if let Ok(mut results) = plan.collect() {
        match *take_scope {
            TakeScope::Local => {
                for (q_idx, vecs) in results.iter_mut().enumerate() {
                    agg.before += vecs.len();
                    for (idx, score) in vecs.drain(..) {
                        let global_idx = chunk.base_offset + idx;
                        agg.results[q_idx].push((global_idx, score));
                        agg.after += 1;
                    }
                }
            }
            TakeScope::Global => {
                let vecs = results.remove(0);
                agg.before += vecs.len();
                for (idx, score) in vecs.into_iter() {
                    let global_idx = chunk.base_offset + idx;
                    agg.results[0].push((global_idx, score));
                    agg.after += 1;
                }
            }
        }
    }

    agg
}

fn build_row_mask_for_chunk(
    compiled: &CompiledFilter,
    columns: &HashMap<String, Column>,
    base: usize,
    len: usize,
) -> BitVec {
    let mut candidates = bitvec![1; len];
    for clause in &compiled.clauses {
        let mut clause_mask = bitvec![0; len];
        for leaf in clause {
            match leaf {
                ColumnFilter::Numeric { column, cmp, rhs } => {
                    apply_numeric_leaf_row_mask(
                        columns,
                        base,
                        len,
                        column,
                        *cmp,
                        rhs,
                        &mut clause_mask,
                    );
                }
                ColumnFilter::String { column, cmp, rhs } => {
                    apply_string_leaf_row_mask(
                        columns,
                        base,
                        len,
                        column,
                        *cmp,
                        rhs,
                        &mut clause_mask,
                    );
                }
            }
        }
        candidates &= clause_mask;
    }
    candidates
}

// Leaf helpers for row mask
fn apply_numeric_leaf_row_mask(
    columns: &HashMap<String, Column>,
    base: usize,
    len: usize,
    column: &str,
    cmp: CmpOp,
    rhs: &NumericLiteral,
    clause_mask: &mut BitVec,
) {
    if let Some(col) = columns.get(column) {
        match col.dtype() {
            DataType::Float32 => {
                let vals = col.f32_values().unwrap();
                let nulls = col.null_mask();
                let thr = match rhs {
                    NumericLiteral::F64(v) => *v as f32,
                    NumericLiteral::I64(v) => *v as f32,
                };
                apply_rows_mask_f32(vals, nulls, base, len, cmp, thr, clause_mask);
            }
            DataType::Int32 => {
                let vals = col.i32_values().unwrap();
                let nulls = col.null_mask();
                let thr = match rhs {
                    NumericLiteral::I64(v) => *v as i32,
                    NumericLiteral::F64(v) => *v as i32,
                };
                apply_rows_mask_i32(vals, nulls, base, len, cmp, thr, clause_mask);
            }
            DataType::Float64 => {
                let vals = col.f64_values().unwrap();
                let nulls = col.null_mask();
                let thr = match rhs {
                    NumericLiteral::F64(v) => *v,
                    NumericLiteral::I64(v) => *v as f64,
                };
                apply_rows_mask_f64(vals, nulls, base, len, cmp, thr, clause_mask);
            }
            DataType::Int64 | DataType::DateTime => {
                let vals = if col.dtype() == DataType::Int64 {
                    col.i64_values().unwrap()
                } else {
                    col.datetime_values().unwrap()
                };
                let nulls = col.null_mask();
                let thr = match rhs {
                    NumericLiteral::I64(v) => *v,
                    NumericLiteral::F64(v) => *v as i64,
                };
                apply_rows_mask_i64(vals, nulls, base, len, cmp, thr, clause_mask);
            }
            DataType::String => {}
        }
    }
}

fn apply_string_leaf_row_mask(
    columns: &HashMap<String, Column>,
    base: usize,
    len: usize,
    column: &str,
    cmp: CmpOp,
    rhs: &str,
    clause_mask: &mut BitVec,
) {
    if let Some(col) = columns.get(column) {
        let vals = col.string_values().unwrap();
        let nulls = col.null_mask();
        for off in 0..len {
            if nulls.get(base + off).map(|b| *b).unwrap_or(false) {
                continue;
            }
            let v = &vals[base + off];
            let sat = match cmp {
                CmpOp::Eq => v == rhs,
                CmpOp::Neq => v != rhs,
                _ => false,
            };
            if sat {
                clause_mask.set(off, true);
            }
        }
    }
}

impl<'a> MetaQueryPlan<'a> {
    fn new(store: &'a MetaStore, queries: Vec<Vec<f32>>, metric: Metric) -> Self {
        Self {
            store,
            queries,
            metric,
            meta_filter: None,
            vec_filter: None,
            take_type: None,
            take_count: None,
            take_scope: TakeScope::Local,
            capture_stats: false,
        }
    }

    pub fn meta_filter(mut self, expr: Expr) -> Result<Self, String> {
        let compiled = expr
            .compile(&self.store.schema)
            .map_err(|e| format!("meta_filter compile error: {e}"))?;
        self.meta_filter = Some(compiled);
        Ok(self)
    }

    pub fn vec_filter(mut self, score: f32, cmp: VecCmp) -> Self {
        self.vec_filter = Some((score, cmp));
        self
    }

    pub fn take(mut self, k: usize) -> Self {
        self.take_count = Some(k);
        self.take_type = Some(match self.metric {
            Metric::Euclidean => TakeType::Min,
            Metric::Cosine | Metric::DotProduct => TakeType::Max,
        });
        self
    }

    pub fn take_global(mut self, k: usize) -> Self {
        self.take_count = Some(k);
        self.take_scope = TakeScope::Global;
        self.take_type = Some(match self.metric {
            Metric::Euclidean => TakeType::Min,
            Metric::Cosine | Metric::DotProduct => TakeType::Max,
        });
        self
    }

    pub fn with_stats(mut self) -> Self {
        self.capture_stats = true;
        self
    }

    // parallel is always on; no with_parallel needed

    pub fn collect(self) -> Result<Vec<Vec<(usize, f32)>>, String> {
        let total_start = if self.capture_stats {
            Some(Instant::now())
        } else {
            None
        };
        let k = self.take_count.unwrap_or_else(|| {
            // default: number of rows across store
            self.store.chunks.iter().map(|c| c.len).sum()
        });
        let take_type = self.take_type.unwrap_or(match self.metric {
            Metric::Euclidean => TakeType::Min,
            _ => TakeType::Max,
        });

        // Preselect chunks via zone stats
        let prune_start = if self.capture_stats {
            Some(Instant::now())
        } else {
            None
        };
        let candidate_chunks: Vec<&MetaChunk> = match self.meta_filter.as_ref() {
            Some(compiled) => self
                .store
                .build_chunk_mask_for_plan(compiled)
                .iter()
                .by_vals()
                .enumerate()
                .filter(|&(_, keep)| keep)
                .map(|(i, _)| &self.store.chunks[i])
                .collect(),
            None => self.store.chunks.iter().collect(),
        };
        let prune_duration = prune_start.map(|t| t.elapsed()).unwrap_or_default();

        // Aggregated results per-query (or single vector for global)
        let n_queries = self.queries.len();
        let mut aggregated: Vec<Vec<(usize, f32)>> = match self.take_scope {
            TakeScope::Local => vec![Vec::new(); n_queries],
            TakeScope::Global => vec![Vec::new()],
        };

        // Stats counters
        let total_chunks = self.store.chunks.len();
        let evaluated_chunks = candidate_chunks.len();
        let pruned_chunks = total_chunks - evaluated_chunks;
        let mut vectors_compared: usize = 0;
        let mut results_before_postfilter: usize = 0;
        let mut results_after_postfilter: usize = 0;

        let score_start = if self.capture_stats {
            Some(Instant::now())
        } else {
            None
        };
        // Extract inputs once
        let take_scope = self.take_scope;
        let metric_ref = &self.metric;
        let queries_ref = &self.queries;
        let vec_filter_opt = self.vec_filter.clone();
        let meta_filter_ref = self.meta_filter.as_ref();
        let columns_ref = &self.store.columns;

        let per_chunk: Vec<ChunkAgg> = candidate_chunks
            .par_iter()
            .map(|c| {
                process_chunk(
                    c,
                    &take_scope,
                    metric_ref,
                    queries_ref,
                    vec_filter_opt.clone(),
                    meta_filter_ref,
                    columns_ref,
                    k,
                    n_queries,
                )
            })
            .collect();

        for agg in per_chunk {
            if self.capture_stats {
                vectors_compared += agg.compared;
                results_before_postfilter += agg.before;
                results_after_postfilter += agg.after;
            }
            match take_scope {
                TakeScope::Local => {
                    for (q_idx, mut v) in agg.results.into_iter().enumerate() {
                        aggregated[q_idx].append(&mut v);
                    }
                }
                TakeScope::Global => {
                    let mut v = agg.results.into_iter().next().unwrap();
                    aggregated[0].append(&mut v);
                }
            }
        }
        let score_duration = score_start.map(|t| t.elapsed()).unwrap_or_default();

        let merge_start = if self.capture_stats {
            Some(Instant::now())
        } else {
            None
        };
        // Final top-k per bucket
        for bucket in aggregated.iter_mut() {
            // sort and truncate by take_type
            match take_type {
                TakeType::Min => bucket.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap()),
                TakeType::Max => bucket.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap()),
            }
            if bucket.len() > k {
                bucket.truncate(k);
            }
        }
        if self.capture_stats {
            let merge_duration = merge_start.map(|t| t.elapsed()).unwrap_or_default();
            let total_duration = total_start.map(|t| t.elapsed()).unwrap_or_default();
            let stats = MetaQueryStats {
                total_chunks,
                pruned_chunks,
                evaluated_chunks,
                vectors_compared,
                results_before_postfilter,
                results_after_postfilter,
                prune_duration,
                score_duration,
                merge_duration,
                total_duration,
            };
            *self.store.last_stats.borrow_mut() = Some(stats);
        }
        Ok(aggregated)
    }
}

#[derive(Debug, Clone)]
pub struct MetaQueryStats {
    pub total_chunks: usize,
    pub pruned_chunks: usize,
    pub evaluated_chunks: usize,
    pub vectors_compared: usize,
    pub results_before_postfilter: usize,
    pub results_after_postfilter: usize,
    pub prune_duration: Duration,
    pub score_duration: Duration,
    pub merge_duration: Duration,
    pub total_duration: Duration,
}

#[derive(Debug, Clone)]
pub struct MetaBuildStats {
    pub n_rows: usize,
    pub dim: usize,
    pub n_chunks: usize,
    pub vectors_ingest_duration: Duration,
    pub zonemap_build_duration: Duration,
    pub build_total_duration: Duration,
}
