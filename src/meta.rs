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
    meta_compute::{
        BloomBuild, ChunkAgg, MetaChunk, ZoneStat, build_zone_stat_for_range, process_chunk,
    },
    type_utils::DataType,
    vec::{Cmp as VecCmp, Metric, TakeType, VecStore},
};
use bitvec::bitvec;
use bitvec::prelude::BitVec;
use itertools::Itertools;
use rayon::prelude::*;

pub struct MetaQueryResults {
    pub columns: Vec<String>,
    pub data: HashMap<String, Column>,
    pub indices: Vec<usize>,
    pub scores: Vec<f32>,
}

impl MetaQueryResults {
    pub fn len(&self) -> usize {
        self.indices.len()
    }
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.data.get(name)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BloomConfig {
    Fpr(f64),
    Bits(usize),
}

#[derive(Debug)]
pub struct MetaStore {
    schema: HashMap<String, DataType>,
    columns: HashMap<String, Column>,
    chunk_size: usize,
    chunks: Vec<MetaChunk>,
    last_stats: std::cell::RefCell<Option<MetaQueryStats>>,
    build_stats: Option<MetaBuildStats>,
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
    bloom: BloomConfig,
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

    /// Configure bloom filter by target false-positive rate (0 < fpr < 1).
    pub fn with_bloom_fpr(mut self, fpr: f64) -> Self {
        // Clamp to sane bounds, fallback to default when non-finite
        let f = if fpr.is_finite() {
            fpr.clamp(1e-2, 0.5)
        } else {
            0.01
        };
        self.bloom = BloomConfig::Fpr(f);
        self
    }

    /// Configure bloom filter by total number of bits.
    /// Use this to size the filter explicitly (e.g. 1024, 4096, ...).
    pub fn with_bloom_bits(mut self, bits: usize) -> Self {
        let b = bits.max(64); // minimal sane floor
        self.bloom = BloomConfig::Bits(b);
        self
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

    // Supply multiple fully-built columns and validates against schema
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

        // every column in schema must be present and have the same length as vectors
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

        let mut packed_ranges_f64: HashMap<String, PackedRanges<f64>> = HashMap::new();
        let mut packed_ranges_i64: HashMap<String, PackedRanges<i64>> = HashMap::new();
        let mut packed_ranges_f32: HashMap<String, PackedRanges<f32>> = HashMap::new();
        let mut packed_ranges_i32: HashMap<String, PackedRanges<i32>> = HashMap::new();

        let mut chunks: Vec<MetaChunk> = Vec::new();
        let mut base_offset = 0usize;
        for chunk_iter in &vectors.into_iter().chunks(self.chunk_size) {
            let chunk_vecs: Vec<Vec<f32>> = chunk_iter.collect();
            if chunk_vecs.is_empty() {
                continue;
            }

            let mut vs = VecStore::new(dim);
            let ingest_start = Instant::now();
            vs.add_vectors(chunk_vecs)?;
            vectors_ingest_duration += ingest_start.elapsed();

            let end = base_offset + vs.len();

            // Build stats for this chunk
            let zstart = Instant::now();
            let bloom_cfg = match self.bloom {
                BloomConfig::Fpr(p) => BloomBuild::Fpr(p),
                BloomConfig::Bits(b) => BloomBuild::Bits(b),
            };
            let stats: HashMap<String, ZoneStat> = self
                .schema
                .iter()
                .map(|(name, dtype)| {
                    let col = self
                        .columns
                        .get(name)
                        .ok_or_else(|| format!("missing column '{name}' in builder columns"))?;
                    let stat = build_zone_stat_for_range(col, *dtype, base_offset, end, bloom_cfg)?;
                    Ok::<(String, ZoneStat), String>((name.clone(), stat))
                })
                .collect::<Result<_, _>>()?;
            zonemap_build_duration += zstart.elapsed();

            // Append packed per-chunk ranges
            self.schema
                .iter()
                .for_each(|(name, dtype)| match (dtype, stats.get(name).unwrap()) {
                    (DataType::Float32, ZoneStat::Float { min, max, non_null }) => {
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
                });

            chunks.push(MetaChunk {
                base_offset,
                len: end - base_offset,
                vec_store: vs,
                stats,
            });

            base_offset = end;
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

impl MetaStore {
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

    pub fn from_columns(columns: Vec<Column>) -> MetaStoreBuilder {
        let mut schema = HashMap::new();
        let mut col_map = HashMap::new();
        for col in columns {
            let name = col.name().to_string();
            schema.insert(name.clone(), col.dtype());
            col_map.insert(name, col);
        }
        MetaStoreBuilder {
            schema,
            columns: col_map,
            vectors: None,
            chunk_size: 1024,
            bloom: BloomConfig::Fpr(0.01),
        }
    }

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
            bloom: BloomConfig::Fpr(0.01),
        }
    }

    pub fn head(&self) {
        self.head_n(5)
    }

    pub fn head_n(&self, n: usize) {
        println!("{}", crate::display::metastore_head(self, n));
    }

    // Accessors for display helpers
    pub fn schema(&self) -> &HashMap<String, DataType> {
        &self.schema
    }
    pub fn columns(&self) -> &HashMap<String, Column> {
        &self.columns
    }
    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
    }
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn last_query_stats(&self) -> Option<MetaQueryStats> {
        self.last_stats.borrow().clone()
    }

    /// Return build-time stats captured when the MetaStore was constructed.
    pub fn build_stats(&self) -> Option<MetaBuildStats> {
        self.build_stats.clone()
    }

    // Build a per-chunk mask for the compiled plan
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

    /// Print only build stats as an ASCII table.
    pub fn print_build_stats(&self) {
        match &self.build_stats {
            Some(b) => println!("{}", crate::display::format_build_stats(b)),
            None => println!("(no build stats)"),
        }
    }

    /// Print only the last query stats as an ASCII table.
    pub fn print_last_query_stats(&self) {
        match self.last_query_stats() {
            Some(s) => println!("{}", crate::display::format_query_stats(&s)),
            None => println!("(no query stats)"),
        }
    }

    /// Backwards-compatible combined stats printer (build + last query)
    pub fn print_last_stats(&self) {
        self.print_build_stats();
        self.print_last_query_stats();
    }

    /// Begin a query plan over this MetaStore with a single query vector
    pub fn query(&self, query: Vec<f32>, metric: Metric) -> MetaQueryPlan<'_> {
        MetaQueryPlan::new(self, vec![query], metric)
    }

    /// Begin a query plan with multiple query vectors
    pub fn query_batch(&self, queries: Vec<Vec<f32>>, metric: Metric) -> MetaQueryPlan<'_> {
        MetaQueryPlan::new(self, queries, metric)
    }
}

#[derive(Debug)]
pub struct MetaQueryPlan<'a> {
    store: &'a MetaStore,
    queries: Vec<Vec<f32>>,
    metric: Metric,
    meta_filter: Option<CompiledFilter>,
    vec_filter: Option<(f32, VecCmp)>,
    take_type: Option<TakeType>,
    take_count: Option<usize>,
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

    pub fn collect(self) -> Result<MetaQueryResults, String> {
        let total_start = Instant::now();
        let k = self
            .take_count
            .unwrap_or_else(|| self.store.chunks.iter().map(|c| c.len).sum());
        let take_type = self.take_type.unwrap_or(match self.metric {
            Metric::Euclidean => TakeType::Min,
            _ => TakeType::Max,
        });

        // Preselect chunks via zone stats
        let prune_start = Instant::now();
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
        let prune_duration = prune_start.elapsed();

        // Aggregated results into a single list across all queries
        let mut aggregated: Vec<(usize, f32)> = Vec::new();

        // Stats counters
        let total_chunks = self.store.chunks.len();
        let evaluated_chunks = candidate_chunks.len();
        let pruned_chunks = total_chunks - evaluated_chunks;
        let mut vectors_compared: usize = 0;

        let score_start = Instant::now();
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
                    metric_ref,
                    queries_ref,
                    vec_filter_opt.clone(),
                    meta_filter_ref,
                    columns_ref,
                    k,
                )
            })
            .collect();

        per_chunk.into_iter().for_each(|agg| {
            vectors_compared += agg.compared;
            aggregated.extend(agg.results);
        });
        let score_duration = score_start.elapsed();

        let merge_start = Instant::now();
        // Final top-k per bucket
        // sort and truncate final list
        match take_type {
            TakeType::Min => aggregated.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap()),
            TakeType::Max => aggregated.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap()),
        }
        if aggregated.len() > k {
            aggregated.truncate(k);
        }
        let merge_duration = merge_start.elapsed();
        let total_duration = total_start.elapsed();
        let stats = MetaQueryStats {
            total_chunks,
            pruned_chunks,
            evaluated_chunks,
            vectors_compared,
            prune_duration,
            score_duration,
            merge_duration,
            total_duration,
        };
        *self.store.last_stats.borrow_mut() = Some(stats);
        // Build result columns and collect indices/scores in result order
        let mut col_names: Vec<String> = self.store.schema.keys().cloned().collect();
        col_names.sort();

        let (indices, scores): (Vec<usize>, Vec<f32>) = aggregated.iter().cloned().unzip();

        // Materialize typed columns with values copied from source columns
        let mut data: HashMap<String, Column> = HashMap::with_capacity(col_names.len());
        for name in &col_names {
            if let Some(src) = self.store.columns.get(name) {
                let mut dst = Column::new(name, src.dtype());
                match src.dtype() {
                    DataType::Int32 => {
                        let vals = src.i32_values().unwrap();
                        let nulls = src.null_mask();
                        indices
                            .iter()
                            .try_for_each(|&gi| {
                                if nulls.get(gi).map(|b| *b).unwrap_or(false) {
                                    dst.push(Option::<i32>::None)
                                } else {
                                    dst.push(vals[gi])
                                }
                            })
                            .map_err(|e| e.to_string())?;
                    }
                    DataType::Int64 => {
                        let vals = src.i64_values().unwrap();
                        let nulls = src.null_mask();
                        indices
                            .iter()
                            .try_for_each(|&gi| {
                                if nulls.get(gi).map(|b| *b).unwrap_or(false) {
                                    dst.push(Option::<i64>::None)
                                } else {
                                    dst.push(vals[gi])
                                }
                            })
                            .map_err(|e| e.to_string())?;
                    }
                    DataType::Float32 => {
                        let vals = src.f32_values().unwrap();
                        let nulls = src.null_mask();
                        indices
                            .iter()
                            .try_for_each(|&gi| {
                                if nulls.get(gi).map(|b| *b).unwrap_or(false) {
                                    dst.push(Option::<f32>::None)
                                } else {
                                    dst.push(vals[gi])
                                }
                            })
                            .map_err(|e| e.to_string())?;
                    }
                    DataType::Float64 => {
                        let vals = src.f64_values().unwrap();
                        let nulls = src.null_mask();
                        indices
                            .iter()
                            .try_for_each(|&gi| {
                                if nulls.get(gi).map(|b| *b).unwrap_or(false) {
                                    dst.push(Option::<f64>::None)
                                } else {
                                    dst.push(vals[gi])
                                }
                            })
                            .map_err(|e| e.to_string())?;
                    }
                    DataType::String => {
                        let vals = src.string_values().unwrap();
                        let nulls = src.null_mask();
                        indices
                            .iter()
                            .try_for_each(|&gi| {
                                if nulls.get(gi).map(|b| *b).unwrap_or(false) {
                                    dst.push(Option::<&str>::None)
                                } else {
                                    dst.push(vals[gi].as_str())
                                }
                            })
                            .map_err(|e| e.to_string())?;
                    }
                    DataType::DateTime => {
                        let vals = src.datetime_values().unwrap();
                        let nulls = src.null_mask();
                        indices
                            .iter()
                            .try_for_each(|&gi| {
                                if nulls.get(gi).map(|b| *b).unwrap_or(false) {
                                    dst.push(crate::col::ColumnValue::DateTime(None))
                                } else {
                                    dst.push(crate::col::ColumnValue::DateTime(Some(vals[gi])))
                                }
                            })
                            .map_err(|e| e.to_string())?;
                    }
                }
                data.insert(name.clone(), dst);
            }
        }

        Ok(MetaQueryResults {
            columns: col_names,
            data,
            indices,
            scores,
        })
    }
}

#[derive(Debug, Clone)]
pub struct MetaQueryStats {
    pub total_chunks: usize,
    pub pruned_chunks: usize,
    pub evaluated_chunks: usize,
    pub vectors_compared: usize,
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
