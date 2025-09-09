//! Lightweight ASCII table rendering for human-friendly output
//!
//! Internal helpers used to format MetaStore heads, query results, and stats
//! as compact tables for examples and demos.
use crate::col::Column;
use crate::meta::{MetaBuildStats, MetaQueryResults, MetaQueryStats, MetaStore};
use crate::type_utils::DataType;
use chrono::DateTime;
use std::fmt;

/// Minimal ASCII table helper used for pretty printing results and heads.
pub struct AsciiTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    title: Option<String>,
}

impl AsciiTable {
    pub fn new(headers: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        Self {
            headers,
            rows,
            title: None,
        }
    }

    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }

    pub fn render(&self) -> String {
        if self.headers.is_empty() {
            return String::new();
        }

        let cols = self.headers.len();
        let mut widths: Vec<usize> = vec![0; cols];

        for (i, h) in self.headers.iter().enumerate() {
            widths[i] = widths[i].max(h.len());
        }
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate().take(cols) {
                widths[i] = widths[i].max(cell.len());
            }
        }

        let sep = {
            let mut s = String::from("+");
            for w in &widths {
                s.push_str(&"-".repeat(*w + 2));
                s.push('+');
            }
            s
        };

        let mut out = String::new();
        if let Some(t) = &self.title {
            out.push_str(t);
            out.push('\n');
        }
        out.push_str(&sep);
        out.push('\n');

        // Header row
        out.push('|');
        for (i, h) in self.headers.iter().enumerate() {
            out.push(' ');
            out.push_str(h);
            out.push_str(&" ".repeat(widths[i] - h.len() + 1));
            out.push('|');
        }
        out.push('\n');
        out.push_str(&sep);
        out.push('\n');

        // Data rows
        for row in &self.rows {
            out.push('|');
            for (i, _) in widths.iter().enumerate().take(cols) {
                let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
                out.push(' ');
                out.push_str(cell);
                out.push_str(&" ".repeat(widths[i] - cell.len() + 1));
                out.push('|');
            }
            out.push('\n');
        }
        out.push_str(&sep);
        out
    }
}

impl fmt::Display for AsciiTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// Render a single cell from a Column at row index `i` into a String
fn format_cell(col: &Column, i: usize) -> String {
    let is_null = col.null_mask().get(i).map(|b| *b).unwrap_or(false);
    if is_null {
        return "NULL".to_string();
    }

    match col.dtype() {
        DataType::Int32 => col.i32_values().map(|v| v[i].to_string()).unwrap(),
        DataType::Int64 => col.i64_values().map(|v| v[i].to_string()).unwrap(),
        DataType::Float32 => col.f32_values().map(|v| format!("{:.4}", v[i])).unwrap(),
        DataType::Float64 => col.f64_values().map(|v| format!("{:.4}", v[i])).unwrap(),
        DataType::String => col.string_values().map(|v| v[i].clone()).unwrap(),
        DataType::DateTime => col
            .datetime_values()
            .map(|v| {
                DateTime::from_timestamp_millis(v[i])
                    .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                    .unwrap_or_else(|| format!("{}", v[i]))
            })
            .unwrap(),
    }
}

/// Pretty-print the head of a MetaStore as an ASCII table.
pub fn metastore_head(meta: &MetaStore, n: usize) -> String {
    let mut cols: Vec<String> = meta.schema().keys().cloned().collect();
    cols.sort();

    // Determine row count
    let total_rows = meta.columns().values().next().map(|c| c.len()).unwrap_or(0);
    let limit = total_rows.min(n);

    // Build headers
    let mut headers: Vec<String> = Vec::with_capacity(cols.len() + 1);
    headers.push("index".to_string());
    headers.extend(cols.iter().cloned());

    // Build rows
    let mut rows: Vec<Vec<String>> = Vec::with_capacity(limit);
    for i in 0..limit {
        let mut row: Vec<String> = Vec::with_capacity(headers.len());
        row.push(i.to_string());
        for name in &cols {
            if let Some(col) = meta.columns().get(name) {
                row.push(format_cell(col, i));
            } else {
                row.push(String::new());
            }
        }
        rows.push(row);
    }

    AsciiTable::new(headers, rows)
        .with_title(format!(
            "MetaStore • rows={} • chunks={} • chunk_size={}",
            total_rows,
            meta.n_chunks(),
            meta.chunk_size()
        ))
        .render()
}

impl fmt::Display for MetaQueryResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut headers = vec!["index".to_string(), "score".to_string()];
        headers.extend(self.columns.iter().cloned());

        let mut rows: Vec<Vec<String>> = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            let mut line = vec![
                self.indices[i].to_string(),
                format!("{:.6}", self.scores[i]),
            ];
            for c in &self.columns {
                if let Some(col) = self.data.get(c) {
                    line.push(format_cell(col, i));
                } else {
                    line.push(String::new());
                }
            }
            rows.push(line);
        }

        let table = AsciiTable::new(headers, rows).render();
        write!(f, "{table}")
    }
}

impl fmt::Debug for MetaQueryResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

/// Build stats table
pub fn format_build_stats(b: &MetaBuildStats) -> String {
    let headers = vec!["metric".to_string(), "value".to_string()];
    let rows = vec![
        vec!["rows".into(), b.n_rows.to_string()],
        vec!["dimensions".into(), b.dim.to_string()],
        vec!["chunks".into(), b.n_chunks.to_string()],
        vec![
            "vector_ingest_ms".into(),
            format!("{:.3}", b.vectors_ingest_duration.as_secs_f64() * 1000.0),
        ],
        vec![
            "zonemap_build_ms".into(),
            format!("{:.3}", b.zonemap_build_duration.as_secs_f64() * 1000.0),
        ],
        vec![
            "build_total_ms".into(),
            format!("{:.3}", b.build_total_duration.as_secs_f64() * 1000.0),
        ],
    ];
    AsciiTable::new(headers, rows)
        .with_title("MetaStore Build Stats".to_string())
        .to_string()
}

/// Query stats table
pub fn format_query_stats(s: &MetaQueryStats) -> String {
    let headers = vec!["metric".to_string(), "value".to_string()];
    let rows = vec![
        vec!["total_chunks".into(), s.total_chunks.to_string()],
        vec!["pruned_chunks".into(), s.pruned_chunks.to_string()],
        vec!["evaluated_chunks".into(), s.evaluated_chunks.to_string()],
        vec!["vectors_compared".into(), s.vectors_compared.to_string()],
        vec![
            "prune_ms".into(),
            format!("{:.3}", s.prune_duration.as_secs_f64() * 1000.0),
        ],
        vec![
            "score_ms".into(),
            format!("{:.3}", s.score_duration.as_secs_f64() * 1000.0),
        ],
        vec![
            "merge_ms".into(),
            format!("{:.3}", s.merge_duration.as_secs_f64() * 1000.0),
        ],
        vec![
            "total_ms".into(),
            format!("{:.3}", s.total_duration.as_secs_f64() * 1000.0),
        ],
    ];
    AsciiTable::new(headers, rows)
        .with_title("Last Meta Query Stats".to_string())
        .to_string()
}
