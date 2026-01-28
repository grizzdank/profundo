//! Recall - semantic search across session memory
//!
//! Embeds a query and finds similar chunks from past conversations.

use anyhow::Result;
use colored::Colorize;
use std::collections::HashMap;

use crate::db::{Database, StoredChunk};
use crate::openrouter::OpenRouterClient;
use crate::session::Session;
use crate::Paths;

/// Search result with similarity score
#[derive(Debug)]
pub struct SearchResult {
    pub chunk: StoredChunk,
    pub similarity: f32,
}

/// Configuration for recall search
#[derive(Clone, Debug)]
pub struct RecallConfig {
    /// Number of results to return
    pub top_k: usize,
    /// Minimum similarity threshold (0.0 - 1.0)
    pub threshold: f32,
    /// If true, skip lexical search and use pure semantic recall.
    /// Not exposed via CLI; can be toggled via PROFUNDO_SEMANTIC_ONLY=1.
    pub semantic_only: bool,
    /// specific display options
    pub show_full: bool,
    pub context_turns: Option<usize>,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            threshold: 0.3,
            semantic_only: false,
            show_full: false,
            context_turns: None,
        }
    }
}

/// Search for similar content in memory
pub async fn search(paths: &Paths, query: &str, mut config: RecallConfig) -> Result<Vec<SearchResult>> {
    if std::env::var("PROFUNDO_SEMANTIC_ONLY").ok().as_deref() == Some("1") {
        config.semantic_only = true;
    }

    let client = OpenRouterClient::from_env()?;
    let db = Database::open(&paths.db_path)?;

    // Embed the query
    let query_embedding = client.embed(query).await?;

    if config.semantic_only {
        // Legacy path: load all chunks, brute-force cosine similarity
        let chunks = db.load_all_chunks()?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        return semantic_only_search(chunks, &query_embedding, config);
    }

    hybrid_search(&db, &query_embedding, query, config)
}

fn semantic_only_search(
    chunks: Vec<StoredChunk>,
    query_embedding: &[f32],
    config: RecallConfig,
) -> Result<Vec<SearchResult>> {
    let mut results: Vec<SearchResult> = chunks
        .into_iter()
        .map(|chunk| {
            let similarity = cosine_similarity(query_embedding, &chunk.embedding);
            SearchResult { chunk, similarity }
        })
        .filter(|r| r.similarity >= config.threshold)
        .collect();

    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    results.truncate(config.top_k);
    Ok(results)
}

/// BM25-first hybrid search.
///
/// Instead of loading all chunks for brute-force cosine similarity,
/// we use BM25 to get a candidate set, then only load embeddings for
/// those candidates. This is O(candidate_pool) instead of O(all_chunks).
///
/// At 1,400 chunks this doesn't matter much; at 50k+ it's the difference
/// between instant and sluggish.
fn hybrid_search(
    db: &Database,
    query_embedding: &[f32],
    query: &str,
    config: RecallConfig,
) -> Result<Vec<SearchResult>> {
    // BM25 candidate pool — cast a wide net
    let candidate_pool_size = (config.top_k * 40).max(200);
    let lexical = db.bm25_search(query, candidate_pool_size).unwrap_or_default();

    // Collect BM25 candidate rowids
    let bm25_rowids: Vec<i64> = lexical.iter().map(|(rowid, _)| *rowid).collect();

    // Load only candidate chunks (embeddings included) for semantic scoring
    let candidates = if bm25_rowids.is_empty() {
        // BM25 returned nothing (e.g., query terms not in corpus) — fall back to full scan
        db.load_all_chunks()?
    } else {
        db.load_chunks_by_rowids(&bm25_rowids)?
    };

    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Semantic ranking over candidates only
    let mut semantic: Vec<(i64, f32)> = candidates
        .iter()
        .map(|c| (c.rowid, cosine_similarity(query_embedding, &c.embedding)))
        .filter(|(_, sim)| *sim >= config.threshold)
        .collect();

    semantic.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Build lookup maps
    let mut by_rowid: HashMap<i64, StoredChunk> = HashMap::with_capacity(candidates.len());
    for c in candidates {
        by_rowid.insert(c.rowid, c);
    }

    let mut sem_rank: HashMap<i64, usize> = HashMap::new();
    let mut sem_sim: HashMap<i64, f32> = HashMap::new();
    for (i, (rowid, sim)) in semantic.iter().enumerate() {
        sem_rank.insert(*rowid, i + 1);
        sem_sim.insert(*rowid, *sim);
    }

    let mut lex_rank: HashMap<i64, usize> = HashMap::new();
    for (i, (rowid, _rank)) in lexical.iter().enumerate() {
        lex_rank.insert(*rowid, i + 1);
    }

    // Reciprocal Rank Fusion
    const RRF_K: f32 = 60.0;
    const FALLBACK_RANK: f32 = 10_000.0;

    let mut fused: Vec<(f32, i64)> = Vec::new();

    for rowid in sem_rank
        .keys()
        .chain(lex_rank.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
    {
        let sr = sem_rank.get(&rowid).copied().map(|r| r as f32).unwrap_or(FALLBACK_RANK);
        let br = lex_rank.get(&rowid).copied().map(|r| r as f32).unwrap_or(FALLBACK_RANK);
        let score = 1.0 / (RRF_K + sr) + 1.0 / (RRF_K + br);
        fused.push((score, rowid));
    }

    fused.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut out: Vec<SearchResult> = Vec::new();
    for (_score, rowid) in fused.into_iter().take(config.top_k) {
        if let Some(chunk) = by_rowid.remove(&rowid) {
            let similarity = sem_sim.get(&rowid).copied().unwrap_or(0.0);
            out.push(SearchResult { chunk, similarity });
        }
    }

    Ok(out)
}

/// Display search results in a nice format
pub fn display_results(paths: &Paths, results: &[SearchResult], query: &str, config: &RecallConfig) {
    if results.is_empty() {
        println!(
            "{} No results found for: {}",
            "→".yellow(),
            query.italic()
        );
        return;
    }

    println!(
        "\n{} Found {} results for: {}\n",
        "→".blue(),
        results.len().to_string().cyan(),
        query.italic()
    );

    for (i, result) in results.iter().enumerate() {
        let date = result
            .chunk
            .timestamp
            .as_ref()
            .and_then(|t| t.split('T').next())
            .unwrap_or("unknown");

        let similarity_pct = (result.similarity * 100.0) as i32;
        let similarity_color = if similarity_pct >= 80 {
            format!("{}%", similarity_pct).green()
        } else if similarity_pct >= 60 {
            format!("{}%", similarity_pct).yellow()
        } else {
            format!("{}%", similarity_pct).red()
        };

        println!(
            "{}. {} [{}] ({})",
            (i + 1).to_string().bold(),
            date.cyan(),
            result.chunk.session_id[..8].to_string().dimmed(),
            similarity_color
        );

        if let Some(context) = config.context_turns {
            let session_path = paths.sessions_dir.join(format!("{}.jsonl", result.chunk.session_id));
            if session_path.exists() {
                match Session::load_turn_range(
                    &session_path, 
                    result.chunk.turn_start.try_into().unwrap_or(0), 
                    result.chunk.turn_end.try_into().unwrap_or(0), 
                    context
                ) {
                    Ok(text) => {
                        println!("{}", text);
                    },
                    Err(e) => {
                        println!("   {}: {}", "Error loading context".red(), e);
                        println!("{}", result.chunk.text);
                    }
                }
            } else {
                println!("   {}: Session file not found, showing saved chunk.", "Warning".yellow());
                println!("{}", result.chunk.text);
            }
        } else if config.show_full {
            println!("{}", result.chunk.text);
        } else {
            // Truncate and display text preview
            let preview = truncate_text(&result.chunk.text, 300);
            for line in preview.lines().take(6) {
                println!("   {}", line.dimmed());
            }
            if result.chunk.text.lines().count() > 6 || result.chunk.text.len() > 300 {
                println!("   {}", "...".dimmed());
            }
        }

        println!();
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Truncate text to a maximum length
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
}
