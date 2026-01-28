//! Recall - semantic search across session memory
//!
//! Embeds a query and finds similar chunks from past conversations.

use anyhow::Result;
use colored::Colorize;
use std::collections::HashMap;
use std::path::Path;

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
    /// Use LLM to expand query with synonyms/variants before searching
    pub expand: bool,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            threshold: 0.3,
            semantic_only: false,
            show_full: false,
            context_turns: None,
            expand: false,
        }
    }
}

/// Expand a query using LLM to generate synonyms/variants
async fn expand_query(client: &OpenRouterClient, query: &str) -> Result<Vec<String>> {
    let system_prompt = "You generate alternative search queries. Return only the queries, one per line. No explanations, no numbering.";
    let user_prompt = format!(
        "Generate 2 alternative search queries for: {}\n\nReturn only the queries, one per line.",
        query
    );

    // Use a cheap, fast model for query expansion
    let model = "deepseek/deepseek-chat";

    match client.chat(system_prompt, &user_prompt, model).await {
        Ok(response) => {
            let variants: Vec<String> = response
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .take(2)
                .map(|s| s.to_string())
                .collect();
            Ok(variants)
        }
        Err(e) => {
            eprintln!("  {} Query expansion failed: {}", "⚠".yellow(), e);
            Ok(Vec::new())
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

    // Optionally expand the query with LLM-generated variants
    let queries: Vec<String> = if config.expand {
        let mut all_queries = vec![query.to_string()];
        eprintln!("  {} Expanding query...", "→".blue());
        let variants = expand_query(&client, query).await?;
        if !variants.is_empty() {
            eprintln!("  {} Variants: {}", "✓".green(), variants.join(", "));
        }
        all_queries.extend(variants);
        all_queries
    } else {
        vec![query.to_string()]
    };

    // Embed the primary query for semantic scoring
    let query_embedding = client.embed(query).await?;

    if config.semantic_only {
        // Legacy path: load all chunks, brute-force cosine similarity
        let chunks = db.load_all_chunks()?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        return semantic_only_search(chunks, &query_embedding, config);
    }

    hybrid_search_expanded(&db, &query_embedding, &queries, config)
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

/// BM25-first hybrid search with multiple query variants.
///
/// Runs BM25 for each query variant and merges candidate pools before
/// semantic scoring. This allows query expansion to find results that
/// match synonyms/variants of the original query.
fn hybrid_search_expanded(
    db: &Database,
    query_embedding: &[f32],
    queries: &[String],
    config: RecallConfig,
) -> Result<Vec<SearchResult>> {
    // BM25 candidate pool — cast a wide net, per query
    let candidate_pool_size_per_query = (config.top_k * 40).max(200);

    // Merge BM25 results from all query variants
    let mut all_bm25: HashMap<i64, f32> = HashMap::new();
    for q in queries {
        let lexical = db.bm25_search(q, candidate_pool_size_per_query).unwrap_or_default();
        for (rowid, rank) in lexical {
            // Keep the best (lowest) rank for each rowid
            all_bm25
                .entry(rowid)
                .and_modify(|existing| {
                    if rank < *existing {
                        *existing = rank;
                    }
                })
                .or_insert(rank);
        }
    }

    // Build lexical ranking from merged results (sorted by rank)
    let mut lexical: Vec<(i64, f32)> = all_bm25.into_iter().collect();
    lexical.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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

    // Semantic ranking over candidates only (always against original query embedding)
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
pub fn display_results(
    paths: &Paths,
    learnings_path: &Path,
    results: &[SearchResult],
    query: &str,
    config: &RecallConfig,
) {
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

    // Cache parsed sessions to avoid re-parsing the same file per result
    let mut session_cache: HashMap<String, Vec<crate::session::Turn>> = HashMap::new();

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

        let id_display = {
            let sid = &result.chunk.session_id;
            if sid.len() >= 8 { &sid[..8] } else { sid.as_str() }
        };

        println!(
            "{}. {} [{}] ({})",
            (i + 1).to_string().bold(),
            date.cyan(),
            id_display.dimmed(),
            similarity_color
        );

        if let Some(context) = config.context_turns {
            display_with_context(paths, result, context, &mut session_cache);
        } else if config.show_full {
            for line in result.chunk.text.lines() {
                println!("   {}", line);
            }
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

    if let Ok(db) = Database::open_with_learnings(&paths.db_path, learnings_path) {
        if let Ok(learnings) = db.search_learnings(query, 3) {
            if !learnings.is_empty() {
                println!("📝 Related Learnings:\n");

                for (learning, _rank) in learnings {
                    let id_display = if learning.session_id.len() >= 8 {
                        &learning.session_id[..8]
                    } else {
                        learning.session_id.as_str()
                    };

                    println!("● {} [{}]", learning.date.cyan(), id_display.dimmed());

                    if !learning.topics.is_empty() {
                        println!("  Topics: {}", learning.topics.join(", "));
                    }
                    if !learning.facts_learned.is_empty() {
                        println!("  Facts: {}", learning.facts_learned.join(" "));
                    }
                    if !learning.decisions.is_empty() {
                        println!("  Decisions: {}", learning.decisions.join(" "));
                    }
                    if !learning.action_items.is_empty() {
                        println!("  Action Items: {}", learning.action_items.join(" "));
                    }
                    if !learning.summary.trim().is_empty() {
                        println!("  Summary: {}", learning.summary);
                    }

                    println!();
                }
            }
        }
    }
}

/// Display a result with surrounding context turns from the session file.
/// Uses a session cache to avoid re-parsing the same file multiple times.
fn display_with_context(
    paths: &Paths,
    result: &SearchResult,
    context: usize,
    session_cache: &mut HashMap<String, Vec<crate::session::Turn>>,
) {
    // Validate turn indices
    let (turn_start, turn_end) = match (
        usize::try_from(result.chunk.turn_start),
        usize::try_from(result.chunk.turn_end),
    ) {
        (Ok(s), Ok(e)) => (s, e),
        _ => {
            println!(
                "   {}: invalid turn indices ({}, {}), showing stored chunk.",
                "Warning".yellow(),
                result.chunk.turn_start,
                result.chunk.turn_end
            );
            display_chunk_indented(&result.chunk.text);
            return;
        }
    };

    let session_id = &result.chunk.session_id;

    // Load turns from cache or parse session file
    let turns = match session_cache.get(session_id) {
        Some(cached) => cached,
        None => {
            let session_path = paths.sessions_dir.join(format!("{}.jsonl", session_id));
            if !session_path.exists() {
                println!(
                    "   {}: session file not found, showing stored chunk.",
                    "Warning".yellow()
                );
                display_chunk_indented(&result.chunk.text);
                return;
            }
            match Session::from_file(&session_path) {
                Ok(session) => {
                    session_cache.insert(session_id.clone(), session.get_turns());
                    session_cache.get(session_id).unwrap()
                }
                Err(e) => {
                    println!(
                        "   {}: failed to parse session: {}, showing stored chunk.",
                        "Warning".yellow(),
                        e
                    );
                    display_chunk_indented(&result.chunk.text);
                    return;
                }
            }
        }
    };

    if turns.is_empty() || turn_start >= turns.len() {
        println!(
            "   {}: turn indices out of range (session has {} turns), showing stored chunk.",
            "Warning".yellow(),
            turns.len()
        );
        display_chunk_indented(&result.chunk.text);
        return;
    }

    let start = turn_start.saturating_sub(context);
    let end = (turn_end + context).min(turns.len());
    let slice = &turns[start..end];

    if slice.is_empty() {
        display_chunk_indented(&result.chunk.text);
        return;
    }

    // Display with visual markers for matched vs context turns
    for (j, turn) in slice.iter().enumerate() {
        let turn_idx = start + j;
        let is_match = turn_idx >= turn_start && turn_idx < turn_end;
        let marker = if is_match { "█" } else { "░" };

        for line in turn.user_text.lines() {
            println!("   {} {}: {}", marker.dimmed(), "User".cyan(), line);
        }
        for line in turn.assistant_text.lines() {
            println!("   {} {}: {}", marker.dimmed(), "Asst".green(), line);
        }
        if j < slice.len() - 1 {
            println!("   {}", "---".dimmed());
        }
    }
}

/// Helper to display chunk text with consistent indentation
fn display_chunk_indented(text: &str) {
    for line in text.lines() {
        println!("   {}", line.dimmed());
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
