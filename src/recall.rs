//! Recall - semantic search across session memory
//!
//! Embeds a query and finds similar chunks from past conversations.

use anyhow::Result;
use colored::Colorize;

use crate::db::{Database, StoredChunk};
use crate::openrouter::OpenRouterClient;
use crate::Paths;

/// Search result with similarity score
#[derive(Debug)]
pub struct SearchResult {
    pub chunk: StoredChunk,
    pub similarity: f32,
}

/// Configuration for recall search
pub struct RecallConfig {
    /// Number of results to return
    pub top_k: usize,
    /// Minimum similarity threshold (0.0 - 1.0)
    pub threshold: f32,
}

impl Default for RecallConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            threshold: 0.3,
        }
    }
}

/// Search for similar content in memory
pub async fn search(paths: &Paths, query: &str, config: RecallConfig) -> Result<Vec<SearchResult>> {
    let client = OpenRouterClient::from_env()?;
    let db = Database::open(&paths.db_path)?;

    // Embed the query
    let query_embedding = client.embed(query).await?;

    // Load all chunks
    let chunks = db.load_all_chunks()?;

    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    // Compute similarities
    let mut results: Vec<SearchResult> = chunks
        .into_iter()
        .map(|chunk| {
            let similarity = cosine_similarity(&query_embedding, &chunk.embedding);
            SearchResult { chunk, similarity }
        })
        .filter(|r| r.similarity >= config.threshold)
        .collect();

    // Sort by similarity (descending)
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

    // Take top_k
    results.truncate(config.top_k);

    Ok(results)
}

/// Display search results in a nice format
pub fn display_results(results: &[SearchResult], query: &str) {
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

        // Truncate and display text preview
        let preview = truncate_text(&result.chunk.text, 300);
        for line in preview.lines().take(6) {
            println!("   {}", line.dimmed());
        }
        if result.chunk.text.lines().count() > 6 {
            println!("   {}", "...".dimmed());
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
