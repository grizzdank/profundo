//! Embedding pipeline
//!
//! Reads session logs, chunks them, generates embeddings, and stores in SQLite.

use anyhow::{Context, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use walkdir::WalkDir;

use crate::db::Database;
use crate::openrouter::OpenRouterClient;
use crate::session::Session;
use crate::Paths;

/// Configuration for the embedding pipeline
pub struct EmbedConfig {
    /// Number of conversation turns per chunk
    pub chunk_size: usize,
    /// Number of turns overlap between chunks
    pub overlap: usize,
    /// Process all sessions, even if already processed
    pub force_reprocess: bool,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            chunk_size: 3,
            overlap: 1,
            force_reprocess: false,
        }
    }
}

/// Run the embedding pipeline
pub async fn run(paths: &Paths, config: EmbedConfig) -> Result<EmbedStats> {
    let client = OpenRouterClient::from_env()?;
    let mut db = Database::open(&paths.db_path)?;

    let sessions = discover_sessions(&paths.sessions_dir)?;
    println!(
        "{} Found {} session files",
        "→".blue(),
        sessions.len().to_string().cyan()
    );

    let mut stats = EmbedStats::default();
    let mut to_process = Vec::new();

    // Filter sessions that need processing
    for (session_id, path, size, mtime) in &sessions {
        if config.force_reprocess || !db.is_session_processed(session_id, *size, *mtime)? {
            to_process.push((session_id.clone(), path.clone(), *size, *mtime));
        } else {
            stats.skipped += 1;
        }
    }

    if to_process.is_empty() {
        println!("{} All sessions already processed", "✓".green());
        return Ok(stats);
    }

    println!(
        "{} Processing {} sessions ({} skipped)",
        "→".blue(),
        to_process.len().to_string().cyan(),
        stats.skipped.to_string().yellow()
    );

    let pb = ProgressBar::new(to_process.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    for (session_id, path, size, mtime) in to_process {
        pb.set_message(format!("{}", session_id));

        match process_session(&mut db, &client, &path, &session_id, size, mtime, &config).await {
            Ok(chunks_count) => {
                stats.processed += 1;
                stats.chunks_created += chunks_count;
            }
            Err(e) => {
                stats.errors += 1;
                eprintln!(
                    "\n{} Error processing {}: {}",
                    "✗".red(),
                    session_id,
                    e
                );
            }
        }

        pb.inc(1);
    }

    pb.finish_with_message("done");

    println!(
        "\n{} Processed {} sessions, created {} chunks ({} errors)",
        "✓".green(),
        stats.processed.to_string().cyan(),
        stats.chunks_created.to_string().cyan(),
        stats.errors.to_string().red()
    );

    Ok(stats)
}

/// Process a single session file
async fn process_session(
    db: &mut Database,
    client: &OpenRouterClient,
    path: &Path,
    session_id: &str,
    size: u64,
    mtime: i64,
    config: &EmbedConfig,
) -> Result<usize> {
    // Parse session
    let session = Session::from_file(path)?;

    // Extract chunks
    let chunks = session.extract_text_chunks(config.chunk_size, config.overlap);

    if chunks.is_empty() {
        // Still mark as processed to avoid re-checking
        db.store_chunks(
            session_id,
            path.to_str().unwrap_or(""),
            size,
            mtime,
            &[],
        )?;
        return Ok(0);
    }

    // Generate embeddings in batch
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = client.embed_batch(&texts).await?;

    // Pair chunks with embeddings
    let chunk_embeddings: Vec<_> = chunks
        .into_iter()
        .zip(embeddings.into_iter())
        .collect();

    // Store in database
    db.store_chunks(
        session_id,
        path.to_str().unwrap_or(""),
        size,
        mtime,
        &chunk_embeddings,
    )?;

    Ok(chunk_embeddings.len())
}

/// Discover all session files
fn discover_sessions(sessions_dir: &Path) -> Result<Vec<(String, std::path::PathBuf, u64, i64)>> {
    let mut sessions = Vec::new();

    for entry in WalkDir::new(sessions_dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();

        // Only process .jsonl files (not .deleted ones)
        if path.extension().map(|e| e == "jsonl").unwrap_or(false)
            && !path
                .to_str()
                .map(|s| s.contains(".deleted"))
                .unwrap_or(false)
        {
            let session_id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            let metadata = std::fs::metadata(path)
                .context("Failed to get file metadata")?;

            let mtime = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);

            sessions.push((session_id, path.to_path_buf(), metadata.len(), mtime));
        }
    }

    // Sort by mtime (oldest first for consistent processing)
    sessions.sort_by_key(|(_, _, _, mtime)| *mtime);

    Ok(sessions)
}

#[derive(Default)]
pub struct EmbedStats {
    pub processed: usize,
    pub skipped: usize,
    pub chunks_created: usize,
    pub errors: usize,
}
