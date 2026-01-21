//! Profundo - Memory system for Pulpito
//!
//! Semantic search and learning extraction from Clawdbot session logs.
//! Named for "the deep" - where memories sink and are retrieved from.

pub mod db;
pub mod embed;
pub mod harvest;
pub mod openrouter;
pub mod recall;
pub mod session;

use std::path::PathBuf;

/// Default paths for Clawdbot/Pulpito integration
pub struct Paths {
    /// Where Clawdbot stores session logs
    pub sessions_dir: PathBuf,
    /// Where Pulpito's memory lives
    pub memory_dir: PathBuf,
    /// Profundo's SQLite database
    pub db_path: PathBuf,
    /// Extracted learnings JSONL
    pub learnings_path: PathBuf,
    /// Cursor for incremental processing
    pub cursor_path: PathBuf,
}

impl Default for Paths {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        let home = PathBuf::from(home);

        Self {
            sessions_dir: home.join(".clawdbot/agents/main/sessions"),
            memory_dir: home.join("pulpito/memory"),
            db_path: home.join("pulpito/memory/profundo.sqlite"),
            learnings_path: home.join("pulpito/memory/learnings.jsonl"),
            cursor_path: home.join("pulpito/memory/.profundo-cursor"),
        }
    }
}

impl Paths {
    /// Create paths with custom base directories
    pub fn with_bases(sessions_dir: PathBuf, memory_dir: PathBuf) -> Self {
        Self {
            sessions_dir,
            db_path: memory_dir.join("profundo.sqlite"),
            learnings_path: memory_dir.join("learnings.jsonl"),
            cursor_path: memory_dir.join(".profundo-cursor"),
            memory_dir,
        }
    }
}
