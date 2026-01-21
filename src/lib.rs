//! Profundo - Memory system for Clawdbot
//!
//! Semantic search and learning extraction from Clawdbot session logs.
//! Named for "the deep" (Spanish: profundo) - where memories sink and are retrieved from.

pub mod db;
pub mod embed;
pub mod harvest;
pub mod openrouter;
pub mod recall;
pub mod session;
pub mod stats;

use std::path::PathBuf;

/// Default paths for Clawdbot integration
pub struct Paths {
    /// Where Clawdbot stores session logs
    pub sessions_dir: PathBuf,
    /// Where the workspace memory lives
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
        let home = PathBuf::from(&home);

        // Read workspace from clawdbot.json, fall back to ~/clawd (Clawdbot default)
        let workspace = read_clawdbot_workspace(&home)
            .unwrap_or_else(|| home.join("clawd"));

        let memory_dir = workspace.join("memory");

        Self {
            sessions_dir: home.join(".clawdbot/agents/main/sessions"),
            db_path: memory_dir.join("profundo.sqlite"),
            learnings_path: memory_dir.join("learnings.jsonl"),
            cursor_path: memory_dir.join(".profundo-cursor"),
            memory_dir,
        }
    }
}

/// Read workspace path from clawdbot.json
fn read_clawdbot_workspace(home: &PathBuf) -> Option<PathBuf> {
    let config_path = home.join(".clawdbot/clawdbot.json");
    let content = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&content).ok()?;

    config
        .get("agents")
        .and_then(|a| a.get("defaults"))
        .and_then(|d| d.get("workspace"))
        .and_then(|w| w.as_str())
        .map(PathBuf::from)
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
