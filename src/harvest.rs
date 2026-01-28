//! Harvest - extract learnings from session logs
//!
//! Uses AI to summarize sessions and extract key facts, decisions, and action items.

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use walkdir::WalkDir;

use crate::openrouter::OpenRouterClient;
use crate::session::Session;
use crate::Paths;

/// Extracted learnings from a session
#[derive(Debug, Serialize, Deserialize)]
pub struct Learning {
    pub session_id: String,
    pub date: String,
    pub topics: Vec<String>,
    pub decisions: Vec<String>,
    pub facts_learned: Vec<String>,
    pub action_items: Vec<String>,
    pub summary: String,
    pub message_count: usize,
    pub cost: f64,
    pub harvested_at: String,
}

/// Configuration for harvest
pub struct HarvestConfig {
    /// Only process sessions since this date
    pub since: Option<NaiveDate>,
    /// Model to use for extraction
    pub model: String,
    /// Minimum messages to process a session
    pub min_messages: usize,
}

impl Default for HarvestConfig {
    fn default() -> Self {
        Self {
            since: None,
            model: "deepseek/deepseek-v3.2".to_string(),
            min_messages: 4,
        }
    }
}

const HARVEST_PROMPT: &str = r#"Analyze this conversation and extract structured learnings.

Respond with ONLY valid JSON (no markdown, no explanation):
{
    "topics": ["topic1", "topic2"],
    "decisions": ["decision made"],
    "facts_learned": ["new fact about user"],
    "action_items": ["task to do"],
    "summary": "One paragraph summary"
}

Rules:
- topics: 2-5 keywords describing what was discussed
- decisions: Only explicit decisions made, not general discussion
- facts_learned: New information about the user (preferences, background, etc.)
- action_items: Tasks that were identified to do
- summary: Brief summary of the conversation's purpose and outcome
- Use empty arrays [] if nothing fits a category
- Be concise and specific

Conversation:
"#;

/// Run the harvest pipeline
pub async fn run(paths: &Paths, config: HarvestConfig) -> Result<HarvestStats> {
    let client = OpenRouterClient::from_env()?;

    // Load already harvested session IDs
    let harvested = load_harvested_ids(&paths.learnings_path)?;
    println!(
        "{} {} sessions already harvested",
        "→".blue(),
        harvested.len().to_string().cyan()
    );

    // Discover sessions to process
    let sessions = discover_sessions_for_harvest(&paths.sessions_dir, &config, &harvested)?;
    println!(
        "{} Found {} sessions to harvest",
        "→".blue(),
        sessions.len().to_string().cyan()
    );

    if sessions.is_empty() {
        println!("{} Nothing new to harvest", "✓".green());
        return Ok(HarvestStats::default());
    }

    let mut stats = HarvestStats::default();
    let mut learnings_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.learnings_path)
        .context("Failed to open learnings file")?;

    for (session_id, path) in sessions {
        let id_display = if session_id.len() >= 8 { &session_id[..8] } else { &session_id };
        print!("{} Harvesting {}... ", "→".blue(), id_display);
        std::io::stdout().flush().ok();

        match harvest_session(&client, &path, &session_id, &config).await {
            Ok(Some(learning)) => {
                // Write to JSONL
                let json = serde_json::to_string(&learning)?;
                writeln!(learnings_file, "{}", json)?;

                println!(
                    "{} topics, {} decisions, {} facts",
                    learning.topics.len().to_string().cyan(),
                    learning.decisions.len().to_string().cyan(),
                    learning.facts_learned.len().to_string().cyan()
                );
                stats.harvested += 1;
            }
            Ok(None) => {
                println!("{}", "skipped (too short)".dimmed());
                stats.skipped += 1;
            }
            Err(e) => {
                println!("{} {}", "error:".red(), e);
                stats.errors += 1;
            }
        }
    }

    println!(
        "\n{} Harvested {} sessions ({} skipped, {} errors)",
        "✓".green(),
        stats.harvested.to_string().cyan(),
        stats.skipped.to_string().yellow(),
        stats.errors.to_string().red()
    );

    Ok(stats)
}

/// Strip markdown code fences from LLM response
/// Handles ```json ... ``` and ``` ... ``` patterns
fn strip_markdown_json(response: &str) -> String {
    let trimmed = response.trim();

    // Check for ```json or ``` at start
    let without_prefix = if trimmed.starts_with("```json") {
        trimmed.strip_prefix("```json").unwrap_or(trimmed)
    } else if trimmed.starts_with("```") {
        trimmed.strip_prefix("```").unwrap_or(trimmed)
    } else {
        trimmed
    };

    // Check for ``` at end
    let without_suffix = if without_prefix.trim_end().ends_with("```") {
        without_prefix.trim_end().strip_suffix("```").unwrap_or(without_prefix)
    } else {
        without_prefix
    };

    without_suffix.trim().to_string()
}

/// Harvest a single session
async fn harvest_session(
    client: &OpenRouterClient,
    path: &Path,
    session_id: &str,
    config: &HarvestConfig,
) -> Result<Option<Learning>> {
    let session = Session::from_file(path)?;

    // Skip short sessions
    if session.message_count < config.min_messages {
        return Ok(None);
    }

    // Format conversation for AI
    let transcript = format_transcript(&session);

    // Truncate if too long (keep within context limits)
    let max_chars = 50000;
    let truncated = if transcript.len() > max_chars {
        format!("{}...\n[truncated]", &transcript[..max_chars])
    } else {
        transcript
    };

    // Call AI for extraction
    let response = client
        .chat("You are a helpful assistant that extracts structured information.",
              &format!("{}{}", HARVEST_PROMPT, truncated),
              &config.model)
        .await?;

    // Parse response as JSON (strip markdown code fences if present)
    let json_str = strip_markdown_json(&response);
    let extracted: ExtractedLearning = serde_json::from_str(&json_str)
        .context("Failed to parse AI response as JSON")?;

    let date = session
        .first_timestamp
        .map(|t| t.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| "unknown".to_string());

    Ok(Some(Learning {
        session_id: session_id.to_string(),
        date,
        topics: extracted.topics,
        decisions: extracted.decisions,
        facts_learned: extracted.facts_learned,
        action_items: extracted.action_items,
        summary: extracted.summary,
        message_count: session.message_count,
        cost: session.total_cost,
        harvested_at: Utc::now().to_rfc3339(),
    }))
}

#[derive(Deserialize)]
struct ExtractedLearning {
    topics: Vec<String>,
    decisions: Vec<String>,
    facts_learned: Vec<String>,
    action_items: Vec<String>,
    summary: String,
}

/// Format session as readable transcript
fn format_transcript(session: &Session) -> String {
    let mut lines = Vec::new();

    for msg in &session.messages {
        if msg.msg_type != "message" {
            continue;
        }

        let Some(ref content) = msg.message else {
            continue;
        };
        let Some(ref role) = content.role else {
            continue;
        };

        let text = content
            .content
            .as_ref()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|b| match b {
                        crate::session::ContentBlock::Text { text, .. } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();

        if !text.is_empty() {
            lines.push(format!("{}: {}", role.to_uppercase(), text));
        }
    }

    lines.join("\n\n")
}

/// Load session IDs that have already been harvested
fn load_harvested_ids(path: &Path) -> Result<std::collections::HashSet<String>> {
    let mut ids = std::collections::HashSet::new();

    if !path.exists() {
        return Ok(ids);
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if let Ok(learning) = serde_json::from_str::<Learning>(&line) {
            ids.insert(learning.session_id);
        }
    }

    Ok(ids)
}

/// Discover sessions that need harvesting
fn discover_sessions_for_harvest(
    sessions_dir: &Path,
    config: &HarvestConfig,
    already_harvested: &std::collections::HashSet<String>,
) -> Result<Vec<(String, std::path::PathBuf)>> {
    let mut sessions = Vec::new();

    for entry in WalkDir::new(sessions_dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();

        // Only .jsonl files, not deleted, not macOS resource forks
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !path.extension().map(|e| e == "jsonl").unwrap_or(false)
            || path.to_str().map(|s| s.contains(".deleted")).unwrap_or(false)
            || filename.starts_with("._")
        {
            continue;
        }

        let session_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Skip already harvested
        if already_harvested.contains(&session_id) {
            continue;
        }

        // Check date filter
        if let Some(since) = config.since {
            // Quick check: read first line to get timestamp
            if let Ok(file) = File::open(path) {
                let reader = BufReader::new(file);
                if let Some(Ok(first_line)) = reader.lines().next() {
                    if let Ok(msg) = serde_json::from_str::<crate::session::SessionMessage>(&first_line) {
                        if let Some(ts) = msg.timestamp {
                            if let Ok(dt) = DateTime::parse_from_rfc3339(&ts) {
                                if dt.date_naive() < since {
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }

        sessions.push((session_id, path.to_path_buf()));
    }

    // Sort by name (which includes date info)
    sessions.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(sessions)
}

#[derive(Default)]
pub struct HarvestStats {
    pub harvested: usize,
    pub skipped: usize,
    pub errors: usize,
}
