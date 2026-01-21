//! Session log parsing
//!
//! Parses Clawdbot's JSONL session format into structured data.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// A single message in a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub timestamp: Option<String>,
    pub message: Option<MessageContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageContent {
    pub role: Option<String>,
    pub content: Option<Vec<ContentBlock>>,
    pub model: Option<String>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentBlock {
    Text {
        #[serde(rename = "type")]
        block_type: String,
        text: String,
    },
    ToolCall {
        #[serde(rename = "type")]
        block_type: String,
        name: Option<String>,
        input: Option<serde_json::Value>,
    },
    ToolResult {
        #[serde(rename = "type")]
        block_type: String,
        content: Option<String>,
    },
    Other(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input: Option<u64>,
    pub output: Option<u64>,
    #[serde(rename = "cacheRead")]
    pub cache_read: Option<u64>,
    #[serde(rename = "cacheWrite")]
    pub cache_write: Option<u64>,
    #[serde(rename = "totalTokens")]
    pub total_tokens: Option<u64>,
    pub cost: Option<Cost>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cost {
    pub input: Option<f64>,
    pub output: Option<f64>,
    #[serde(rename = "cacheRead")]
    pub cache_read: Option<f64>,
    #[serde(rename = "cacheWrite")]
    pub cache_write: Option<f64>,
    pub total: Option<f64>,
}

/// Aggregated token statistics
#[derive(Debug, Clone, Default)]
pub struct TokenStats {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub total_tokens: u64,
    pub input_cost: f64,
    pub output_cost: f64,
    pub cache_read_cost: f64,
    pub cache_write_cost: f64,
    pub total_cost: f64,
    pub message_count: usize,
}

impl TokenStats {
    pub fn cache_hit_rate(&self) -> f64 {
        let total_input = self.input_tokens + self.cache_read_tokens;
        if total_input == 0 {
            0.0
        } else {
            self.cache_read_tokens as f64 / total_input as f64
        }
    }
}

/// Model usage statistics
#[derive(Debug, Clone, Default)]
pub struct ModelStats {
    pub model: String,
    pub stats: TokenStats,
}

/// A parsed session with metadata
#[derive(Debug)]
pub struct Session {
    pub id: String,
    pub messages: Vec<SessionMessage>,
    pub first_timestamp: Option<DateTime<Utc>>,
    pub last_timestamp: Option<DateTime<Utc>>,
    pub total_cost: f64,
    pub message_count: usize,
    pub token_stats: TokenStats,
    pub models_used: Vec<String>,
}

impl Session {
    /// Parse a session from a JSONL file
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open session file")?;
        let reader = BufReader::new(file);

        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut messages = Vec::new();
        let mut total_cost = 0.0;
        let mut token_stats = TokenStats::default();
        let mut models_used = std::collections::HashSet::new();

        for line in reader.lines() {
            let line = line.context("Failed to read line")?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<SessionMessage>(&line) {
                Ok(msg) => {
                    // Accumulate stats from assistant messages
                    if let Some(ref content) = msg.message {
                        // Track model
                        if let Some(ref model) = content.model {
                            models_used.insert(model.clone());
                        }

                        // Accumulate token usage
                        if let Some(ref usage) = content.usage {
                            token_stats.input_tokens += usage.input.unwrap_or(0);
                            token_stats.output_tokens += usage.output.unwrap_or(0);
                            token_stats.cache_read_tokens += usage.cache_read.unwrap_or(0);
                            token_stats.cache_write_tokens += usage.cache_write.unwrap_or(0);
                            token_stats.total_tokens += usage.total_tokens.unwrap_or(0);
                            token_stats.message_count += 1;

                            if let Some(ref cost) = usage.cost {
                                token_stats.input_cost += cost.input.unwrap_or(0.0);
                                token_stats.output_cost += cost.output.unwrap_or(0.0);
                                token_stats.cache_read_cost += cost.cache_read.unwrap_or(0.0);
                                token_stats.cache_write_cost += cost.cache_write.unwrap_or(0.0);
                                token_stats.total_cost += cost.total.unwrap_or(0.0);
                                total_cost += cost.total.unwrap_or(0.0);
                            }
                        }
                    }
                    messages.push(msg);
                }
                Err(e) => {
                    // Log but continue - some lines might be malformed
                    eprintln!("Warning: Failed to parse line in {}: {}", id, e);
                }
            }
        }

        let first_timestamp = messages
            .first()
            .and_then(|m| m.timestamp.as_ref())
            .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
            .map(|dt| dt.with_timezone(&Utc));

        let last_timestamp = messages
            .last()
            .and_then(|m| m.timestamp.as_ref())
            .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
            .map(|dt| dt.with_timezone(&Utc));

        let message_count = messages
            .iter()
            .filter(|m| m.msg_type == "message")
            .count();

        Ok(Self {
            id,
            messages,
            first_timestamp,
            last_timestamp,
            total_cost,
            message_count,
            token_stats,
            models_used: models_used.into_iter().collect(),
        })
    }

    /// Extract text content from messages for embedding
    pub fn extract_text_chunks(&self, chunk_size: usize, overlap: usize) -> Vec<TextChunk> {
        let mut chunks = Vec::new();

        // Collect conversation turns (user message + assistant response)
        let mut turns: Vec<Turn> = Vec::new();
        let mut current_turn: Option<Turn> = None;

        for msg in &self.messages {
            if msg.msg_type != "message" {
                continue;
            }

            let Some(ref content) = msg.message else {
                continue;
            };
            let Some(ref role) = content.role else {
                continue;
            };

            let text = Self::extract_text_from_content(content);
            if text.is_empty() {
                continue;
            }

            match role.as_str() {
                "user" => {
                    // Save previous turn if exists
                    if let Some(turn) = current_turn.take() {
                        turns.push(turn);
                    }
                    // Start new turn
                    current_turn = Some(Turn {
                        user_text: text,
                        assistant_text: String::new(),
                        timestamp: msg.timestamp.clone(),
                    });
                }
                "assistant" => {
                    if let Some(ref mut turn) = current_turn {
                        if !turn.assistant_text.is_empty() {
                            turn.assistant_text.push_str("\n\n");
                        }
                        turn.assistant_text.push_str(&text);
                    }
                }
                _ => {}
            }
        }

        // Don't forget the last turn
        if let Some(turn) = current_turn {
            turns.push(turn);
        }

        // Create chunks with sliding window
        let step = chunk_size.saturating_sub(overlap).max(1);
        for i in (0..turns.len()).step_by(step) {
            let end = (i + chunk_size).min(turns.len());
            let chunk_turns = &turns[i..end];

            if chunk_turns.is_empty() {
                continue;
            }

            let text = chunk_turns
                .iter()
                .map(|t| format!("User: {}\n\nAssistant: {}", t.user_text, t.assistant_text))
                .collect::<Vec<_>>()
                .join("\n\n---\n\n");

            let timestamp = chunk_turns
                .first()
                .and_then(|t| t.timestamp.clone());

            chunks.push(TextChunk {
                session_id: self.id.clone(),
                turn_start: i,
                turn_end: end,
                timestamp,
                text,
            });
        }

        chunks
    }

    fn extract_text_from_content(content: &MessageContent) -> String {
        let Some(ref blocks) = content.content else {
            return String::new();
        };

        blocks
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Debug)]
struct Turn {
    user_text: String,
    assistant_text: String,
    timestamp: Option<String>,
}

/// A chunk of text extracted from a session for embedding
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub session_id: String,
    pub turn_start: usize,
    pub turn_end: usize,
    pub timestamp: Option<String>,
    pub text: String,
}
