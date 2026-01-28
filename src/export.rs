//! Export - convert learnings to markdown for Clawdbot indexing
//!
//! Formats harvested learnings as markdown that Clawdbot can index via its
//! native memory search.

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate, Weekday};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::harvest::Learning;
use crate::session::TokenStats;
use crate::stats::{self, StatsConfig};
use crate::Paths;

/// Load all learnings from the JSONL file
pub fn load_learnings(path: &Path) -> Result<Vec<Learning>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);

    let learnings: Vec<Learning> = reader
        .lines()
        .filter_map(|l| l.ok())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();

    Ok(learnings)
}

/// Filter learnings by date
pub fn filter_by_date(learnings: &[Learning], date: NaiveDate) -> Vec<&Learning> {
    let date_str = date.format("%Y-%m-%d").to_string();
    learnings
        .iter()
        .filter(|l| l.date == date_str)
        .collect()
}

/// Format a single learning as markdown bullets
pub fn format_learning_bullets(learning: &Learning) -> String {
    let mut lines = Vec::new();

    // Topics as tags
    if !learning.topics.is_empty() {
        lines.push(format!("- **Topics**: {}", learning.topics.join(", ")));
    }

    // Summary
    if !learning.summary.is_empty() {
        lines.push(format!("- {}", learning.summary));
    }

    // Decisions
    for decision in &learning.decisions {
        lines.push(format!("- **Decision**: {}", decision));
    }

    // Facts learned
    for fact in &learning.facts_learned {
        lines.push(format!("- **Learned**: {}", fact));
    }

    // Action items
    for action in &learning.action_items {
        lines.push(format!("- [ ] {}", action));
    }

    lines.join("\n")
}

/// Format stats as a compact markdown summary
pub fn format_stats_summary(stats: &TokenStats) -> String {
    let cache_rate = stats.cache_hit_rate() * 100.0;
    format!(
        "**Tokens**: {:.1}K in / {:.1}K out | **Cache**: {:.0}% | **Cost**: ${:.4}",
        stats.input_tokens as f64 / 1000.0,
        stats.output_tokens as f64 / 1000.0,
        cache_rate,
        stats.total_cost
    )
}

/// Export all learnings to a markdown file
pub fn export_to_markdown(paths: &Paths, output_path: &Path) -> Result<ExportStats> {
    let learnings = load_learnings(&paths.learnings_path)?;

    if learnings.is_empty() {
        return Ok(ExportStats::default());
    }

    // Group by date
    let mut by_date: std::collections::BTreeMap<String, Vec<&Learning>> =
        std::collections::BTreeMap::new();

    for learning in &learnings {
        by_date
            .entry(learning.date.clone())
            .or_default()
            .push(learning);
    }

    let mut content = String::new();
    content.push_str("# Profundo Learnings\n\n");
    content.push_str("Extracted insights from conversation sessions.\n\n");

    let mut total_decisions = 0;
    let mut total_facts = 0;
    let mut total_actions = 0;

    // Write in reverse chronological order
    for (date, day_learnings) in by_date.iter().rev() {
        content.push_str(&format!("## {}\n\n", date));

        for learning in day_learnings {
            content.push_str(&format_learning_bullets(learning));
            content.push_str("\n\n");

            total_decisions += learning.decisions.len();
            total_facts += learning.facts_learned.len();
            total_actions += learning.action_items.len();
        }
    }

    fs::write(output_path, content).context("Failed to write export file")?;

    Ok(ExportStats {
        sessions: learnings.len(),
        decisions: total_decisions,
        facts: total_facts,
        actions: total_actions,
    })
}

/// Write a Profundo rollup section to a daily log file
pub fn write_rollup(paths: &Paths, date: NaiveDate) -> Result<RollupStats> {
    let learnings = load_learnings(&paths.learnings_path)?;
    let day_learnings = filter_by_date(&learnings, date);

    // Get stats for this specific date
    let stats_config = StatsConfig {
        since: Some(date),
        until: Some(date),
    };
    let day_stats = stats::collect(paths, stats_config)?;

    // Build the Profundo section
    let mut section = String::new();
    section.push_str("\n## Profundo\n\n");

    // Stats line
    if day_stats.session_count > 0 {
        section.push_str(&format!(
            "{} ({} sessions)\n\n",
            format_stats_summary(&day_stats.total),
            day_stats.session_count
        ));
    }

    // Learnings
    if day_learnings.is_empty() {
        section.push_str("_No learnings harvested for this date._\n");
    } else {
        for learning in &day_learnings {
            // Session reference
            section.push_str(&format!(
                "### Session `{}`\n\n",
                if learning.session_id.len() >= 8 { &learning.session_id[..8] } else { &learning.session_id }
            ));
            section.push_str(&format_learning_bullets(learning));
            section.push_str("\n\n");
        }
    }

    // Determine the daily log path
    let daily_log_path = paths.memory_dir.join(format!("{}.md", date));

    // Check if file exists and already has a Profundo section
    if daily_log_path.exists() {
        let existing = fs::read_to_string(&daily_log_path)?;
        if existing.contains("## Profundo") {
            // Replace existing section
            let updated = replace_profundo_section(&existing, &section);
            fs::write(&daily_log_path, updated)?;
        } else {
            // Append to existing file
            let mut file = OpenOptions::new()
                .append(true)
                .open(&daily_log_path)?;
            file.write_all(section.as_bytes())?;
        }
    } else {
        // Create new file with header and Profundo section
        let weekday = weekday_name(date.weekday());
        let header = format!(
            "# {} {} ({})\n",
            month_name(date.month()),
            date.day(),
            weekday
        );
        let content = format!("{}{}", header, section);
        fs::write(&daily_log_path, content)?;
    }

    Ok(RollupStats {
        sessions: day_learnings.len(),
        stats_sessions: day_stats.session_count,
        path: daily_log_path,
    })
}

/// Replace an existing ## Profundo section with a new one
fn replace_profundo_section(content: &str, new_section: &str) -> String {
    let mut result = String::new();
    let mut in_profundo = false;
    let mut found_next_section = false;

    for line in content.lines() {
        if line.starts_with("## Profundo") {
            in_profundo = true;
            continue;
        }

        if in_profundo {
            // Check if we've hit another ## section
            if line.starts_with("## ") {
                in_profundo = false;
                found_next_section = true;
                // Insert new section before this line
                result.push_str(new_section);
                result.push_str(line);
                result.push('\n');
            }
            // Skip old Profundo content
            continue;
        }

        result.push_str(line);
        result.push('\n');
    }

    // If Profundo was at the end, append new section
    if in_profundo && !found_next_section {
        // Remove trailing newline before adding section
        result = result.trim_end().to_string();
        result.push_str(new_section);
    }

    result
}

fn weekday_name(wd: Weekday) -> &'static str {
    match wd {
        Weekday::Mon => "Monday",
        Weekday::Tue => "Tuesday",
        Weekday::Wed => "Wednesday",
        Weekday::Thu => "Thursday",
        Weekday::Fri => "Friday",
        Weekday::Sat => "Saturday",
        Weekday::Sun => "Sunday",
    }
}

fn month_name(month: u32) -> &'static str {
    match month {
        1 => "Jan",
        2 => "Feb",
        3 => "Mar",
        4 => "Apr",
        5 => "May",
        6 => "Jun",
        7 => "Jul",
        8 => "Aug",
        9 => "Sep",
        10 => "Oct",
        11 => "Nov",
        12 => "Dec",
        _ => "???",
    }
}

#[derive(Default)]
pub struct ExportStats {
    pub sessions: usize,
    pub decisions: usize,
    pub facts: usize,
    pub actions: usize,
}

pub struct RollupStats {
    pub sessions: usize,
    pub stats_sessions: usize,
    pub path: std::path::PathBuf,
}
