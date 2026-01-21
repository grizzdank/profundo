//! Stats - token usage and cost analytics
//!
//! Aggregates usage data across all sessions for reporting.

use anyhow::Result;
use chrono::{NaiveDate, Utc};
use colored::Colorize;
use std::collections::HashMap;
use walkdir::WalkDir;

use crate::session::{Session, TokenStats};
use crate::Paths;

/// Aggregated stats across all sessions
#[derive(Debug, Default)]
pub struct AggregatedStats {
    pub total: TokenStats,
    pub by_model: HashMap<String, TokenStats>,
    pub by_date: HashMap<NaiveDate, TokenStats>,
    pub session_count: usize,
    pub date_range: Option<(NaiveDate, NaiveDate)>,
}

/// Configuration for stats command
pub struct StatsConfig {
    /// Only include sessions since this date
    pub since: Option<NaiveDate>,
    /// Only include sessions until this date
    pub until: Option<NaiveDate>,
}

impl Default for StatsConfig {
    fn default() -> Self {
        Self {
            since: None,
            until: None,
        }
    }
}

/// Collect stats from all sessions
pub fn collect(paths: &Paths, config: StatsConfig) -> Result<AggregatedStats> {
    let mut stats = AggregatedStats::default();

    for entry in WalkDir::new(&paths.sessions_dir)
        .max_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();

        // Only .jsonl files
        if !path.extension().map(|e| e == "jsonl").unwrap_or(false) {
            continue;
        }

        // Skip deleted sessions
        if path.to_str().map(|s| s.contains(".deleted")).unwrap_or(false) {
            continue;
        }

        let session = match Session::from_file(path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Get session date
        let session_date = session
            .first_timestamp
            .map(|dt| dt.date_naive())
            .unwrap_or_else(|| Utc::now().date_naive());

        // Apply date filters
        if let Some(since) = config.since {
            if session_date < since {
                continue;
            }
        }
        if let Some(until) = config.until {
            if session_date > until {
                continue;
            }
        }

        // Update date range
        stats.date_range = Some(match stats.date_range {
            None => (session_date, session_date),
            Some((min, max)) => (min.min(session_date), max.max(session_date)),
        });

        // Aggregate totals from session
        add_stats(&mut stats.total, &session.token_stats);

        // Aggregate by date from session totals
        let date_stats = stats.by_date.entry(session_date).or_default();
        add_stats(date_stats, &session.token_stats);

        // Aggregate by model from individual messages (not session level)
        collect_per_model_stats(&session, &mut stats.by_model);

        stats.session_count += 1;
    }

    Ok(stats)
}

/// Collect per-model stats from individual messages in a session
fn collect_per_model_stats(session: &Session, by_model: &mut HashMap<String, TokenStats>) {
    for msg in &session.messages {
        if msg.msg_type != "message" {
            continue;
        }

        let Some(ref content) = msg.message else {
            continue;
        };

        // Only count messages with a model (assistant responses)
        let Some(ref model) = content.model else {
            continue;
        };

        let Some(ref usage) = content.usage else {
            continue;
        };

        let model_stats = by_model.entry(model.clone()).or_default();

        model_stats.input_tokens += usage.input.unwrap_or(0);
        model_stats.output_tokens += usage.output.unwrap_or(0);
        model_stats.cache_read_tokens += usage.cache_read.unwrap_or(0);
        model_stats.cache_write_tokens += usage.cache_write.unwrap_or(0);
        model_stats.total_tokens += usage.total_tokens.unwrap_or(0);
        model_stats.message_count += 1;

        if let Some(ref cost) = usage.cost {
            model_stats.input_cost += cost.input.unwrap_or(0.0);
            model_stats.output_cost += cost.output.unwrap_or(0.0);
            model_stats.cache_read_cost += cost.cache_read.unwrap_or(0.0);
            model_stats.cache_write_cost += cost.cache_write.unwrap_or(0.0);
            model_stats.total_cost += cost.total.unwrap_or(0.0);
        }
    }
}

fn add_stats(target: &mut TokenStats, source: &TokenStats) {
    target.input_tokens += source.input_tokens;
    target.output_tokens += source.output_tokens;
    target.cache_read_tokens += source.cache_read_tokens;
    target.cache_write_tokens += source.cache_write_tokens;
    target.total_tokens += source.total_tokens;
    target.input_cost += source.input_cost;
    target.output_cost += source.output_cost;
    target.cache_read_cost += source.cache_read_cost;
    target.cache_write_cost += source.cache_write_cost;
    target.total_cost += source.total_cost;
    target.message_count += source.message_count;
}

/// Display stats in a formatted report
pub fn display(stats: &AggregatedStats) {
    println!("\n{}", "Token Usage Statistics".bold());
    println!("{}", "═".repeat(50));

    // Date range
    if let Some((start, end)) = stats.date_range {
        println!(
            "Period: {} to {} ({} sessions)",
            start.to_string().cyan(),
            end.to_string().cyan(),
            stats.session_count.to_string().cyan()
        );
    }
    println!();

    // Overall totals
    println!("{}", "Overall Totals".bold());
    println!("  Input tokens:       {:>12}", format_tokens(stats.total.input_tokens));
    println!("  Output tokens:      {:>12}", format_tokens(stats.total.output_tokens));
    println!("  Cache read:         {:>12}", format_tokens(stats.total.cache_read_tokens));
    println!("  Cache write:        {:>12}", format_tokens(stats.total.cache_write_tokens));
    println!("  {} {:>12}", "Total tokens:".bold(), format_tokens(stats.total.total_tokens).bold());
    println!();

    // Cache efficiency
    let cache_rate = stats.total.cache_hit_rate() * 100.0;
    let cache_color = if cache_rate > 50.0 {
        "green"
    } else if cache_rate > 20.0 {
        "yellow"
    } else {
        "red"
    };
    println!(
        "  Cache hit rate:     {:>11}%",
        format!("{:.1}", cache_rate).color(cache_color)
    );
    println!();

    // Cost breakdown
    println!("{}", "Cost Breakdown".bold());
    println!("  Input cost:         {:>12}", format_cost(stats.total.input_cost));
    println!("  Output cost:        {:>12}", format_cost(stats.total.output_cost));
    println!("  Cache read cost:    {:>12}", format_cost(stats.total.cache_read_cost));
    println!("  Cache write cost:   {:>12}", format_cost(stats.total.cache_write_cost));
    println!("  {} {:>12}", "Total cost:".bold(), format_cost(stats.total.total_cost).bold());
    println!();

    // Cost savings from cache
    let potential_input_cost = (stats.total.input_tokens + stats.total.cache_read_tokens) as f64
        * (stats.total.input_cost / stats.total.input_tokens.max(1) as f64);
    let cache_savings = potential_input_cost - stats.total.input_cost - stats.total.cache_read_cost;
    if cache_savings > 0.0 {
        println!(
            "  Cache savings:      {:>12}",
            format!("${:.4}", cache_savings).green()
        );
        println!();
    }

    // By model
    if !stats.by_model.is_empty() {
        println!("{}", "Usage by Model".bold());
        let mut models: Vec<_> = stats.by_model.iter().collect();
        models.sort_by(|a, b| b.1.total_cost.partial_cmp(&a.1.total_cost).unwrap());

        for (model, model_stats) in models {
            let pct = (model_stats.total_cost / stats.total.total_cost * 100.0) as u32;
            println!(
                "  {:<30} {:>10} ({:>2}%)",
                model.cyan(),
                format_cost(model_stats.total_cost),
                pct
            );
        }
        println!();
    }

    // Recent daily trend (last 7 days)
    if stats.by_date.len() > 1 {
        println!("{}", "Recent Daily Cost".bold());
        let mut dates: Vec<_> = stats.by_date.iter().collect();
        dates.sort_by_key(|(d, _)| *d);

        // Show last 7 days
        let recent: Vec<_> = dates.into_iter().rev().take(7).collect();
        for (date, day_stats) in recent.into_iter().rev() {
            let bar_len = ((day_stats.total_cost / stats.total.total_cost) * 30.0) as usize;
            let bar = "█".repeat(bar_len.max(1));
            println!(
                "  {} {} {}",
                date.to_string().dimmed(),
                format_cost(day_stats.total_cost),
                bar.cyan()
            );
        }
    }
}

fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        format!("{:.2}M", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}K", tokens as f64 / 1_000.0)
    } else {
        tokens.to_string()
    }
}

fn format_cost(cost: f64) -> String {
    format!("${:.4}", cost)
}
