//! Profundo CLI
//!
//! Memory system for Pulpito - semantic search and learning extraction.

use anyhow::Result;
use chrono::NaiveDate;
use clap::{Parser, Subcommand};
use colored::Colorize;
use std::path::PathBuf;

use profundo::db::Database;
use profundo::Paths;

#[derive(Parser)]
#[command(name = "profundo")]
#[command(about = "Memory system for Pulpito - semantic search and learning extraction")]
#[command(version)]
struct Cli {
    /// Custom sessions directory
    #[arg(long, global = true)]
    sessions_dir: Option<PathBuf>,

    /// Custom memory directory
    #[arg(long, global = true)]
    memory_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Embed session logs for semantic search
    Embed {
        /// Reprocess all sessions, even if already embedded
        #[arg(long)]
        full: bool,

        /// Number of conversation turns per chunk
        #[arg(long, default_value = "3")]
        chunk_size: usize,

        /// Overlap between chunks
        #[arg(long, default_value = "1")]
        overlap: usize,
    },

    /// Search memory for similar content
    Recall {
        /// Search query
        query: String,

        /// Number of results to return
        #[arg(short = 'n', long, default_value = "5")]
        top_k: usize,

        /// Minimum similarity threshold (0.0 - 1.0)
        #[arg(short, long, default_value = "0.3")]
        threshold: f32,
    },

    /// Extract learnings from sessions
    Harvest {
        /// Only process sessions since this date (YYYY-MM-DD)
        #[arg(long)]
        since: Option<String>,

        /// Model to use for extraction
        #[arg(long, default_value = "deepseek/deepseek-v3.2")]
        model: String,

        /// Minimum messages to process a session
        #[arg(long, default_value = "4")]
        min_messages: usize,
    },

    /// Show memory status
    Status,

    /// Show token usage statistics
    Stats {
        /// Only include sessions since this date (YYYY-MM-DD)
        #[arg(long)]
        since: Option<String>,

        /// Only include sessions until this date (YYYY-MM-DD)
        #[arg(long)]
        until: Option<String>,
    },

    /// Search extracted learnings
    Learnings {
        /// Search query (searches topics, summaries, facts)
        query: Option<String>,

        /// Show last N entries
        #[arg(short = 'n', long, default_value = "10")]
        last: usize,
    },

    /// Export learnings to markdown for Clawdbot indexing
    Export {
        /// Output file path (default: memory/learnings.md)
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },

    /// Write daily rollup to memory log (learnings + stats)
    Rollup {
        /// Date to rollup (YYYY-MM-DD, default: yesterday)
        #[arg(long)]
        date: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up paths
    let paths = match (&cli.sessions_dir, &cli.memory_dir) {
        (Some(s), Some(m)) => Paths::with_bases(s.clone(), m.clone()),
        _ => Paths::default(),
    };

    match cli.command {
        Commands::Embed {
            full,
            chunk_size,
            overlap,
        } => {
            println!(
                "\n{} Profundo Embed\n",
                "🌊".to_string()
            );

            let config = profundo::embed::EmbedConfig {
                chunk_size,
                overlap,
                force_reprocess: full,
            };

            profundo::embed::run(&paths, config).await?;
        }

        Commands::Recall {
            query,
            top_k,
            threshold,
        } => {
            let config = profundo::recall::RecallConfig { top_k, threshold };
            let results = profundo::recall::search(&paths, &query, config).await?;
            profundo::recall::display_results(&results, &query);
        }

        Commands::Harvest {
            since,
            model,
            min_messages,
        } => {
            println!(
                "\n{} Profundo Harvest\n",
                "🌊".to_string()
            );

            let since_date = since
                .map(|s| NaiveDate::parse_from_str(&s, "%Y-%m-%d"))
                .transpose()
                .map_err(|e| anyhow::anyhow!("Invalid date format: {}", e))?;

            let config = profundo::harvest::HarvestConfig {
                since: since_date,
                model,
                min_messages,
            };

            profundo::harvest::run(&paths, config).await?;
        }

        Commands::Status => {
            println!(
                "\n{} Profundo Status\n",
                "🌊".to_string()
            );

            show_status(&paths)?;
        }

        Commands::Stats { since, until } => {
            let since_date = since
                .map(|s| NaiveDate::parse_from_str(&s, "%Y-%m-%d"))
                .transpose()
                .map_err(|e| anyhow::anyhow!("Invalid since date: {}", e))?;

            let until_date = until
                .map(|s| NaiveDate::parse_from_str(&s, "%Y-%m-%d"))
                .transpose()
                .map_err(|e| anyhow::anyhow!("Invalid until date: {}", e))?;

            let config = profundo::stats::StatsConfig {
                since: since_date,
                until: until_date,
            };

            let stats = profundo::stats::collect(&paths, config)?;
            profundo::stats::display(&stats);
        }

        Commands::Learnings { query, last } => {
            show_learnings(&paths, query.as_deref(), last)?;
        }

        Commands::Export { output } => {
            println!(
                "\n{} Profundo Export\n",
                "🌊".to_string()
            );

            let output_path = output.unwrap_or_else(|| paths.memory_dir.join("learnings.md"));

            let stats = profundo::export::export_to_markdown(&paths, &output_path)?;

            if stats.sessions == 0 {
                println!(
                    "{} No learnings to export. Run {} first.",
                    "→".yellow(),
                    "profundo harvest".cyan()
                );
            } else {
                println!(
                    "{} Exported {} sessions to {}",
                    "✓".green(),
                    stats.sessions.to_string().cyan(),
                    output_path.display().to_string().dimmed()
                );
                println!(
                    "  {} decisions, {} facts, {} action items",
                    stats.decisions.to_string().cyan(),
                    stats.facts.to_string().cyan(),
                    stats.actions.to_string().cyan()
                );
            }
        }

        Commands::Rollup { date } => {
            println!(
                "\n{} Profundo Rollup\n",
                "🌊".to_string()
            );

            // Default to yesterday (for morning review of previous day)
            let target_date = match date {
                Some(d) => NaiveDate::parse_from_str(&d, "%Y-%m-%d")
                    .map_err(|e| anyhow::anyhow!("Invalid date format: {}", e))?,
                None => chrono::Utc::now().date_naive() - chrono::Duration::days(1),
            };

            let stats = profundo::export::write_rollup(&paths, target_date)?;

            println!(
                "{} Wrote rollup for {} to {}",
                "✓".green(),
                target_date.to_string().cyan(),
                stats.path.display().to_string().dimmed()
            );
            println!(
                "  {} learnings from {} sessions",
                stats.sessions.to_string().cyan(),
                stats.stats_sessions.to_string().cyan()
            );
        }
    }

    Ok(())
}

fn show_status(paths: &Paths) -> Result<()> {
    // Database stats
    if paths.db_path.exists() {
        let db = Database::open(&paths.db_path)?;
        let stats = db.stats()?;

        println!("{}", "Embeddings Database".bold());
        println!(
            "  {} chunks from {} sessions",
            stats.chunks_count.to_string().cyan(),
            stats.sessions_count.to_string().cyan()
        );
        if let Some(last) = stats.last_processed {
            println!("  Last processed: {}", last.dimmed());
        }
        println!("  Path: {}", paths.db_path.display().to_string().dimmed());
    } else {
        println!(
            "{} No embeddings database yet. Run {} to create.",
            "→".yellow(),
            "profundo embed".cyan()
        );
    }

    println!();

    // Learnings stats
    if paths.learnings_path.exists() {
        let count = std::fs::read_to_string(&paths.learnings_path)?
            .lines()
            .count();

        println!("{}", "Learnings".bold());
        println!("  {} entries", count.to_string().cyan());
        println!(
            "  Path: {}",
            paths.learnings_path.display().to_string().dimmed()
        );
    } else {
        println!(
            "{} No learnings yet. Run {} to create.",
            "→".yellow(),
            "profundo harvest".cyan()
        );
    }

    println!();

    // Sessions directory
    if paths.sessions_dir.exists() {
        let session_count = std::fs::read_dir(&paths.sessions_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "jsonl")
                    .unwrap_or(false)
            })
            .count();

        let total_size: u64 = std::fs::read_dir(&paths.sessions_dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();

        println!("{}", "Session Logs".bold());
        println!(
            "  {} sessions ({:.1} MB)",
            session_count.to_string().cyan(),
            total_size as f64 / 1_000_000.0
        );
        println!(
            "  Path: {}",
            paths.sessions_dir.display().to_string().dimmed()
        );
    } else {
        println!(
            "{} Sessions directory not found: {}",
            "✗".red(),
            paths.sessions_dir.display()
        );
    }

    Ok(())
}

fn show_learnings(paths: &Paths, query: Option<&str>, last: usize) -> Result<()> {
    use profundo::harvest::Learning;
    use std::io::BufRead;

    if !paths.learnings_path.exists() {
        println!(
            "{} No learnings yet. Run {} to create.",
            "→".yellow(),
            "profundo harvest".cyan()
        );
        return Ok(());
    }

    let file = std::fs::File::open(&paths.learnings_path)?;
    let reader = std::io::BufReader::new(file);

    let mut learnings: Vec<Learning> = reader
        .lines()
        .filter_map(|l| l.ok())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();

    // Filter by query if provided
    if let Some(q) = query {
        let q_lower = q.to_lowercase();
        learnings.retain(|l| {
            l.topics.iter().any(|t| t.to_lowercase().contains(&q_lower))
                || l.summary.to_lowercase().contains(&q_lower)
                || l.facts_learned
                    .iter()
                    .any(|f| f.to_lowercase().contains(&q_lower))
                || l.decisions
                    .iter()
                    .any(|d| d.to_lowercase().contains(&q_lower))
        });
    }

    // Take last N
    let start = learnings.len().saturating_sub(last);
    let learnings = &learnings[start..];

    if learnings.is_empty() {
        println!(
            "{} No learnings found{}",
            "→".yellow(),
            query.map(|q| format!(" matching '{}'", q)).unwrap_or_default()
        );
        return Ok(());
    }

    println!(
        "\n{} {} learnings{}\n",
        "→".blue(),
        learnings.len().to_string().cyan(),
        query.map(|q| format!(" matching '{}'", q)).unwrap_or_default()
    );

    for learning in learnings {
        println!(
            "{} {} [{}]",
            "●".cyan(),
            learning.date.bold(),
            &learning.session_id[..8].dimmed()
        );

        if !learning.topics.is_empty() {
            println!(
                "  Topics: {}",
                learning.topics.join(", ").italic()
            );
        }

        if !learning.decisions.is_empty() {
            println!("  Decisions:");
            for d in &learning.decisions {
                println!("    • {}", d);
            }
        }

        if !learning.facts_learned.is_empty() {
            println!("  Facts:");
            for f in &learning.facts_learned {
                println!("    • {}", f);
            }
        }

        if !learning.action_items.is_empty() {
            println!("  Actions:");
            for a in &learning.action_items {
                println!("    • {}", a);
            }
        }

        println!("  Summary: {}", learning.summary.dimmed());
        println!();
    }

    Ok(())
}
