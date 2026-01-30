# Profundo

Memory system for OpenClaw - semantic search and learning extraction from session logs.

Named for "the deep" (Spanish: *profundo*) - where memories sink and are retrieved from.

## Features

- **Embed**: Index session logs for semantic search using OpenRouter embeddings
- **Recall**: Search past conversations by meaning, not just keywords
- **Harvest**: Extract learnings (topics, decisions, facts, action items) using AI
- **Learnings**: Browse and search extracted insights
- **Export**: Write learnings to markdown for OpenClaw indexing
- **Rollup**: Daily summary appended to memory logs (learnings + stats)
- **Stats**: Token usage analytics with per-model breakdown, cache efficiency, and cost trends

## Installation

```bash
# Build from source
cargo build --release

# Copy binary to path
cp target/release/profundo ~/.local/bin/
```

## Configuration

Profundo needs an OpenRouter API key. It checks these locations in order:

1. **Environment variable** (for standalone use):
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-..."
   ```

2. **OpenClaw config** (automatic for OpenClaw users):
   ```
   ~/.openclaw/openclaw.json → models.providers.openrouter.apiKey
   ```

If you're using OpenClaw with OpenRouter configured, no additional setup is needed.

## Usage

```bash
# Index session logs for semantic search
profundo embed

# Force reprocess all sessions
profundo embed --full

# Search memory (includes conversation chunks + harvested learnings)
profundo recall "what did we discuss about oauth"

# Search with more results
profundo recall "token efficiency" -n 10

# Expand query with AI-generated related terms
profundo recall "auth flow" --expand

# Extract learnings from recent sessions
profundo harvest

# Harvest only sessions since a date
profundo harvest --since 2026-01-15

# View recent learnings
profundo learnings

# Search learnings
profundo learnings "decisions about infrastructure"

# Export learnings to markdown (for OpenClaw indexing)
profundo export

# Export to custom path
profundo export -o ~/openclaw/memory/insights.md

# Write daily rollup to memory log (defaults to yesterday)
profundo rollup

# Rollup for a specific date
profundo rollup --date 2026-01-20

# Show status
profundo status

# Token usage statistics
profundo stats

# Stats for a specific date range
profundo stats --since 2026-01-01 --until 2026-01-15
```

## Directory Structure

Profundo reads the workspace path from your OpenClaw config (`~/.openclaw/openclaw.json` → `agents.defaults.workspace`). If not configured, it defaults to `~/openclaw`.

```
~/<workspace>/memory/
├── profundo.sqlite    # Embeddings database
├── learnings.jsonl    # Extracted insights (internal)
├── learnings.md       # Exported markdown (OpenClaw can index)
├── YYYY-MM-DD.md      # Daily logs with ## Profundo sections
└── .profundo-cursor   # Processing state
```

## How It Works

### Embedding Pipeline
1. Reads OpenClaw session logs from `~/.openclaw/agents/main/sessions/`
2. Chunks conversations by turns (user + assistant pairs)
3. Generates embeddings via OpenRouter (text-embedding-3-small)
4. Stores in SQLite for fast similarity search

### Recall Search
1. Embeds your query (optionally expanded with `--expand` for better coverage)
2. Computes cosine similarity against all stored chunks
3. Searches harvested learnings for matching insights
4. Returns combined results: conversation segments + relevant learnings

### Harvest Pipeline
1. Reads session transcripts
2. Uses AI (DeepSeek V3.2 by default) to extract structured learnings
3. Appends to `learnings.jsonl`

### Stats & Cost Tracking
Token usage and costs are read directly from OpenClaw's session logs — not calculated with hardcoded rates. OpenClaw logs costs based on API pricing at the time of each request.

**OAuth users:** If you're using Anthropic OAuth (usage limits instead of per-token billing), the costs shown represent *equivalent API rates*, not actual charges. This is still useful for understanding relative usage and cost efficiency across models.

## Cost

- Embeddings: ~$0.02 per 1M tokens (very cheap)
- Harvesting: ~$0.005 per session using DeepSeek V3.2

## Cron Setup

```bash
# Add to crontab for automatic processing

# Embed new sessions hourly
0 * * * * OPENROUTER_API_KEY=... /path/to/profundo embed

# Harvest learnings at 5:00am
0 5 * * * OPENROUTER_API_KEY=... /path/to/profundo harvest --since $(date -d yesterday +%Y-%m-%d)

# Daily rollup at 5:15am (after harvest, before morning review)
15 5 * * * /path/to/profundo rollup
```

The rollup command writes a `## Profundo` section to yesterday's daily log with:
- Token usage stats (input/output, cache rate, cost)
- Harvested learnings (topics, decisions, facts, action items)

## Integration with OpenClaw

Add to your workspace's `TOOLS.md`:
```markdown
## Memory Tools

- `profundo recall "query"` - Semantic search of past conversations
- `profundo learnings "query"` - Search extracted insights
- `profundo export` - Export learnings to markdown (overwrites learnings.md)
- `profundo rollup` - Append daily summary to memory log
- `profundo status` - Show memory system status
- `profundo stats` - Token usage and cost analytics
```
