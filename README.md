# Profundo

Memory system for Pulpito - semantic search and learning extraction from Clawdbot session logs.

Named for "the deep" (Spanish: *profundo*) - where memories sink and are retrieved from.

## Features

- **Embed**: Index session logs for semantic search using OpenRouter embeddings
- **Recall**: Search past conversations by meaning, not just keywords
- **Harvest**: Extract learnings (topics, decisions, facts, action items) using AI
- **Learnings**: Browse and search extracted insights
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

2. **Clawdbot config** (automatic for Clawdbot users):
   ```
   ~/.clawdbot/clawdbot.json → models.providers.openrouter.apiKey
   ```

If you're using Clawdbot with OpenRouter configured, no additional setup is needed.

## Usage

```bash
# Index session logs for semantic search
profundo embed

# Force reprocess all sessions
profundo embed --full

# Search memory
profundo recall "what did we discuss about oauth"

# Search with more results
profundo recall "token efficiency" -n 10

# Extract learnings from recent sessions
profundo harvest

# Harvest only sessions since a date
profundo harvest --since 2026-01-15

# View recent learnings
profundo learnings

# Search learnings
profundo learnings "decisions about infrastructure"

# Show status
profundo status

# Token usage statistics
profundo stats

# Stats for a specific date range
profundo stats --since 2026-01-01 --until 2026-01-15
```

## Directory Structure

```
~/pulpito/memory/
├── profundo.sqlite    # Embeddings database
├── learnings.jsonl    # Extracted insights
└── .profundo-cursor   # Processing state
```

## How It Works

### Embedding Pipeline
1. Reads Clawdbot session logs from `~/.clawdbot/agents/main/sessions/`
2. Chunks conversations by turns (user + assistant pairs)
3. Generates embeddings via OpenRouter (text-embedding-3-small)
4. Stores in SQLite for fast similarity search

### Recall Search
1. Embeds your query
2. Computes cosine similarity against all stored chunks
3. Returns top-k most similar conversation segments

### Harvest Pipeline
1. Reads session transcripts
2. Uses AI (DeepSeek V3.2 by default) to extract structured learnings
3. Appends to `learnings.jsonl`

## Cost

- Embeddings: ~$0.02 per 1M tokens (very cheap)
- Harvesting: ~$0.005 per session using DeepSeek V3.2

## Cron Setup

```bash
# Add to crontab for automatic processing
# Embed new sessions hourly
0 * * * * OPENROUTER_API_KEY=... /path/to/profundo embed

# Harvest learnings at 5:30am before morning review
30 5 * * * OPENROUTER_API_KEY=... /path/to/profundo harvest --since $(date -d yesterday +%Y-%m-%d)
```

## Integration with Pulpito

Add to `~/pulpito/TOOLS.md`:
```markdown
## Memory Tools

- `profundo recall "query"` - Semantic search of past conversations
- `profundo learnings "query"` - Search extracted insights
- `profundo status` - Show memory system status
- `profundo stats` - Token usage and cost analytics
```
