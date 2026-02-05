
## Subagent Session Tracking

**Problem:** Subagent sessions (spawned via `sessions_spawn`) get `.deleted` suffix after completion. Profundo skips deleted files, so subagent work isn't indexed for recall/learnings.

**Impact:** Losing valuable context from research tasks, comparisons, background work.

**Options:**
1. Index sessions BEFORE deletion (hook into cleanup process)
2. Include `.deleted` files in indexing (they still exist, just marked)
3. Change cleanup policy to delay deletion until after embed cron runs
4. Copy subagent sessions to a "completed" archive before deletion

**Recommendation:** Option 2 is simplest — just include `*.jsonl.deleted.*` in the glob pattern for embed/harvest. The content is still there.

**Added:** Jan 31, 2026
