//! Database operations for Profundo
//!
//! SQLite storage for embeddings and processing state.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

use crate::session::TextChunk;

/// Embedded chunk stored in the database
#[derive(Debug, Clone)]
pub struct StoredChunk {
    /// SQLite rowid for joining against FTS results
    pub rowid: i64,
    pub id: String,
    pub session_id: String,
    pub turn_start: i32,
    pub turn_end: i32,
    pub timestamp: Option<String>,
    pub text: String,
    pub embedding: Vec<f32>,
}

/// Database handle for Profundo
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open or create the database
    pub fn open(path: &Path) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create database directory")?;
        }

        let conn = Connection::open(path)
            .context("Failed to open database")?;

        let db = Self { conn };
        db.init_schema()?;

        Ok(db)
    }

    /// Initialize database schema
    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_start INTEGER NOT NULL,
                turn_end INTEGER NOT NULL,
                timestamp TEXT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_session_id ON chunks(session_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_timestamp ON chunks(timestamp);

            -- Full-text search index for chunks.text (FTS5)
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content=chunks,
                content_rowid=rowid
            );

            -- Keep FTS index in sync
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
            END;

            -- Processed sessions bookkeeping
            CREATE TABLE IF NOT EXISTS sessions_processed (
                session_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_mtime INTEGER NOT NULL,
                chunks_count INTEGER NOT NULL,
                processed_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        ).context("Failed to initialize schema")?;

        // One-time rebuild of FTS index for existing rows
        let already_built: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM state WHERE key = 'chunks_fts_built'",
                [],
                |row| row.get(0),
            )
            .optional()?;

        if already_built.is_none() {
            self.conn
                .execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')", [])
                .context("Failed to rebuild FTS index")?;
            self.conn.execute(
                "INSERT OR REPLACE INTO state(key, value) VALUES('chunks_fts_built', '1')",
                [],
            )?;
        }

        Ok(())
    }

    /// Check if a session has been processed (and file hasn't changed)
    pub fn is_session_processed(&self, session_id: &str, file_size: u64, file_mtime: i64) -> Result<bool> {
        let result: Option<(i64, i64)> = self.conn
            .query_row(
                "SELECT file_size, file_mtime FROM sessions_processed WHERE session_id = ?",
                params![session_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .context("Failed to check session status")?;

        match result {
            Some((stored_size, stored_mtime)) => {
                Ok(stored_size == file_size as i64 && stored_mtime == file_mtime)
            }
            None => Ok(false),
        }
    }

    /// Store chunks for a session
    pub fn store_chunks(
        &mut self,
        session_id: &str,
        file_path: &str,
        file_size: u64,
        file_mtime: i64,
        chunks: &[(TextChunk, Vec<f32>)],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;

        // Delete existing chunks for this session
        tx.execute(
            "DELETE FROM chunks WHERE session_id = ?",
            params![session_id],
        )?;

        // Insert new chunks (scoped to drop stmt before commit)
        {
            let mut stmt = tx.prepare(
                "INSERT INTO chunks (id, session_id, turn_start, turn_end, timestamp, text, embedding)
                 VALUES (?, ?, ?, ?, ?, ?, ?)"
            )?;

            for (chunk, embedding) in chunks {
                let id = uuid::Uuid::new_v4().to_string();
                let embedding_bytes = embedding_to_bytes(embedding);

                stmt.execute(params![
                    id,
                    chunk.session_id,
                    chunk.turn_start as i32,
                    chunk.turn_end as i32,
                    chunk.timestamp,
                    chunk.text,
                    embedding_bytes,
                ])?;
            }
        }

        // Update processed status
        tx.execute(
            "INSERT OR REPLACE INTO sessions_processed (session_id, file_path, file_size, file_mtime, chunks_count)
             VALUES (?, ?, ?, ?, ?)",
            params![session_id, file_path, file_size as i64, file_mtime, chunks.len() as i32],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Load all chunks for similarity search
    pub fn load_all_chunks(&self) -> Result<Vec<StoredChunk>> {
        let mut stmt = self.conn.prepare(
            "SELECT rowid, id, session_id, turn_start, turn_end, timestamp, text, embedding FROM chunks"
        )?;

        let chunks = stmt
            .query_map([], |row| {
                let embedding_bytes: Vec<u8> = row.get(7)?;
                Ok(StoredChunk {
                    rowid: row.get(0)?,
                    id: row.get(1)?,
                    session_id: row.get(2)?,
                    turn_start: row.get(3)?,
                    turn_end: row.get(4)?,
                    timestamp: row.get(5)?,
                    text: row.get(6)?,
                    embedding: bytes_to_embedding(&embedding_bytes),
                })
            })?
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to load chunks")?;

        Ok(chunks)
    }

    /// BM25-ranked lexical search using FTS5.
    ///
    /// Returns (rowid, rank) pairs ordered by ascending rank (lower is better).
    /// Sanitizes query to prevent FTS5 syntax errors from special characters.
    pub fn bm25_search(&self, query: &str, limit: usize) -> Result<Vec<(i64, f32)>> {
        let safe_query = sanitize_fts_query(query);
        if safe_query.is_empty() {
            return Ok(Vec::new());
        }

        let mut stmt = self.conn.prepare(
            "SELECT rowid, bm25(chunks_fts) as rank \
             FROM chunks_fts \
             WHERE chunks_fts MATCH ? \
             ORDER BY rank \
             LIMIT ?"
        )?;

        let rows = stmt
            .query_map(params![safe_query, limit as i64], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, f32>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to run BM25 search")?;

        Ok(rows)
    }

    /// Load chunks by rowids (for pre-filtered semantic search).
    ///
    /// Only loads embeddings for the specified rowids instead of the entire table.
    pub fn load_chunks_by_rowids(&self, rowids: &[i64]) -> Result<Vec<StoredChunk>> {
        if rowids.is_empty() {
            return Ok(Vec::new());
        }

        // Build parameterized IN clause
        let placeholders: Vec<String> = rowids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT rowid, id, session_id, turn_start, turn_end, timestamp, text, embedding \
             FROM chunks WHERE rowid IN ({})",
            placeholders.join(",")
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::ToSql> = rowids
            .iter()
            .map(|r| r as &dyn rusqlite::ToSql)
            .collect();

        let chunks = stmt
            .query_map(params.as_slice(), |row| {
                let embedding_bytes: Vec<u8> = row.get(7)?;
                Ok(StoredChunk {
                    rowid: row.get(0)?,
                    id: row.get(1)?,
                    session_id: row.get(2)?,
                    turn_start: row.get(3)?,
                    turn_end: row.get(4)?,
                    timestamp: row.get(5)?,
                    text: row.get(6)?,
                    embedding: bytes_to_embedding(&embedding_bytes),
                })
            })?
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to load chunks by rowids")?;

        Ok(chunks)
    }

    /// Get database statistics
    pub fn stats(&self) -> Result<DbStats> {
        let chunks_count: i64 = self.conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;

        let sessions_count: i64 = self.conn
            .query_row("SELECT COUNT(*) FROM sessions_processed", [], |row| row.get(0))?;

        let last_processed: Option<String> = self.conn
            .query_row(
                "SELECT MAX(processed_at) FROM sessions_processed",
                [],
                |row| row.get(0),
            )
            .optional()?
            .flatten();

        Ok(DbStats {
            chunks_count: chunks_count as usize,
            sessions_count: sessions_count as usize,
            last_processed,
        })
    }
}

#[derive(Debug)]
pub struct DbStats {
    pub chunks_count: usize,
    pub sessions_count: usize,
    pub last_processed: Option<String>,
}

/// Sanitize a user query for FTS5 MATCH.
///
/// Wraps each whitespace-delimited token in double quotes to prevent
/// FTS5 syntax errors from special characters (unbalanced quotes,
/// boolean operators like AND/OR/NOT/NEAR, parentheses, etc.).
fn sanitize_fts_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|token| {
            // Strip any existing quotes to avoid nesting
            let clean = token.replace('"', "");
            if clean.is_empty() {
                return String::new();
            }
            format!("\"{}\"", clean)
        })
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Convert embedding vector to bytes for storage
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

/// Convert bytes back to embedding vector
fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
