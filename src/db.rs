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
            "SELECT id, session_id, turn_start, turn_end, timestamp, text, embedding FROM chunks"
        )?;

        let chunks = stmt
            .query_map([], |row| {
                let embedding_bytes: Vec<u8> = row.get(6)?;
                Ok(StoredChunk {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    turn_start: row.get(2)?,
                    turn_end: row.get(3)?,
                    timestamp: row.get(4)?,
                    text: row.get(5)?,
                    embedding: bytes_to_embedding(&embedding_bytes),
                })
            })?
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to load chunks")?;

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
