//! SQLCipher database operations

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

use crate::crypto::{self, KeyDerivation};
use crate::error::{Error, Result};
use crate::models::*;

/// Encrypted database wrapper
pub struct Database {
    conn: Connection,
    hmac_key: [u8; 32],
}

impl Database {
    /// Open or create an encrypted database
    pub fn open<P: AsRef<Path>>(path: P, key: &[u8; 32]) -> Result<Self> {
        let conn = Connection::open(path)?;

        // Set SQLCipher encryption key
        let key_hex = KeyDerivation::key_to_hex(key);
        conn.execute_batch(&format!("PRAGMA key = {};", key_hex))?;

        // Verify encryption is working
        conn.execute_batch("SELECT count(*) FROM sqlite_master;")?;

        // Set secure PRAGMA settings
        conn.execute_batch(
            "
            PRAGMA cipher_memory_security = ON;
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA temp_store = MEMORY;
            ",
        )?;

        // Derive HMAC key for audit logs
        let hmac_key = crypto::sha256(key);

        let db = Database { conn, hmac_key };
        db.initialize_schema()?;

        Ok(db)
    }

    /// Create an in-memory database (for testing)
    pub fn open_in_memory(key: &[u8; 32]) -> Result<Self> {
        let conn = Connection::open_in_memory()?;

        let key_hex = KeyDerivation::key_to_hex(key);
        conn.execute_batch(&format!("PRAGMA key = {};", key_hex))?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;

        let hmac_key = crypto::sha256(key);

        let db = Database { conn, hmac_key };
        db.initialize_schema()?;

        Ok(db)
    }

    /// Initialize database schema
    fn initialize_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            -- Holdings table
            CREATE TABLE IF NOT EXISTS holdings (
                id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                name TEXT,
                quantity REAL NOT NULL,
                cost_basis REAL NOT NULL,
                acquisition_date TEXT NOT NULL,
                notes TEXT,
                asset_class TEXT NOT NULL DEFAULT 'stock',
                sector TEXT,
                exchange TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_holdings_ticker ON holdings(ticker);

            -- Watchlist table
            CREATE TABLE IF NOT EXISTS watchlist (
                id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL UNIQUE,
                name TEXT,
                notes TEXT,
                added_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_watchlist_ticker ON watchlist(ticker);

            -- Conversations table
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- Messages table
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);

            -- Wallet addresses table
            CREATE TABLE IF NOT EXISTS wallet_addresses (
                id TEXT PRIMARY KEY,
                address TEXT NOT NULL UNIQUE,
                label TEXT,
                derivation_path TEXT NOT NULL,
                idx INTEGER NOT NULL,
                address_type TEXT NOT NULL DEFAULT 'receive',
                created_at TEXT NOT NULL
            );

            -- Audit log table
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT,
                details TEXT,
                hmac TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);

            -- Config table
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            ",
        )?;

        Ok(())
    }

    // ========================================================================
    // Holdings CRUD
    // ========================================================================

    /// Create a new holding
    pub fn create_holding(&self, req: CreateHolding) -> Result<Holding> {
        let id = crypto::generate_id();
        let now = Utc::now();
        let acquisition_date = req.acquisition_date.unwrap_or(now);
        let asset_class = req.asset_class.unwrap_or_default();

        self.conn.execute(
            "INSERT INTO holdings (id, ticker, quantity, cost_basis, acquisition_date,
             notes, asset_class, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                id,
                req.ticker.to_uppercase(),
                req.quantity,
                req.cost_basis,
                acquisition_date.to_rfc3339(),
                req.notes,
                asset_class.as_str(),
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;

        self.log_action("create", "holding", Some(&id), None)?;

        self.get_holding(&id)
    }

    /// Get a holding by ID
    pub fn get_holding(&self, id: &str) -> Result<Holding> {
        self.conn
            .query_row(
                "SELECT id, ticker, name, quantity, cost_basis, acquisition_date,
                 notes, asset_class, sector, exchange, created_at, updated_at
                 FROM holdings WHERE id = ?1",
                params![id],
                |row| {
                    Ok(Holding {
                        id: row.get(0)?,
                        ticker: row.get(1)?,
                        name: row.get(2)?,
                        quantity: row.get(3)?,
                        cost_basis: row.get(4)?,
                        acquisition_date: parse_datetime(row.get::<_, String>(5)?),
                        notes: row.get(6)?,
                        asset_class: AssetClass::from_str(&row.get::<_, String>(7)?),
                        sector: row.get(8)?,
                        exchange: row.get(9)?,
                        created_at: parse_datetime(row.get::<_, String>(10)?),
                        updated_at: parse_datetime(row.get::<_, String>(11)?),
                    })
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    Error::not_found("holding", id)
                }
                _ => e.into(),
            })
    }

    /// List all holdings
    pub fn list_holdings(&self) -> Result<Vec<Holding>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, ticker, name, quantity, cost_basis, acquisition_date,
             notes, asset_class, sector, exchange, created_at, updated_at
             FROM holdings ORDER BY ticker",
        )?;

        let holdings = stmt
            .query_map([], |row| {
                Ok(Holding {
                    id: row.get(0)?,
                    ticker: row.get(1)?,
                    name: row.get(2)?,
                    quantity: row.get(3)?,
                    cost_basis: row.get(4)?,
                    acquisition_date: parse_datetime(row.get::<_, String>(5)?),
                    notes: row.get(6)?,
                    asset_class: AssetClass::from_str(&row.get::<_, String>(7)?),
                    sector: row.get(8)?,
                    exchange: row.get(9)?,
                    created_at: parse_datetime(row.get::<_, String>(10)?),
                    updated_at: parse_datetime(row.get::<_, String>(11)?),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(holdings)
    }

    /// Update a holding
    pub fn update_holding(&self, id: &str, req: UpdateHolding) -> Result<Holding> {
        let now = Utc::now();

        // Get current holding
        let current = self.get_holding(id)?;

        let quantity = req.quantity.unwrap_or(current.quantity);
        let cost_basis = req.cost_basis.unwrap_or(current.cost_basis);
        let acquisition_date = req.acquisition_date.unwrap_or(current.acquisition_date);
        let notes = req.notes.or(current.notes);

        self.conn.execute(
            "UPDATE holdings SET quantity = ?1, cost_basis = ?2, acquisition_date = ?3,
             notes = ?4, updated_at = ?5 WHERE id = ?6",
            params![
                quantity,
                cost_basis,
                acquisition_date.to_rfc3339(),
                notes,
                now.to_rfc3339(),
                id,
            ],
        )?;

        self.log_action("update", "holding", Some(id), None)?;

        self.get_holding(id)
    }

    /// Delete a holding
    pub fn delete_holding(&self, id: &str) -> Result<()> {
        let rows = self.conn.execute("DELETE FROM holdings WHERE id = ?1", params![id])?;

        if rows == 0 {
            return Err(Error::not_found("holding", id));
        }

        self.log_action("delete", "holding", Some(id), None)?;

        Ok(())
    }

    // ========================================================================
    // Watchlist CRUD
    // ========================================================================

    /// Add to watchlist
    pub fn add_to_watchlist(&self, ticker: &str, notes: Option<&str>) -> Result<WatchlistItem> {
        let id = crypto::generate_id();
        let now = Utc::now();
        let ticker = ticker.to_uppercase();

        self.conn.execute(
            "INSERT INTO watchlist (id, ticker, notes, added_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, ticker, notes, now.to_rfc3339()],
        )?;

        self.log_action("add", "watchlist", Some(&id), Some(&ticker))?;

        self.get_watchlist_item(&ticker)
    }

    /// Get watchlist item by ticker
    pub fn get_watchlist_item(&self, ticker: &str) -> Result<WatchlistItem> {
        self.conn
            .query_row(
                "SELECT id, ticker, name, notes, added_at FROM watchlist WHERE ticker = ?1",
                params![ticker.to_uppercase()],
                |row| {
                    Ok(WatchlistItem {
                        id: row.get(0)?,
                        ticker: row.get(1)?,
                        name: row.get(2)?,
                        notes: row.get(3)?,
                        added_at: parse_datetime(row.get::<_, String>(4)?),
                    })
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Error::not_found("watchlist", ticker),
                _ => e.into(),
            })
    }

    /// List watchlist
    pub fn list_watchlist(&self) -> Result<Vec<WatchlistItem>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, ticker, name, notes, added_at FROM watchlist ORDER BY ticker",
        )?;

        let items = stmt
            .query_map([], |row| {
                Ok(WatchlistItem {
                    id: row.get(0)?,
                    ticker: row.get(1)?,
                    name: row.get(2)?,
                    notes: row.get(3)?,
                    added_at: parse_datetime(row.get::<_, String>(4)?),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(items)
    }

    /// Remove from watchlist
    pub fn remove_from_watchlist(&self, ticker: &str) -> Result<()> {
        let ticker = ticker.to_uppercase();
        let rows = self
            .conn
            .execute("DELETE FROM watchlist WHERE ticker = ?1", params![ticker])?;

        if rows == 0 {
            return Err(Error::not_found("watchlist", &ticker));
        }

        self.log_action("remove", "watchlist", None, Some(&ticker))?;

        Ok(())
    }

    // ========================================================================
    // Conversations
    // ========================================================================

    /// Create a new conversation
    pub fn create_conversation(&self, title: Option<&str>) -> Result<Conversation> {
        let id = crypto::generate_id();
        let now = Utc::now();

        self.conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, title, now.to_rfc3339(), now.to_rfc3339()],
        )?;

        self.get_conversation(&id)
    }

    /// Get a conversation
    pub fn get_conversation(&self, id: &str) -> Result<Conversation> {
        self.conn
            .query_row(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?1",
                params![id],
                |row| {
                    Ok(Conversation {
                        id: row.get(0)?,
                        title: row.get(1)?,
                        created_at: parse_datetime(row.get::<_, String>(2)?),
                        updated_at: parse_datetime(row.get::<_, String>(3)?),
                    })
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Error::not_found("conversation", id),
                _ => e.into(),
            })
    }

    /// List conversations
    pub fn list_conversations(&self, limit: i32) -> Result<Vec<Conversation>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, created_at, updated_at FROM conversations
             ORDER BY updated_at DESC LIMIT ?1",
        )?;

        let convos = stmt
            .query_map(params![limit], |row| {
                Ok(Conversation {
                    id: row.get(0)?,
                    title: row.get(1)?,
                    created_at: parse_datetime(row.get::<_, String>(2)?),
                    updated_at: parse_datetime(row.get::<_, String>(3)?),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(convos)
    }

    /// Add message to conversation
    pub fn add_message(&self, req: CreateMessage) -> Result<Message> {
        let id = crypto::generate_id();
        let now = Utc::now();

        let metadata_json = req
            .metadata
            .as_ref()
            .map(|m| serde_json::to_string(m))
            .transpose()?;

        self.conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, timestamp, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                id,
                req.conversation_id,
                req.role.as_str(),
                req.content,
                now.to_rfc3339(),
                metadata_json,
            ],
        )?;

        // Update conversation timestamp
        self.conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![now.to_rfc3339(), req.conversation_id],
        )?;

        self.get_message(&id)
    }

    /// Get a message
    pub fn get_message(&self, id: &str) -> Result<Message> {
        self.conn
            .query_row(
                "SELECT id, conversation_id, role, content, timestamp, metadata
                 FROM messages WHERE id = ?1",
                params![id],
                |row| {
                    let metadata_str: Option<String> = row.get(5)?;
                    let metadata = metadata_str
                        .map(|s| serde_json::from_str(&s))
                        .transpose()
                        .unwrap_or(None);

                    Ok(Message {
                        id: row.get(0)?,
                        conversation_id: row.get(1)?,
                        role: MessageRole::from_str(&row.get::<_, String>(2)?),
                        content: row.get(3)?,
                        timestamp: parse_datetime(row.get::<_, String>(4)?),
                        metadata,
                    })
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Error::not_found("message", id),
                _ => e.into(),
            })
    }

    /// Get messages for a conversation
    pub fn get_messages(&self, conversation_id: &str) -> Result<Vec<Message>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, conversation_id, role, content, timestamp, metadata
             FROM messages WHERE conversation_id = ?1 ORDER BY timestamp",
        )?;

        let messages = stmt
            .query_map(params![conversation_id], |row| {
                let metadata_str: Option<String> = row.get(5)?;
                let metadata = metadata_str
                    .map(|s| serde_json::from_str(&s))
                    .transpose()
                    .unwrap_or(None);

                Ok(Message {
                    id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    role: MessageRole::from_str(&row.get::<_, String>(2)?),
                    content: row.get(3)?,
                    timestamp: parse_datetime(row.get::<_, String>(4)?),
                    metadata,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(messages)
    }

    // ========================================================================
    // Config
    // ========================================================================

    /// Get config value
    pub fn get_config(&self, key: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT value FROM config WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    /// Set config value
    pub fn set_config(&self, key: &str, value: &str) -> Result<()> {
        let now = Utc::now();

        self.conn.execute(
            "INSERT INTO config (key, value, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT(key) DO UPDATE SET value = ?2, updated_at = ?3",
            params![key, value, now.to_rfc3339()],
        )?;

        Ok(())
    }

    // ========================================================================
    // Audit Log
    // ========================================================================

    /// Log an action with HMAC integrity
    fn log_action(
        &self,
        action: &str,
        entity_type: &str,
        entity_id: Option<&str>,
        details: Option<&str>,
    ) -> Result<()> {
        let id = crypto::generate_id();
        let now = Utc::now();

        // Create data for HMAC
        let data = format!(
            "{}:{}:{}:{}:{}",
            id,
            now.to_rfc3339(),
            action,
            entity_type,
            entity_id.unwrap_or("")
        );
        let hmac = crypto::hmac::generate(&self.hmac_key, data.as_bytes());
        let hmac_hex = hex::encode(hmac);

        self.conn.execute(
            "INSERT INTO audit_log (id, timestamp, action, entity_type, entity_id, details, hmac)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                id,
                now.to_rfc3339(),
                action,
                entity_type,
                entity_id,
                details,
                hmac_hex,
            ],
        )?;

        Ok(())
    }

    /// Verify audit log integrity
    pub fn verify_audit_log(&self) -> Result<bool> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, action, entity_type, entity_id, hmac FROM audit_log",
        )?;

        let entries = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, String>(5)?,
            ))
        })?;

        for entry in entries {
            let (id, timestamp, action, entity_type, entity_id, hmac_hex) = entry?;

            let data = format!(
                "{}:{}:{}:{}:{}",
                id,
                timestamp,
                action,
                entity_type,
                entity_id.as_deref().unwrap_or("")
            );

            let expected_hmac = hex::decode(&hmac_hex)
                .map_err(|e| Error::IntegrityCheckFailed(format!("Invalid HMAC hex: {}", e)))?;

            let mut expected = [0u8; 32];
            if expected_hmac.len() != 32 {
                return Err(Error::IntegrityCheckFailed("Invalid HMAC length".into()));
            }
            expected.copy_from_slice(&expected_hmac);

            if !crypto::hmac::verify(&self.hmac_key, data.as_bytes(), &expected) {
                return Err(Error::IntegrityCheckFailed(format!(
                    "HMAC verification failed for entry {}",
                    id
                )));
            }
        }

        Ok(true)
    }
}

/// Parse RFC3339 datetime string
fn parse_datetime(s: String) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(&s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Database {
        let key = [0u8; 32];
        Database::open_in_memory(&key).unwrap()
    }

    #[test]
    fn test_create_and_get_holding() {
        let db = test_db();

        let holding = db
            .create_holding(CreateHolding {
                ticker: "AAPL".into(),
                quantity: 100.0,
                cost_basis: 150.0,
                acquisition_date: None,
                notes: Some("Test holding".into()),
                asset_class: Some(AssetClass::Stock),
            })
            .unwrap();

        assert_eq!(holding.ticker, "AAPL");
        assert_eq!(holding.quantity, 100.0);
        assert_eq!(holding.cost_basis, 150.0);

        let retrieved = db.get_holding(&holding.id).unwrap();
        assert_eq!(retrieved.ticker, holding.ticker);
    }

    #[test]
    fn test_list_holdings() {
        let db = test_db();

        db.create_holding(CreateHolding {
            ticker: "AAPL".into(),
            quantity: 100.0,
            cost_basis: 150.0,
            acquisition_date: None,
            notes: None,
            asset_class: None,
        })
        .unwrap();

        db.create_holding(CreateHolding {
            ticker: "MSFT".into(),
            quantity: 50.0,
            cost_basis: 300.0,
            acquisition_date: None,
            notes: None,
            asset_class: None,
        })
        .unwrap();

        let holdings = db.list_holdings().unwrap();
        assert_eq!(holdings.len(), 2);
    }

    #[test]
    fn test_watchlist() {
        let db = test_db();

        db.add_to_watchlist("NVDA", Some("Watch for earnings")).unwrap();
        db.add_to_watchlist("GOOGL", None).unwrap();

        let items = db.list_watchlist().unwrap();
        assert_eq!(items.len(), 2);

        db.remove_from_watchlist("NVDA").unwrap();
        let items = db.list_watchlist().unwrap();
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_conversations() {
        let db = test_db();

        let convo = db.create_conversation(Some("Test chat")).unwrap();
        assert_eq!(convo.title, Some("Test chat".into()));

        db.add_message(CreateMessage {
            conversation_id: convo.id.clone(),
            role: MessageRole::User,
            content: "Hello".into(),
            metadata: None,
        })
        .unwrap();

        db.add_message(CreateMessage {
            conversation_id: convo.id.clone(),
            role: MessageRole::Assistant,
            content: "Hi there!".into(),
            metadata: None,
        })
        .unwrap();

        let messages = db.get_messages(&convo.id).unwrap();
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_audit_log_integrity() {
        let db = test_db();

        db.create_holding(CreateHolding {
            ticker: "AAPL".into(),
            quantity: 100.0,
            cost_basis: 150.0,
            acquisition_date: None,
            notes: None,
            asset_class: None,
        })
        .unwrap();

        assert!(db.verify_audit_log().unwrap());
    }
}
