//! Error types for the database layer

use thiserror::Error;

/// Result type alias for database operations
pub type Result<T> = std::result::Result<T, Error>;

/// Database error types
#[derive(Error, Debug)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Encryption error: {0}")]
    Encryption(String),

    #[error("Key derivation failed: {0}")]
    KeyDerivation(String),

    #[error("Authentication failed")]
    AuthenticationFailed,

    #[error("Invalid PIN format")]
    InvalidPinFormat,

    #[error("PIN lockout: {attempts} failed attempts, locked for {remaining_seconds} seconds")]
    PinLockout {
        attempts: u32,
        remaining_seconds: u64,
    },

    #[error("Database not initialized")]
    NotInitialized,

    #[error("Database already initialized")]
    AlreadyInitialized,

    #[error("Integrity check failed: {0}")]
    IntegrityCheckFailed(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Not found: {entity} with id {id}")]
    NotFound { entity: String, id: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}

impl Error {
    /// Create a not found error
    pub fn not_found(entity: impl Into<String>, id: impl Into<String>) -> Self {
        Error::NotFound {
            entity: entity.into(),
            id: id.into(),
        }
    }
}
