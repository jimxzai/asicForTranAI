//! Error types for the sync library

use thiserror::Error;

/// Result type alias for sync operations
pub type SyncResult<T> = Result<T, SyncError>;

/// Errors that can occur during sync operations
#[derive(Error, Debug)]
pub enum SyncError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Network error
    #[error("Network error: {0}")]
    Network(String),

    /// mDNS discovery error
    #[error("Discovery error: {0}")]
    Discovery(String),

    /// TLS/encryption error
    #[error("Encryption error: {0}")]
    Encryption(String),

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    Authentication(String),

    /// Device not paired
    #[error("Device not paired")]
    NotPaired,

    /// Pairing failed
    #[error("Pairing failed: {0}")]
    PairingFailed(String),

    /// Invalid verification code
    #[error("Invalid verification code")]
    InvalidVerificationCode,

    /// Pairing timeout
    #[error("Pairing timeout")]
    PairingTimeout,

    /// Protocol version mismatch
    #[error("Protocol version mismatch: local={local}, remote={remote}")]
    ProtocolMismatch { local: u32, remote: u32 },

    /// Sync conflict
    #[error("Sync conflict: {0}")]
    Conflict(String),

    /// Connection refused
    #[error("Connection refused by peer")]
    ConnectionRefused,

    /// Connection timeout
    #[error("Connection timeout")]
    ConnectionTimeout,

    /// Connection closed unexpectedly
    #[error("Connection closed unexpectedly")]
    ConnectionClosed,

    /// Invalid message format
    #[error("Invalid message format: {0}")]
    InvalidMessage(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Database error
    #[error("Database error: {0}")]
    Database(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Peer disconnected
    #[error("Peer disconnected")]
    PeerDisconnected,

    /// Operation cancelled
    #[error("Operation cancelled")]
    Cancelled,

    /// Rate limited
    #[error("Rate limited, try again later")]
    RateLimited,

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl SyncError {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            SyncError::Network(_)
                | SyncError::ConnectionTimeout
                | SyncError::ConnectionClosed
                | SyncError::PeerDisconnected
                | SyncError::RateLimited
        )
    }

    /// Check if this error is a connection error
    pub fn is_connection_error(&self) -> bool {
        matches!(
            self,
            SyncError::Network(_)
                | SyncError::ConnectionRefused
                | SyncError::ConnectionTimeout
                | SyncError::ConnectionClosed
                | SyncError::PeerDisconnected
        )
    }

    /// Check if this error is an authentication error
    pub fn is_auth_error(&self) -> bool {
        matches!(
            self,
            SyncError::Authentication(_)
                | SyncError::NotPaired
                | SyncError::InvalidVerificationCode
        )
    }
}

impl From<serde_json::Error> for SyncError {
    fn from(e: serde_json::Error) -> Self {
        SyncError::Serialization(e.to_string())
    }
}

impl From<bincode::Error> for SyncError {
    fn from(e: bincode::Error) -> Self {
        SyncError::Serialization(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_retryable() {
        assert!(SyncError::Network("test".into()).is_retryable());
        assert!(SyncError::ConnectionTimeout.is_retryable());
        assert!(!SyncError::NotPaired.is_retryable());
        assert!(!SyncError::InvalidVerificationCode.is_retryable());
    }

    #[test]
    fn test_error_connection() {
        assert!(SyncError::ConnectionRefused.is_connection_error());
        assert!(SyncError::PeerDisconnected.is_connection_error());
        assert!(!SyncError::NotPaired.is_connection_error());
    }

    #[test]
    fn test_error_auth() {
        assert!(SyncError::NotPaired.is_auth_error());
        assert!(SyncError::InvalidVerificationCode.is_auth_error());
        assert!(!SyncError::ConnectionTimeout.is_auth_error());
    }
}
