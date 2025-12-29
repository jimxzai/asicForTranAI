//! Wallet error types

use thiserror::Error;

/// Result type for wallet operations
pub type Result<T> = std::result::Result<T, WalletError>;

/// Wallet error types
#[derive(Error, Debug)]
pub enum WalletError {
    #[error("Invalid mnemonic: {0}")]
    InvalidMnemonic(String),

    #[error("Invalid passphrase")]
    InvalidPassphrase,

    #[error("Invalid derivation path: {0}")]
    InvalidDerivationPath(String),

    #[error("Key derivation failed: {0}")]
    KeyDerivationFailed(String),

    #[error("Invalid private key")]
    InvalidPrivateKey,

    #[error("Invalid public key")]
    InvalidPublicKey,

    #[error("Invalid address: {0}")]
    InvalidAddress(String),

    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),

    #[error("Signing failed: {0}")]
    SigningFailed(String),

    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    #[error("Insufficient funds: need {needed}, have {available}")]
    InsufficientFunds { needed: u64, available: u64 },

    #[error("Invalid amount: {0}")]
    InvalidAmount(String),

    #[error("Network mismatch: expected {expected:?}, got {actual:?}")]
    NetworkMismatch { expected: String, actual: String },

    #[error("Invalid network: {0}")]
    InvalidNetwork(String),

    #[error("Wallet not initialized")]
    NotInitialized,

    #[error("Wallet already initialized")]
    AlreadyInitialized,

    #[error("Air-gap encoding failed: {0}")]
    AirGapEncodingFailed(String),

    #[error("Air-gap decoding failed: {0}")]
    AirGapDecodingFailed(String),

    #[error("QR code generation failed: {0}")]
    QrCodeFailed(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Crypto error: {0}")]
    Crypto(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
