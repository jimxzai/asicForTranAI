//! Gazillioner Encrypted Database Layer
//!
//! This crate provides a secure, encrypted database layer using SQLCipher
//! for the Gazillioner financial intelligence platform.
//!
//! Features:
//! - AES-256 encryption via SQLCipher
//! - Key derivation from Device Root Key + PIN
//! - HMAC integrity for audit logs
//! - FFI interface for Go integration

pub mod db;
pub mod crypto;
pub mod models;
pub mod error;
pub mod ffi;
pub mod audit;

pub use db::Database;
pub use crypto::KeyDerivation;
pub use error::{Error, Result};
pub use models::*;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
