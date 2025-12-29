//! Data models for the database layer

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Holding represents a portfolio position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Holding {
    pub id: String,
    pub ticker: String,
    pub name: Option<String>,
    pub quantity: f64,
    pub cost_basis: f64,
    pub acquisition_date: DateTime<Utc>,
    pub notes: Option<String>,
    pub asset_class: AssetClass,
    pub sector: Option<String>,
    pub exchange: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Asset class enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetClass {
    Stock,
    Etf,
    MutualFund,
    Bond,
    Crypto,
    Cash,
    Other,
}

impl Default for AssetClass {
    fn default() -> Self {
        AssetClass::Stock
    }
}

impl AssetClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            AssetClass::Stock => "stock",
            AssetClass::Etf => "etf",
            AssetClass::MutualFund => "mutual_fund",
            AssetClass::Bond => "bond",
            AssetClass::Crypto => "crypto",
            AssetClass::Cash => "cash",
            AssetClass::Other => "other",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "stock" => AssetClass::Stock,
            "etf" => AssetClass::Etf,
            "mutual_fund" | "mutualfund" => AssetClass::MutualFund,
            "bond" => AssetClass::Bond,
            "crypto" | "cryptocurrency" => AssetClass::Crypto,
            "cash" => AssetClass::Cash,
            _ => AssetClass::Other,
        }
    }
}

/// Watchlist item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchlistItem {
    pub id: String,
    pub ticker: String,
    pub name: Option<String>,
    pub notes: Option<String>,
    pub added_at: DateTime<Utc>,
}

/// Conversation with AI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub conversation_id: String,
    pub role: MessageRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: Option<MessageMetadata>,
}

/// Message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

impl MessageRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::System => "system",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "system" => MessageRole::System,
            _ => MessageRole::User,
        }
    }
}

/// Metadata for assistant messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageMetadata {
    pub prompt_tokens: Option<i32>,
    pub completion_tokens: Option<i32>,
    pub latency_ms: Option<f64>,
    pub model: Option<String>,
}

/// Bitcoin wallet address
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletAddress {
    pub id: String,
    pub address: String,
    pub label: Option<String>,
    pub derivation_path: String,
    pub index: i32,
    pub address_type: AddressType,
    pub created_at: DateTime<Utc>,
}

/// Address type (receive or change)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AddressType {
    Receive,
    Change,
}

impl AddressType {
    pub fn as_str(&self) -> &'static str {
        match self {
            AddressType::Receive => "receive",
            AddressType::Change => "change",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "change" => AddressType::Change,
            _ => AddressType::Receive,
        }
    }
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub action: String,
    pub entity_type: String,
    pub entity_id: Option<String>,
    pub details: Option<String>,
    pub hmac: String,
}

/// Configuration key-value pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigEntry {
    pub key: String,
    pub value: String,
    pub updated_at: DateTime<Utc>,
}

/// Request to create a holding
#[derive(Debug, Clone)]
pub struct CreateHolding {
    pub ticker: String,
    pub quantity: f64,
    pub cost_basis: f64,
    pub acquisition_date: Option<DateTime<Utc>>,
    pub notes: Option<String>,
    pub asset_class: Option<AssetClass>,
}

/// Request to update a holding
#[derive(Debug, Clone)]
pub struct UpdateHolding {
    pub quantity: Option<f64>,
    pub cost_basis: Option<f64>,
    pub acquisition_date: Option<DateTime<Utc>>,
    pub notes: Option<String>,
}

/// Request to create a message
#[derive(Debug, Clone)]
pub struct CreateMessage {
    pub conversation_id: String,
    pub role: MessageRole,
    pub content: String,
    pub metadata: Option<MessageMetadata>,
}
