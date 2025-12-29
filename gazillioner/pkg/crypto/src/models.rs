//! Data models for the database layer

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ============================================================================
// Portfolio - Multi-portfolio support
// ============================================================================

/// Portfolio represents a collection of holdings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub portfolio_type: PortfolioType,
    pub currency: String,
    pub is_default: bool,
    pub color: Option<String>,           // For UI display
    pub icon: Option<String>,            // For UI display
    pub benchmark_ticker: Option<String>, // e.g., "SPY" for comparison
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Portfolio type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortfolioType {
    Investment,     // General investment portfolio
    Retirement,     // 401k, IRA, etc.
    Trading,        // Active trading account
    Crypto,         // Cryptocurrency holdings
    RealEstate,     // Real estate investments
    Education,      // 529 plans, education savings
    Emergency,      // Emergency fund
    Custom,         // User-defined
}

impl Default for PortfolioType {
    fn default() -> Self {
        PortfolioType::Investment
    }
}

impl PortfolioType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PortfolioType::Investment => "investment",
            PortfolioType::Retirement => "retirement",
            PortfolioType::Trading => "trading",
            PortfolioType::Crypto => "crypto",
            PortfolioType::RealEstate => "real_estate",
            PortfolioType::Education => "education",
            PortfolioType::Emergency => "emergency",
            PortfolioType::Custom => "custom",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "investment" => PortfolioType::Investment,
            "retirement" => PortfolioType::Retirement,
            "trading" => PortfolioType::Trading,
            "crypto" => PortfolioType::Crypto,
            "real_estate" | "realestate" => PortfolioType::RealEstate,
            "education" => PortfolioType::Education,
            "emergency" => PortfolioType::Emergency,
            _ => PortfolioType::Custom,
        }
    }
}

// ============================================================================
// Holding - Updated with portfolio_id
// ============================================================================

/// Holding represents a portfolio position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Holding {
    pub id: String,
    pub portfolio_id: String,             // Foreign key to Portfolio
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

/// Portfolio summary with computed values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSummary {
    pub portfolio_id: String,
    pub portfolio_name: String,
    pub holdings_count: i32,
    pub total_cost: f64,
}

/// Aggregated summary across all portfolios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedSummary {
    pub portfolios_count: i32,
    pub holdings_count: i32,
    pub total_cost: f64,
}

/// Request to create a portfolio
#[derive(Debug, Clone)]
pub struct CreatePortfolio {
    pub name: String,
    pub description: Option<String>,
    pub portfolio_type: Option<PortfolioType>,
    pub currency: Option<String>,
    pub is_default: Option<bool>,
    pub color: Option<String>,
    pub icon: Option<String>,
    pub benchmark_ticker: Option<String>,
}

/// Request to update a portfolio
#[derive(Debug, Clone)]
pub struct UpdatePortfolio {
    pub name: Option<String>,
    pub description: Option<String>,
    pub portfolio_type: Option<PortfolioType>,
    pub is_default: Option<bool>,
    pub color: Option<String>,
    pub icon: Option<String>,
    pub benchmark_ticker: Option<String>,
}

/// Request to create a holding
#[derive(Debug, Clone)]
pub struct CreateHolding {
    pub portfolio_id: String,
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

// ============================================================================
// Broker Connections - Encrypted credential storage
// ============================================================================

/// Broker type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrokerType {
    Alpaca,
    Ibkr,
    Schwab,
    Coinbase,
}

impl BrokerType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BrokerType::Alpaca => "alpaca",
            BrokerType::Ibkr => "ibkr",
            BrokerType::Schwab => "schwab",
            BrokerType::Coinbase => "coinbase",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "alpaca" => BrokerType::Alpaca,
            "ibkr" | "interactive_brokers" => BrokerType::Ibkr,
            "schwab" | "charles_schwab" => BrokerType::Schwab,
            "coinbase" => BrokerType::Coinbase,
            _ => BrokerType::Alpaca, // Default
        }
    }
}

/// Broker connection status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrokerStatus {
    Disconnected,
    Connecting,
    Connected,
    Error,
    TokenExpired,
}

impl BrokerStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            BrokerStatus::Disconnected => "disconnected",
            BrokerStatus::Connecting => "connecting",
            BrokerStatus::Connected => "connected",
            BrokerStatus::Error => "error",
            BrokerStatus::TokenExpired => "token_expired",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "connecting" => BrokerStatus::Connecting,
            "connected" => BrokerStatus::Connected,
            "error" => BrokerStatus::Error,
            "token_expired" => BrokerStatus::TokenExpired,
            _ => BrokerStatus::Disconnected,
        }
    }
}

/// Stored broker connection with encrypted credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokerConnection {
    pub id: String,
    pub broker_type: BrokerType,
    pub display_name: String,
    pub account_id: Option<String>,
    pub status: BrokerStatus,
    pub paper_trading: bool,
    pub error_message: Option<String>,
    pub connected_at: Option<DateTime<Utc>>,
    pub token_expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Encrypted OAuth tokens (stored separately, never exposed in JSON)
#[derive(Debug, Clone)]
pub struct BrokerCredentials {
    pub connection_id: String,
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
    pub token_type: Option<String>,
    pub scope: Option<String>,
}

/// Request to create a broker connection
#[derive(Debug, Clone)]
pub struct CreateBrokerConnection {
    pub broker_type: BrokerType,
    pub display_name: String,
    pub paper_trading: bool,
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
}

/// Request to update broker tokens
#[derive(Debug, Clone)]
pub struct UpdateBrokerTokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub token_type: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
    pub scope: Option<String>,
}

// ============================================================================
// P2P Sync - Device pairing and sync metadata
// ============================================================================

/// Paired device for P2P sync
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedDevice {
    pub device_id: String,
    pub device_name: String,
    pub first_paired_at: DateTime<Utc>,
    pub last_sync_at: Option<DateTime<Utc>>,
    pub sync_enabled: bool,
    pub trust_level: TrustLevel,
}

/// Trust level for paired devices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    Standard,
    Trusted,
    Blocked,
}

impl TrustLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            TrustLevel::Standard => "standard",
            TrustLevel::Trusted => "trusted",
            TrustLevel::Blocked => "blocked",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "trusted" => TrustLevel::Trusted,
            "blocked" => TrustLevel::Blocked,
            _ => TrustLevel::Standard,
        }
    }
}

/// Sync session record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncSession {
    pub id: String,
    pub device_id: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: SyncStatus,
    pub records_sent: i32,
    pub records_received: i32,
    pub bytes_transferred: i64,
    pub error_message: Option<String>,
}

/// Sync session status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncStatus {
    InProgress,
    Completed,
    PartiallyCompleted,
    Failed,
    Cancelled,
}

impl SyncStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            SyncStatus::InProgress => "in_progress",
            SyncStatus::Completed => "completed",
            SyncStatus::PartiallyCompleted => "partially_completed",
            SyncStatus::Failed => "failed",
            SyncStatus::Cancelled => "cancelled",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "completed" => SyncStatus::Completed,
            "partially_completed" => SyncStatus::PartiallyCompleted,
            "failed" => SyncStatus::Failed,
            "cancelled" => SyncStatus::Cancelled,
            _ => SyncStatus::InProgress,
        }
    }
}
