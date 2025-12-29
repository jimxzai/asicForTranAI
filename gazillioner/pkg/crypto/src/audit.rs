//! Audit logging with HMAC integrity verification

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::crypto;

/// Audit log entry for tracking all database operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique identifier
    pub id: String,
    /// Timestamp of the action
    pub timestamp: DateTime<Utc>,
    /// Type of action performed
    pub action: AuditAction,
    /// Type of entity affected
    pub entity_type: EntityType,
    /// ID of the entity (if applicable)
    pub entity_id: Option<String>,
    /// Additional details in JSON format
    pub details: Option<String>,
    /// HMAC for integrity verification
    pub hmac: String,
}

/// Types of auditable actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditAction {
    Create,
    Read,
    Update,
    Delete,
    Import,
    Export,
    Login,
    Logout,
    PinChange,
    ConfigChange,
    WalletInit,
    AddressGenerate,
}

impl AuditAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            AuditAction::Create => "create",
            AuditAction::Read => "read",
            AuditAction::Update => "update",
            AuditAction::Delete => "delete",
            AuditAction::Import => "import",
            AuditAction::Export => "export",
            AuditAction::Login => "login",
            AuditAction::Logout => "logout",
            AuditAction::PinChange => "pin_change",
            AuditAction::ConfigChange => "config_change",
            AuditAction::WalletInit => "wallet_init",
            AuditAction::AddressGenerate => "address_generate",
        }
    }
}

/// Types of entities that can be audited
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    Holding,
    Watchlist,
    Conversation,
    Message,
    WalletAddress,
    Config,
    Session,
}

impl EntityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntityType::Holding => "holding",
            EntityType::Watchlist => "watchlist",
            EntityType::Conversation => "conversation",
            EntityType::Message => "message",
            EntityType::WalletAddress => "wallet_address",
            EntityType::Config => "config",
            EntityType::Session => "session",
        }
    }
}

/// Audit log builder for creating entries
pub struct AuditBuilder {
    action: AuditAction,
    entity_type: EntityType,
    entity_id: Option<String>,
    details: Option<String>,
}

impl AuditBuilder {
    /// Create a new audit builder
    pub fn new(action: AuditAction, entity_type: EntityType) -> Self {
        Self {
            action,
            entity_type,
            entity_id: None,
            details: None,
        }
    }

    /// Set the entity ID
    pub fn entity_id(mut self, id: impl Into<String>) -> Self {
        self.entity_id = Some(id.into());
        self
    }

    /// Set additional details
    pub fn details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Build the audit entry with HMAC
    pub fn build(self, hmac_key: &[u8]) -> AuditEntry {
        let id = crypto::generate_id();
        let timestamp = Utc::now();

        // Create data for HMAC
        let data = format!(
            "{}:{}:{}:{}:{}",
            id,
            timestamp.to_rfc3339(),
            self.action.as_str(),
            self.entity_type.as_str(),
            self.entity_id.as_deref().unwrap_or("")
        );

        let hmac = crypto::hmac::generate(hmac_key, data.as_bytes());
        let hmac_hex = hex::encode(hmac);

        AuditEntry {
            id,
            timestamp,
            action: self.action,
            entity_type: self.entity_type,
            entity_id: self.entity_id,
            details: self.details,
            hmac: hmac_hex,
        }
    }
}

/// Verify the integrity of an audit entry
pub fn verify_entry(entry: &AuditEntry, hmac_key: &[u8]) -> bool {
    let data = format!(
        "{}:{}:{}:{}:{}",
        entry.id,
        entry.timestamp.to_rfc3339(),
        entry.action.as_str(),
        entry.entity_type.as_str(),
        entry.entity_id.as_deref().unwrap_or("")
    );

    let expected_hmac = match hex::decode(&entry.hmac) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => return false,
    };

    crypto::hmac::verify(hmac_key, data.as_bytes(), &expected_hmac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_entry_creation() {
        let key = b"test-hmac-key-for-audit-testing!";

        let entry = AuditBuilder::new(AuditAction::Create, EntityType::Holding)
            .entity_id("holding-123")
            .details("Created AAPL holding")
            .build(key);

        assert_eq!(entry.action, AuditAction::Create);
        assert_eq!(entry.entity_type, EntityType::Holding);
        assert_eq!(entry.entity_id, Some("holding-123".to_string()));
        assert!(!entry.hmac.is_empty());
    }

    #[test]
    fn test_audit_entry_verification() {
        let key = b"test-hmac-key-for-audit-testing!";

        let entry = AuditBuilder::new(AuditAction::Delete, EntityType::Watchlist)
            .entity_id("watchlist-456")
            .build(key);

        assert!(verify_entry(&entry, key));

        // Wrong key should fail
        let wrong_key = b"wrong-key-should-fail-verify!!!";
        assert!(!verify_entry(&entry, wrong_key));
    }

    #[test]
    fn test_tampered_entry_fails_verification() {
        let key = b"test-hmac-key-for-audit-testing!";

        let mut entry = AuditBuilder::new(AuditAction::Update, EntityType::Config)
            .entity_id("config-key")
            .build(key);

        // Tamper with the entry
        entry.entity_id = Some("tampered-id".to_string());

        // Should fail verification
        assert!(!verify_entry(&entry, key));
    }
}
