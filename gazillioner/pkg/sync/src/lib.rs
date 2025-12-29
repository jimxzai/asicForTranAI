//! Gazillioner P2P Sync Library
//!
//! Provides secure peer-to-peer synchronization between devices on the local network.
//!
//! # Features
//! - mDNS service discovery for automatic device detection
//! - TLS 1.3 encrypted transport with mutual authentication
//! - Device pairing with verification codes
//! - Conflict resolution strategies
//! - Audit logging for all sync operations
//!
//! # Architecture
//! ```text
//! Device A                              Device B
//! ┌─────────────┐                      ┌─────────────┐
//! │ SyncManager │◄─── mDNS Discovery ──►│ SyncManager │
//! │      │      │                      │      │      │
//! │ P2P Server  │◄══ TLS 1.3 Mutual ══►│ P2P Client  │
//! │      │      │    Authentication    │      │      │
//! │ SQLCipher DB│                      │ SQLCipher DB│
//! └─────────────┘                      └─────────────┘
//! ```

pub mod audit;
pub mod conflict;
pub mod diff;
pub mod discovery;
pub mod error;
pub mod ffi;
pub mod pairing;
pub mod protocol;
pub mod transport;

// Re-exports
pub use error::{SyncError, SyncResult};
pub use discovery::ServiceDiscovery;
pub use pairing::{PairingManager, VerificationCode};
pub use protocol::{SyncMessage, SyncProtocol};
pub use transport::SecureTransport;
pub use conflict::{ConflictResolver, Resolution, Strategy};

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Service name for mDNS discovery
pub const SERVICE_NAME: &str = "_gazillioner._tcp.local.";

/// Default sync port
pub const DEFAULT_PORT: u16 = 47392;

/// Protocol version
pub const PROTOCOL_VERSION: u32 = 1;

/// Device identifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DeviceId(pub String);

impl DeviceId {
    /// Generate a new random device ID
    pub fn new() -> Self {
        DeviceId(uuid::Uuid::new_v4().to_string())
    }

    /// Create from an existing string
    pub fn from_string(s: String) -> Self {
        DeviceId(s)
    }

    /// Get the device ID as a string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for DeviceId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Information about this device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub device_id: DeviceId,
    pub device_name: String,
    pub platform: String,
    pub app_version: String,
    pub protocol_version: u32,
}

impl DeviceInfo {
    pub fn new(device_id: DeviceId, device_name: String) -> Self {
        Self {
            device_id,
            device_name,
            platform: std::env::consts::OS.to_string(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            protocol_version: PROTOCOL_VERSION,
        }
    }
}

/// Information about a discovered peer device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub device_id: DeviceId,
    pub device_name: String,
    pub addresses: Vec<std::net::IpAddr>,
    pub port: u16,
    pub discovered_at: DateTime<Utc>,
    pub is_paired: bool,
}

/// Sync session state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncState {
    /// Not connected to any peer
    Idle,
    /// Discovering peers on the network
    Discovering,
    /// Connecting to a peer
    Connecting,
    /// Waiting for pairing confirmation
    Pairing,
    /// Syncing data
    Syncing,
    /// Sync completed successfully
    Completed,
    /// An error occurred
    Error,
}

/// Statistics about a sync session
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    pub records_sent: u32,
    pub records_received: u32,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub conflicts_resolved: u32,
    pub duration_ms: u64,
}

/// Configuration for the sync service
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// This device's information
    pub device_info: DeviceInfo,
    /// Port to listen on for incoming connections
    pub listen_port: u16,
    /// Whether to enable mDNS discovery
    pub enable_discovery: bool,
    /// Maximum number of concurrent sync sessions
    pub max_concurrent_syncs: usize,
    /// Timeout for sync operations in seconds
    pub sync_timeout_secs: u64,
    /// Conflict resolution strategy
    pub conflict_strategy: Strategy,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            device_info: DeviceInfo::new(DeviceId::new(), "Unknown Device".to_string()),
            listen_port: DEFAULT_PORT,
            enable_discovery: true,
            max_concurrent_syncs: 2,
            sync_timeout_secs: 300,
            conflict_strategy: Strategy::LastWriteWins,
        }
    }
}

/// Main sync manager that coordinates all sync operations
pub struct SyncManager {
    config: SyncConfig,
    state: SyncState,
    discovery: Option<ServiceDiscovery>,
    pairing: PairingManager,
}

impl SyncManager {
    /// Create a new sync manager with the given configuration
    pub fn new(config: SyncConfig) -> Self {
        Self {
            pairing: PairingManager::new(config.device_info.device_id.clone()),
            config,
            state: SyncState::Idle,
            discovery: None,
        }
    }

    /// Get the current sync state
    pub fn state(&self) -> SyncState {
        self.state
    }

    /// Get the device info
    pub fn device_info(&self) -> &DeviceInfo {
        &self.config.device_info
    }

    /// Start mDNS discovery
    pub async fn start_discovery(&mut self) -> SyncResult<()> {
        if !self.config.enable_discovery {
            return Err(SyncError::Config("Discovery is disabled".into()));
        }

        self.state = SyncState::Discovering;
        let discovery = ServiceDiscovery::new(
            self.config.device_info.clone(),
            self.config.listen_port,
        )?;
        discovery.start().await?;
        self.discovery = Some(discovery);
        Ok(())
    }

    /// Stop mDNS discovery
    pub async fn stop_discovery(&mut self) -> SyncResult<()> {
        if let Some(discovery) = self.discovery.take() {
            discovery.stop().await?;
        }
        if self.state == SyncState::Discovering {
            self.state = SyncState::Idle;
        }
        Ok(())
    }

    /// Get list of discovered peers
    pub fn discovered_peers(&self) -> Vec<PeerInfo> {
        self.discovery
            .as_ref()
            .map(|d| d.peers())
            .unwrap_or_default()
    }

    /// Initiate pairing with a peer device
    pub async fn initiate_pairing(&mut self, peer: &PeerInfo) -> SyncResult<VerificationCode> {
        self.state = SyncState::Pairing;
        self.pairing.initiate_pairing(&peer.device_id).await
    }

    /// Confirm pairing with verification code
    pub async fn confirm_pairing(&mut self, code: &VerificationCode) -> SyncResult<()> {
        self.pairing.confirm_pairing(code).await?;
        self.state = SyncState::Idle;
        Ok(())
    }

    /// Start sync with a paired device
    pub async fn sync_with(&mut self, device_id: &DeviceId) -> SyncResult<SyncStats> {
        if !self.pairing.is_paired(device_id) {
            return Err(SyncError::NotPaired);
        }

        self.state = SyncState::Syncing;

        // TODO: Implement actual sync logic
        let stats = SyncStats::default();

        self.state = SyncState::Completed;
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_id_generation() {
        let id1 = DeviceId::new();
        let id2 = DeviceId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_default_config() {
        let config = SyncConfig::default();
        assert_eq!(config.listen_port, DEFAULT_PORT);
        assert!(config.enable_discovery);
    }

    #[test]
    fn test_sync_manager_initial_state() {
        let config = SyncConfig::default();
        let manager = SyncManager::new(config);
        assert_eq!(manager.state(), SyncState::Idle);
    }
}
