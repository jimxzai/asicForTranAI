//! Audit logging for sync operations
//!
//! Records all sync events with HMAC integrity verification to
//! ensure tamper-evident logging.

use crate::error::{SyncError, SyncResult};
use crate::protocol::DataType;
use crate::DeviceId;
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::VecDeque;
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

/// Types of sync events that can be audited
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncEventType {
    /// Discovery started
    DiscoveryStarted,
    /// Discovery stopped
    DiscoveryStopped,
    /// Peer discovered
    PeerDiscovered { device_id: DeviceId },
    /// Peer lost
    PeerLost { device_id: DeviceId },
    /// Pairing initiated
    PairingInitiated { device_id: DeviceId },
    /// Pairing completed
    PairingCompleted { device_id: DeviceId },
    /// Pairing failed
    PairingFailed { device_id: DeviceId, reason: String },
    /// Device unpaired
    DeviceUnpaired { device_id: DeviceId },
    /// Sync session started
    SyncStarted { device_id: DeviceId },
    /// Sync session completed
    SyncCompleted {
        device_id: DeviceId,
        records_sent: u32,
        records_received: u32,
    },
    /// Sync session failed
    SyncFailed { device_id: DeviceId, reason: String },
    /// Record synced
    RecordSynced {
        data_type: DataType,
        record_id: String,
        direction: SyncDirection,
    },
    /// Conflict detected
    ConflictDetected {
        record_id: String,
        resolution: String,
    },
    /// Connection established
    ConnectionEstablished { device_id: DeviceId },
    /// Connection closed
    ConnectionClosed { device_id: DeviceId },
    /// Authentication failed
    AuthenticationFailed { device_id: DeviceId },
}

/// Direction of sync
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SyncDirection {
    Sent,
    Received,
}

/// An audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub event: SyncEventType,
    pub session_id: Option<String>,
    pub previous_hash: Option<String>,
    pub hmac: String,
}

impl AuditEntry {
    /// Create a new audit entry
    fn new(event: SyncEventType, session_id: Option<String>, hmac_key: &[u8]) -> Self {
        let mut entry = Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event,
            session_id,
            previous_hash: None,
            hmac: String::new(),
        };
        entry.hmac = entry.compute_hmac(hmac_key);
        entry
    }

    /// Create with chain link to previous entry
    fn with_chain(
        event: SyncEventType,
        session_id: Option<String>,
        previous_hash: String,
        hmac_key: &[u8],
    ) -> Self {
        let mut entry = Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event,
            session_id,
            previous_hash: Some(previous_hash),
            hmac: String::new(),
        };
        entry.hmac = entry.compute_hmac(hmac_key);
        entry
    }

    /// Compute HMAC for this entry
    fn compute_hmac(&self, key: &[u8]) -> String {
        let data = format!(
            "{}|{}|{:?}|{:?}|{:?}",
            self.id, self.timestamp, self.event, self.session_id, self.previous_hash
        );

        let mut mac = HmacSha256::new_from_slice(key)
            .expect("HMAC can take key of any size");
        mac.update(data.as_bytes());
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }

    /// Verify the HMAC of this entry
    pub fn verify(&self, key: &[u8]) -> bool {
        let computed = self.compute_hmac(key);
        // Constant time comparison
        if computed.len() != self.hmac.len() {
            return false;
        }
        let mut result = 0u8;
        for (a, b) in computed.bytes().zip(self.hmac.bytes()) {
            result |= a ^ b;
        }
        result == 0
    }

    /// Get the hash of this entry for chaining
    fn hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let data = format!("{}|{}", self.id, self.hmac);
        let hash = Sha256::digest(data.as_bytes());
        hex::encode(hash)
    }
}

/// Audit log manager
pub struct AuditLog {
    entries: VecDeque<AuditEntry>,
    hmac_key: Vec<u8>,
    max_entries: usize,
    current_session_id: Option<String>,
}

impl AuditLog {
    /// Create a new audit log
    pub fn new(hmac_key: Vec<u8>) -> Self {
        Self {
            entries: VecDeque::new(),
            hmac_key,
            max_entries: 10000,
            current_session_id: None,
        }
    }

    /// Create with custom max entries
    pub fn with_max_entries(hmac_key: Vec<u8>, max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            hmac_key,
            max_entries,
            current_session_id: None,
        }
    }

    /// Start a new sync session
    pub fn start_session(&mut self) -> String {
        let session_id = Uuid::new_v4().to_string();
        self.current_session_id = Some(session_id.clone());
        session_id
    }

    /// End the current session
    pub fn end_session(&mut self) {
        self.current_session_id = None;
    }

    /// Log an event
    pub fn log(&mut self, event: SyncEventType) {
        let entry = if let Some(last) = self.entries.back() {
            AuditEntry::with_chain(
                event,
                self.current_session_id.clone(),
                last.hash(),
                &self.hmac_key,
            )
        } else {
            AuditEntry::new(event, self.current_session_id.clone(), &self.hmac_key)
        };

        self.entries.push_back(entry);

        // Trim if over max
        while self.entries.len() > self.max_entries {
            self.entries.pop_front();
        }
    }

    /// Log discovery started
    pub fn log_discovery_started(&mut self) {
        self.log(SyncEventType::DiscoveryStarted);
    }

    /// Log peer discovered
    pub fn log_peer_discovered(&mut self, device_id: DeviceId) {
        self.log(SyncEventType::PeerDiscovered { device_id });
    }

    /// Log pairing completed
    pub fn log_pairing_completed(&mut self, device_id: DeviceId) {
        self.log(SyncEventType::PairingCompleted { device_id });
    }

    /// Log sync started
    pub fn log_sync_started(&mut self, device_id: DeviceId) {
        self.log(SyncEventType::SyncStarted { device_id });
    }

    /// Log sync completed
    pub fn log_sync_completed(&mut self, device_id: DeviceId, records_sent: u32, records_received: u32) {
        self.log(SyncEventType::SyncCompleted {
            device_id,
            records_sent,
            records_received,
        });
    }

    /// Log record synced
    pub fn log_record_synced(&mut self, data_type: DataType, record_id: String, direction: SyncDirection) {
        self.log(SyncEventType::RecordSynced {
            data_type,
            record_id,
            direction,
        });
    }

    /// Log conflict detected
    pub fn log_conflict(&mut self, record_id: String, resolution: String) {
        self.log(SyncEventType::ConflictDetected {
            record_id,
            resolution,
        });
    }

    /// Get all entries
    pub fn entries(&self) -> &VecDeque<AuditEntry> {
        &self.entries
    }

    /// Get entries for a specific session
    pub fn entries_for_session(&self, session_id: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.session_id.as_deref() == Some(session_id))
            .collect()
    }

    /// Get recent entries
    pub fn recent(&self, count: usize) -> Vec<&AuditEntry> {
        self.entries.iter().rev().take(count).collect()
    }

    /// Verify the integrity of the audit chain
    pub fn verify_chain(&self) -> SyncResult<bool> {
        let mut entries_iter = self.entries.iter();

        // Verify first entry
        if let Some(first) = entries_iter.next() {
            if !first.verify(&self.hmac_key) {
                return Err(SyncError::Internal("First entry HMAC verification failed".into()));
            }
            if first.previous_hash.is_some() {
                return Err(SyncError::Internal("First entry should not have previous hash".into()));
            }

            let mut previous_hash = first.hash();

            // Verify chain
            for entry in entries_iter {
                // Verify HMAC
                if !entry.verify(&self.hmac_key) {
                    return Err(SyncError::Internal(format!(
                        "Entry {} HMAC verification failed",
                        entry.id
                    )));
                }

                // Verify chain
                if entry.previous_hash.as_ref() != Some(&previous_hash) {
                    return Err(SyncError::Internal(format!(
                        "Entry {} chain verification failed",
                        entry.id
                    )));
                }

                previous_hash = entry.hash();
            }
        }

        Ok(true)
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Export entries as JSON
    pub fn export_json(&self) -> SyncResult<String> {
        let entries: Vec<&AuditEntry> = self.entries.iter().collect();
        serde_json::to_string_pretty(&entries).map_err(|e| SyncError::Serialization(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> Vec<u8> {
        vec![0u8; 32]
    }

    #[test]
    fn test_audit_entry_hmac() {
        let entry = AuditEntry::new(SyncEventType::DiscoveryStarted, None, &test_key());
        assert!(entry.verify(&test_key()));
    }

    #[test]
    fn test_audit_entry_tamper_detection() {
        let mut entry = AuditEntry::new(SyncEventType::DiscoveryStarted, None, &test_key());
        entry.id = "tampered".to_string();
        assert!(!entry.verify(&test_key()));
    }

    #[test]
    fn test_audit_log_new() {
        let log = AuditLog::new(test_key());
        assert!(log.entries().is_empty());
    }

    #[test]
    fn test_audit_log_single_entry() {
        let mut log = AuditLog::new(test_key());
        log.log_discovery_started();
        assert_eq!(log.entries().len(), 1);
        assert!(log.verify_chain().unwrap());
    }

    #[test]
    fn test_audit_log_chain() {
        let mut log = AuditLog::new(test_key());
        log.log_discovery_started();
        log.log_peer_discovered(DeviceId::new());
        log.log_pairing_completed(DeviceId::new());

        assert_eq!(log.entries().len(), 3);
        assert!(log.verify_chain().unwrap());
    }

    #[test]
    fn test_audit_log_session() {
        let mut log = AuditLog::new(test_key());
        let session_id = log.start_session();

        log.log_sync_started(DeviceId::new());
        log.log_sync_completed(DeviceId::new(), 10, 5);
        log.end_session();

        let session_entries = log.entries_for_session(&session_id);
        assert_eq!(session_entries.len(), 2);
    }

    #[test]
    fn test_audit_log_max_entries() {
        let mut log = AuditLog::with_max_entries(test_key(), 5);

        for _ in 0..10 {
            log.log_discovery_started();
        }

        assert_eq!(log.entries().len(), 5);
    }
}
