//! Data diffing for efficient sync
//!
//! Computes differences between local and remote data to minimize
//! data transfer during synchronization.

use crate::error::SyncResult;
use crate::protocol::{DataType, SyncOperation, SyncRecord};
use crate::DeviceId;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A change set representing differences between local and remote data
#[derive(Debug, Clone, Default)]
pub struct ChangeSet {
    /// Records that need to be sent to remote
    pub to_send: Vec<SyncRecord>,
    /// Records that need to be received from remote
    pub to_receive: Vec<String>, // Record IDs to request
    /// Records that have conflicts
    pub conflicts: Vec<ConflictInfo>,
}

/// Information about a conflicting record
#[derive(Debug, Clone)]
pub struct ConflictInfo {
    pub record_id: String,
    pub local_version: u64,
    pub remote_version: u64,
    pub local_modified: DateTime<Utc>,
    pub remote_modified: DateTime<Utc>,
}

/// Metadata about a record for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordMeta {
    pub record_id: String,
    pub sync_version: u64,
    pub modified_at: DateTime<Utc>,
    pub checksum: String,
}

/// Diff engine for computing changes
pub struct DiffEngine {
    device_id: DeviceId,
}

impl DiffEngine {
    /// Create a new diff engine
    pub fn new(device_id: DeviceId) -> Self {
        Self { device_id }
    }

    /// Compute the changeset between local and remote metadata
    pub fn compute_changeset(
        &self,
        local_meta: &[RecordMeta],
        remote_meta: &[RecordMeta],
        last_sync_version: u64,
    ) -> ChangeSet {
        let mut changeset = ChangeSet::default();

        // Index local and remote by record ID
        let local_map: HashMap<&str, &RecordMeta> =
            local_meta.iter().map(|m| (m.record_id.as_str(), m)).collect();
        let remote_map: HashMap<&str, &RecordMeta> =
            remote_meta.iter().map(|m| (m.record_id.as_str(), m)).collect();

        // All record IDs
        let all_ids: HashSet<&str> = local_map
            .keys()
            .chain(remote_map.keys())
            .copied()
            .collect();

        for id in all_ids {
            match (local_map.get(id), remote_map.get(id)) {
                // Record exists on both sides
                (Some(local), Some(remote)) => {
                    // Check if they're the same
                    if local.checksum == remote.checksum {
                        // No change needed
                        continue;
                    }

                    // Check versions
                    if local.sync_version > remote.sync_version {
                        // Local is newer, need to send
                        changeset.to_receive.push(id.to_string());
                    } else if remote.sync_version > local.sync_version {
                        // Remote is newer, need to receive
                        changeset.to_receive.push(id.to_string());
                    } else {
                        // Same version but different checksum = conflict
                        changeset.conflicts.push(ConflictInfo {
                            record_id: id.to_string(),
                            local_version: local.sync_version,
                            remote_version: remote.sync_version,
                            local_modified: local.modified_at,
                            remote_modified: remote.modified_at,
                        });
                    }
                }
                // Record only exists locally
                (Some(local), None) => {
                    // If local was modified after last sync, need to send
                    if local.sync_version > last_sync_version {
                        // Will add actual record later when building to_send
                    }
                }
                // Record only exists remotely
                (None, Some(_remote)) => {
                    // Need to receive from remote
                    changeset.to_receive.push(id.to_string());
                }
                // Should never happen
                (None, None) => {}
            }
        }

        changeset
    }

    /// Build sync records from changed data
    pub fn build_sync_records(
        &self,
        data_type: DataType,
        records: Vec<(String, Vec<u8>, DateTime<Utc>)>,
        sync_version: u64,
        operation: SyncOperation,
    ) -> Vec<SyncRecord> {
        records
            .into_iter()
            .map(|(id, data, modified)| SyncRecord {
                data_type,
                record_id: id,
                sync_version,
                origin_device: self.device_id.clone(),
                modified_at: modified,
                operation,
                data,
            })
            .collect()
    }

    /// Compute checksum for record data
    pub fn compute_checksum(data: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let hash = Sha256::digest(data);
        hex::encode(hash)
    }

    /// Create metadata from records
    pub fn create_metadata(
        records: &[(String, Vec<u8>, u64, DateTime<Utc>)],
    ) -> Vec<RecordMeta> {
        records
            .iter()
            .map(|(id, data, version, modified)| RecordMeta {
                record_id: id.clone(),
                sync_version: *version,
                modified_at: *modified,
                checksum: Self::compute_checksum(data),
            })
            .collect()
    }
}

/// Statistics about a diff operation
#[derive(Debug, Clone, Default)]
pub struct DiffStats {
    pub local_records: u32,
    pub remote_records: u32,
    pub to_send: u32,
    pub to_receive: u32,
    pub conflicts: u32,
    pub unchanged: u32,
}

impl DiffStats {
    pub fn from_changeset(changeset: &ChangeSet, local_count: u32, remote_count: u32) -> Self {
        Self {
            local_records: local_count,
            remote_records: remote_count,
            to_send: changeset.to_send.len() as u32,
            to_receive: changeset.to_receive.len() as u32,
            conflicts: changeset.conflicts.len() as u32,
            unchanged: local_count.saturating_sub(
                changeset.to_send.len() as u32 + changeset.conflicts.len() as u32,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device_id() -> DeviceId {
        DeviceId::new()
    }

    #[test]
    fn test_compute_checksum() {
        let data = b"test data";
        let checksum = DiffEngine::compute_checksum(data);
        assert_eq!(checksum.len(), 64); // SHA256 produces 32 bytes = 64 hex chars
    }

    #[test]
    fn test_same_checksum_same_data() {
        let data1 = b"test data";
        let data2 = b"test data";
        assert_eq!(
            DiffEngine::compute_checksum(data1),
            DiffEngine::compute_checksum(data2)
        );
    }

    #[test]
    fn test_different_checksum_different_data() {
        let data1 = b"test data 1";
        let data2 = b"test data 2";
        assert_ne!(
            DiffEngine::compute_checksum(data1),
            DiffEngine::compute_checksum(data2)
        );
    }

    #[test]
    fn test_changeset_empty() {
        let engine = DiffEngine::new(test_device_id());
        let changeset = engine.compute_changeset(&[], &[], 0);

        assert!(changeset.to_send.is_empty());
        assert!(changeset.to_receive.is_empty());
        assert!(changeset.conflicts.is_empty());
    }

    #[test]
    fn test_changeset_only_local() {
        let engine = DiffEngine::new(test_device_id());
        let now = Utc::now();

        let local = vec![RecordMeta {
            record_id: "1".to_string(),
            sync_version: 1,
            modified_at: now,
            checksum: "abc".to_string(),
        }];

        let changeset = engine.compute_changeset(&local, &[], 0);
        // Local-only records don't go to to_receive
        assert!(changeset.to_receive.is_empty());
    }

    #[test]
    fn test_changeset_only_remote() {
        let engine = DiffEngine::new(test_device_id());
        let now = Utc::now();

        let remote = vec![RecordMeta {
            record_id: "1".to_string(),
            sync_version: 1,
            modified_at: now,
            checksum: "abc".to_string(),
        }];

        let changeset = engine.compute_changeset(&[], &remote, 0);
        assert_eq!(changeset.to_receive.len(), 1);
    }

    #[test]
    fn test_changeset_same_record() {
        let engine = DiffEngine::new(test_device_id());
        let now = Utc::now();

        let meta = RecordMeta {
            record_id: "1".to_string(),
            sync_version: 1,
            modified_at: now,
            checksum: "abc".to_string(),
        };

        let changeset = engine.compute_changeset(&[meta.clone()], &[meta], 0);
        assert!(changeset.to_send.is_empty());
        assert!(changeset.to_receive.is_empty());
        assert!(changeset.conflicts.is_empty());
    }

    #[test]
    fn test_changeset_conflict() {
        let engine = DiffEngine::new(test_device_id());
        let now = Utc::now();

        let local = RecordMeta {
            record_id: "1".to_string(),
            sync_version: 1,
            modified_at: now,
            checksum: "abc".to_string(),
        };

        let remote = RecordMeta {
            record_id: "1".to_string(),
            sync_version: 1,
            modified_at: now,
            checksum: "def".to_string(), // Different checksum
        };

        let changeset = engine.compute_changeset(&[local], &[remote], 0);
        assert_eq!(changeset.conflicts.len(), 1);
    }
}
