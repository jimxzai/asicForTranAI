//! Conflict resolution strategies for sync operations
//!
//! Provides various strategies for resolving conflicts when the same
//! record has been modified on multiple devices.

use crate::error::{SyncError, SyncResult};
use crate::protocol::SyncRecord;
use serde::{Deserialize, Serialize};

/// Conflict resolution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Strategy {
    /// Most recent modification wins (based on timestamp)
    LastWriteWins,
    /// Prefer local version
    LocalWins,
    /// Prefer remote version
    RemoteWins,
    /// Merge at field level (for holdings)
    FieldMerge,
    /// Create duplicate and let user resolve
    CreateDuplicate,
    /// Ask user to resolve (manual)
    Manual,
}

impl Default for Strategy {
    fn default() -> Self {
        Strategy::LastWriteWins
    }
}

/// Resolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Resolution {
    /// Use the local version
    UseLocal,
    /// Use the remote version
    UseRemote,
    /// Use a merged version
    UseMerged(Vec<u8>),
    /// Create both as duplicates
    CreateBoth,
    /// Needs manual resolution
    NeedsManual,
}

/// Information about a conflict
#[derive(Debug, Clone)]
pub struct Conflict {
    pub record_id: String,
    pub local_record: SyncRecord,
    pub remote_record: SyncRecord,
}

/// Conflict resolver
pub struct ConflictResolver {
    strategy: Strategy,
}

impl ConflictResolver {
    /// Create a new conflict resolver with the given strategy
    pub fn new(strategy: Strategy) -> Self {
        Self { strategy }
    }

    /// Get the current strategy
    pub fn strategy(&self) -> Strategy {
        self.strategy
    }

    /// Set the strategy
    pub fn set_strategy(&mut self, strategy: Strategy) {
        self.strategy = strategy;
    }

    /// Resolve a conflict between two records
    pub fn resolve(&self, conflict: &Conflict) -> SyncResult<Resolution> {
        match self.strategy {
            Strategy::LastWriteWins => self.resolve_last_write_wins(conflict),
            Strategy::LocalWins => Ok(Resolution::UseLocal),
            Strategy::RemoteWins => Ok(Resolution::UseRemote),
            Strategy::FieldMerge => self.resolve_field_merge(conflict),
            Strategy::CreateDuplicate => Ok(Resolution::CreateBoth),
            Strategy::Manual => Ok(Resolution::NeedsManual),
        }
    }

    /// Resolve using last-write-wins strategy
    fn resolve_last_write_wins(&self, conflict: &Conflict) -> SyncResult<Resolution> {
        // Compare timestamps
        if conflict.local_record.modified_at >= conflict.remote_record.modified_at {
            Ok(Resolution::UseLocal)
        } else {
            Ok(Resolution::UseRemote)
        }
    }

    /// Resolve using field-level merge for holdings
    fn resolve_field_merge(&self, conflict: &Conflict) -> SyncResult<Resolution> {
        // For holdings, we can merge certain fields
        // This is a simplified implementation - real implementation would
        // parse the data and merge fields individually

        // Parse both records as JSON
        let local: serde_json::Value = serde_json::from_slice(&conflict.local_record.data)
            .map_err(|e| SyncError::Conflict(format!("Failed to parse local record: {}", e)))?;
        let remote: serde_json::Value = serde_json::from_slice(&conflict.remote_record.data)
            .map_err(|e| SyncError::Conflict(format!("Failed to parse remote record: {}", e)))?;

        // Start with the local record
        let mut merged = local.clone();

        // Merge specific fields from remote if they're newer
        if let (Some(local_obj), Some(remote_obj)) = (merged.as_object_mut(), remote.as_object()) {
            // Fields that should take the more recent value
            let timestamp_fields = ["quantity", "cost_basis", "notes"];

            for field in timestamp_fields {
                if let Some(remote_val) = remote_obj.get(field) {
                    // In a real implementation, we'd compare field-level timestamps
                    // For now, use the remote value if local doesn't have it
                    if !local_obj.contains_key(field) {
                        local_obj.insert(field.to_string(), remote_val.clone());
                    }
                }
            }

            // Always use the most recent updated_at
            if let (Some(local_updated), Some(remote_updated)) =
                (local_obj.get("updated_at"), remote_obj.get("updated_at"))
            {
                if remote_updated > local_updated {
                    local_obj.insert("updated_at".to_string(), remote_updated.clone());
                }
            }
        }

        let merged_bytes = serde_json::to_vec(&merged)
            .map_err(|e| SyncError::Conflict(format!("Failed to serialize merged record: {}", e)))?;

        Ok(Resolution::UseMerged(merged_bytes))
    }

    /// Batch resolve multiple conflicts
    pub fn resolve_batch(&self, conflicts: &[Conflict]) -> Vec<SyncResult<Resolution>> {
        conflicts.iter().map(|c| self.resolve(c)).collect()
    }
}

/// Statistics about conflict resolution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConflictStats {
    pub total_conflicts: u32,
    pub resolved_local: u32,
    pub resolved_remote: u32,
    pub resolved_merged: u32,
    pub created_duplicates: u32,
    pub needs_manual: u32,
}

impl ConflictStats {
    /// Record a resolution
    pub fn record(&mut self, resolution: &Resolution) {
        self.total_conflicts += 1;
        match resolution {
            Resolution::UseLocal => self.resolved_local += 1,
            Resolution::UseRemote => self.resolved_remote += 1,
            Resolution::UseMerged(_) => self.resolved_merged += 1,
            Resolution::CreateBoth => self.created_duplicates += 1,
            Resolution::NeedsManual => self.needs_manual += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{DataType, SyncOperation};
    use crate::DeviceId;
    use chrono::Utc;

    fn create_test_record(data: &str, modified: chrono::DateTime<Utc>) -> SyncRecord {
        SyncRecord {
            data_type: DataType::Holdings,
            record_id: "test-1".to_string(),
            sync_version: 1,
            origin_device: DeviceId::new(),
            modified_at: modified,
            operation: SyncOperation::Update,
            data: data.as_bytes().to_vec(),
        }
    }

    #[test]
    fn test_last_write_wins_local() {
        let resolver = ConflictResolver::new(Strategy::LastWriteWins);

        let now = Utc::now();
        let earlier = now - chrono::Duration::hours(1);

        let conflict = Conflict {
            record_id: "test-1".to_string(),
            local_record: create_test_record("{}", now),
            remote_record: create_test_record("{}", earlier),
        };

        let resolution = resolver.resolve(&conflict).unwrap();
        assert!(matches!(resolution, Resolution::UseLocal));
    }

    #[test]
    fn test_last_write_wins_remote() {
        let resolver = ConflictResolver::new(Strategy::LastWriteWins);

        let now = Utc::now();
        let earlier = now - chrono::Duration::hours(1);

        let conflict = Conflict {
            record_id: "test-1".to_string(),
            local_record: create_test_record("{}", earlier),
            remote_record: create_test_record("{}", now),
        };

        let resolution = resolver.resolve(&conflict).unwrap();
        assert!(matches!(resolution, Resolution::UseRemote));
    }

    #[test]
    fn test_local_wins() {
        let resolver = ConflictResolver::new(Strategy::LocalWins);
        let now = Utc::now();

        let conflict = Conflict {
            record_id: "test-1".to_string(),
            local_record: create_test_record("{}", now),
            remote_record: create_test_record("{}", now),
        };

        let resolution = resolver.resolve(&conflict).unwrap();
        assert!(matches!(resolution, Resolution::UseLocal));
    }

    #[test]
    fn test_remote_wins() {
        let resolver = ConflictResolver::new(Strategy::RemoteWins);
        let now = Utc::now();

        let conflict = Conflict {
            record_id: "test-1".to_string(),
            local_record: create_test_record("{}", now),
            remote_record: create_test_record("{}", now),
        };

        let resolution = resolver.resolve(&conflict).unwrap();
        assert!(matches!(resolution, Resolution::UseRemote));
    }

    #[test]
    fn test_create_duplicate() {
        let resolver = ConflictResolver::new(Strategy::CreateDuplicate);
        let now = Utc::now();

        let conflict = Conflict {
            record_id: "test-1".to_string(),
            local_record: create_test_record("{}", now),
            remote_record: create_test_record("{}", now),
        };

        let resolution = resolver.resolve(&conflict).unwrap();
        assert!(matches!(resolution, Resolution::CreateBoth));
    }

    #[test]
    fn test_conflict_stats() {
        let mut stats = ConflictStats::default();

        stats.record(&Resolution::UseLocal);
        stats.record(&Resolution::UseRemote);
        stats.record(&Resolution::UseMerged(vec![]));

        assert_eq!(stats.total_conflicts, 3);
        assert_eq!(stats.resolved_local, 1);
        assert_eq!(stats.resolved_remote, 1);
        assert_eq!(stats.resolved_merged, 1);
    }
}
