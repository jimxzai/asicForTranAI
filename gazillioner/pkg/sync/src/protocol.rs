//! Sync protocol message definitions and handling
//!
//! Defines the message format for P2P synchronization including:
//! - Handshake and authentication
//! - Data transfer
//! - Conflict notification
//! - Acknowledgments

use crate::error::{SyncError, SyncResult};
use crate::{DeviceId, DeviceInfo, PROTOCOL_VERSION};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Protocol message types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum MessageType {
    /// Initial handshake
    Hello = 1,
    /// Handshake acknowledgment
    HelloAck = 2,
    /// Authentication challenge
    AuthChallenge = 3,
    /// Authentication response
    AuthResponse = 4,
    /// Sync request
    SyncRequest = 5,
    /// Sync data batch
    SyncData = 6,
    /// Sync acknowledgment
    SyncAck = 7,
    /// Conflict notification
    ConflictNotify = 8,
    /// Conflict resolution
    ConflictResolve = 9,
    /// Sync complete
    SyncComplete = 10,
    /// Error message
    Error = 11,
    /// Ping for keepalive
    Ping = 12,
    /// Pong response
    Pong = 13,
    /// Disconnect notification
    Disconnect = 14,
}

/// Sync message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncMessage {
    /// Message type
    pub msg_type: MessageType,
    /// Message ID for correlation
    pub msg_id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Message payload (encrypted in transit)
    pub payload: Vec<u8>,
    /// HMAC for integrity verification
    pub hmac: Vec<u8>,
}

impl SyncMessage {
    /// Create a new sync message
    pub fn new(msg_type: MessageType, payload: Vec<u8>) -> Self {
        Self {
            msg_type,
            msg_id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            payload,
            hmac: Vec::new(),
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> SyncResult<Vec<u8>> {
        bincode::serialize(self).map_err(|e| SyncError::Serialization(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> SyncResult<Self> {
        bincode::deserialize(data).map_err(|e| SyncError::Serialization(e.to_string()))
    }
}

/// Hello message payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloPayload {
    pub device_info: DeviceInfo,
    pub protocol_version: u32,
    pub nonce: Vec<u8>,
}

impl HelloPayload {
    pub fn new(device_info: DeviceInfo) -> Self {
        let mut nonce = vec![0u8; 32];
        rand::Rng::fill(&mut rand::thread_rng(), &mut nonce[..]);
        Self {
            device_info,
            protocol_version: PROTOCOL_VERSION,
            nonce,
        }
    }
}

/// Sync request payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRequestPayload {
    /// What data types to sync
    pub data_types: Vec<DataType>,
    /// Last sync version we have
    pub last_sync_version: u64,
    /// Maximum batch size
    pub max_batch_size: u32,
}

/// Types of data that can be synced
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Portfolios,
    Holdings,
    Watchlist,
    WalletAddresses,
    Conversations,
}

/// Sync data batch payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncDataPayload {
    /// Batch sequence number
    pub sequence: u32,
    /// Total number of batches
    pub total_batches: u32,
    /// Data records in this batch
    pub records: Vec<SyncRecord>,
    /// Is this the last batch?
    pub is_final: bool,
}

/// A single sync record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRecord {
    /// Record type
    pub data_type: DataType,
    /// Record ID
    pub record_id: String,
    /// Sync version
    pub sync_version: u64,
    /// Origin device ID
    pub origin_device: DeviceId,
    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,
    /// Operation type
    pub operation: SyncOperation,
    /// Serialized record data
    pub data: Vec<u8>,
}

/// Sync operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncOperation {
    Create,
    Update,
    Delete,
}

/// Sync acknowledgment payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncAckPayload {
    /// Sequence number being acknowledged
    pub sequence: u32,
    /// Number of records processed
    pub records_processed: u32,
    /// Any record IDs that had conflicts
    pub conflicts: Vec<String>,
}

/// Conflict notification payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictPayload {
    /// Record that has conflict
    pub record_id: String,
    /// Our version of the record
    pub local_record: SyncRecord,
    /// Remote version of the record
    pub remote_record: SyncRecord,
}

/// Error payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPayload {
    pub code: u32,
    pub message: String,
    pub recoverable: bool,
}

/// Protocol handler
pub struct SyncProtocol {
    device_info: DeviceInfo,
}

impl SyncProtocol {
    /// Create a new protocol handler
    pub fn new(device_info: DeviceInfo) -> Self {
        Self { device_info }
    }

    /// Create a hello message
    pub fn create_hello(&self) -> SyncResult<SyncMessage> {
        let payload = HelloPayload::new(self.device_info.clone());
        let payload_bytes = bincode::serialize(&payload)?;
        Ok(SyncMessage::new(MessageType::Hello, payload_bytes))
    }

    /// Parse a hello message
    pub fn parse_hello(&self, msg: &SyncMessage) -> SyncResult<HelloPayload> {
        if msg.msg_type != MessageType::Hello {
            return Err(SyncError::InvalidMessage("Expected Hello message".into()));
        }
        let payload: HelloPayload = bincode::deserialize(&msg.payload)?;

        // Check protocol version
        if payload.protocol_version != PROTOCOL_VERSION {
            return Err(SyncError::ProtocolMismatch {
                local: PROTOCOL_VERSION,
                remote: payload.protocol_version,
            });
        }

        Ok(payload)
    }

    /// Create a sync request message
    pub fn create_sync_request(
        &self,
        data_types: Vec<DataType>,
        last_sync_version: u64,
    ) -> SyncResult<SyncMessage> {
        let payload = SyncRequestPayload {
            data_types,
            last_sync_version,
            max_batch_size: 100,
        };
        let payload_bytes = bincode::serialize(&payload)?;
        Ok(SyncMessage::new(MessageType::SyncRequest, payload_bytes))
    }

    /// Create a sync data message
    pub fn create_sync_data(
        &self,
        records: Vec<SyncRecord>,
        sequence: u32,
        total_batches: u32,
        is_final: bool,
    ) -> SyncResult<SyncMessage> {
        let payload = SyncDataPayload {
            sequence,
            total_batches,
            records,
            is_final,
        };
        let payload_bytes = bincode::serialize(&payload)?;
        Ok(SyncMessage::new(MessageType::SyncData, payload_bytes))
    }

    /// Create an acknowledgment message
    pub fn create_ack(
        &self,
        sequence: u32,
        records_processed: u32,
        conflicts: Vec<String>,
    ) -> SyncResult<SyncMessage> {
        let payload = SyncAckPayload {
            sequence,
            records_processed,
            conflicts,
        };
        let payload_bytes = bincode::serialize(&payload)?;
        Ok(SyncMessage::new(MessageType::SyncAck, payload_bytes))
    }

    /// Create an error message
    pub fn create_error(&self, code: u32, message: String, recoverable: bool) -> SyncResult<SyncMessage> {
        let payload = ErrorPayload {
            code,
            message,
            recoverable,
        };
        let payload_bytes = bincode::serialize(&payload)?;
        Ok(SyncMessage::new(MessageType::Error, payload_bytes))
    }

    /// Create a ping message
    pub fn create_ping(&self) -> SyncResult<SyncMessage> {
        Ok(SyncMessage::new(MessageType::Ping, Vec::new()))
    }

    /// Create a pong message
    pub fn create_pong(&self, ping_id: String) -> SyncResult<SyncMessage> {
        let mut msg = SyncMessage::new(MessageType::Pong, Vec::new());
        msg.msg_id = ping_id;
        Ok(msg)
    }

    /// Create a sync complete message
    pub fn create_sync_complete(&self) -> SyncResult<SyncMessage> {
        Ok(SyncMessage::new(MessageType::SyncComplete, Vec::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceId;

    fn test_device_info() -> DeviceInfo {
        DeviceInfo::new(DeviceId::new(), "Test Device".to_string())
    }

    #[test]
    fn test_message_serialization() {
        let msg = SyncMessage::new(MessageType::Ping, Vec::new());
        let bytes = msg.to_bytes().unwrap();
        let decoded = SyncMessage::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.msg_type, MessageType::Ping);
    }

    #[test]
    fn test_protocol_hello() {
        let protocol = SyncProtocol::new(test_device_info());
        let hello = protocol.create_hello().unwrap();
        assert_eq!(hello.msg_type, MessageType::Hello);
    }

    #[test]
    fn test_protocol_sync_request() {
        let protocol = SyncProtocol::new(test_device_info());
        let msg = protocol
            .create_sync_request(vec![DataType::Portfolios, DataType::Holdings], 0)
            .unwrap();
        assert_eq!(msg.msg_type, MessageType::SyncRequest);
    }
}
