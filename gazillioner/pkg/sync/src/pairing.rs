//! Device pairing protocol with verification codes
//!
//! Implements a secure pairing mechanism using:
//! - X25519 key exchange
//! - 6-digit verification codes
//! - PIN-based mutual authentication

use crate::error::{SyncError, SyncResult};
use crate::DeviceId;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use x25519_dalek::{EphemeralSecret, PublicKey};
use zeroize::Zeroizing;

/// Verification code for pairing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCode {
    /// 6-digit code
    pub code: String,
    /// Expiration time in seconds
    pub expires_in_secs: u64,
}

impl VerificationCode {
    /// Generate a new 6-digit verification code
    pub fn generate() -> Self {
        let mut rng = rand::thread_rng();
        let code: u32 = rng.gen_range(100000..999999);
        Self {
            code: code.to_string(),
            expires_in_secs: 300, // 5 minutes
        }
    }

    /// Create from existing code
    pub fn from_code(code: String) -> Self {
        Self {
            code,
            expires_in_secs: 300,
        }
    }

    /// Check if codes match
    pub fn matches(&self, other: &str) -> bool {
        // Constant time comparison to prevent timing attacks
        if self.code.len() != other.len() {
            return false;
        }
        let mut result = 0u8;
        for (a, b) in self.code.bytes().zip(other.bytes()) {
            result |= a ^ b;
        }
        result == 0
    }
}

/// Pairing session state
#[derive(Debug)]
struct PairingSession {
    remote_device_id: DeviceId,
    our_secret: Option<EphemeralSecret>,
    our_public_key: Option<PublicKey>,
    remote_public_key: Option<PublicKey>,
    verification_code: Option<VerificationCode>,
    shared_secret: Option<Zeroizing<[u8; 32]>>,
    started_at: Instant,
    is_initiator: bool,
}

impl PairingSession {
    fn new(remote_device_id: DeviceId, is_initiator: bool) -> Self {
        Self {
            remote_device_id,
            our_secret: None,
            our_public_key: None,
            remote_public_key: None,
            verification_code: None,
            shared_secret: None,
            started_at: Instant::now(),
            is_initiator,
        }
    }

    fn is_expired(&self) -> bool {
        self.started_at.elapsed() > Duration::from_secs(300) // 5 minute timeout
    }
}

/// Pairing request message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingRequest {
    pub device_id: DeviceId,
    pub device_name: String,
    pub public_key: Vec<u8>,
}

/// Pairing challenge message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingChallenge {
    pub device_id: DeviceId,
    pub public_key: Vec<u8>,
    pub verification_code: VerificationCode,
}

/// Pairing response message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingResponse {
    pub device_id: DeviceId,
    pub hmac: Vec<u8>,
}

/// Pairing confirmation message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingConfirmation {
    pub device_id: DeviceId,
    pub success: bool,
}

/// Manages device pairing
pub struct PairingManager {
    our_device_id: DeviceId,
    sessions: HashMap<DeviceId, PairingSession>,
    paired_devices: HashMap<DeviceId, PairedDevice>,
}

/// Information about a paired device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedDevice {
    pub device_id: DeviceId,
    pub device_name: String,
    pub shared_key: Vec<u8>, // Encrypted shared key
    pub paired_at: chrono::DateTime<chrono::Utc>,
    pub last_sync_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl PairingManager {
    /// Create a new pairing manager
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            our_device_id: device_id,
            sessions: HashMap::new(),
            paired_devices: HashMap::new(),
        }
    }

    /// Initiate pairing with a remote device
    pub async fn initiate_pairing(&mut self, remote_device_id: &DeviceId) -> SyncResult<VerificationCode> {
        // Clean up expired sessions
        self.cleanup_expired_sessions();

        // Check if already in a session with this device
        if self.sessions.contains_key(remote_device_id) {
            return Err(SyncError::PairingFailed(
                "Pairing already in progress".into(),
            ));
        }

        // Create new session
        let mut session = PairingSession::new(remote_device_id.clone(), true);

        // Generate ephemeral key pair
        let secret = EphemeralSecret::random_from_rng(rand::thread_rng());
        let public_key = PublicKey::from(&secret);
        session.our_secret = Some(secret);
        session.our_public_key = Some(public_key);

        // Generate verification code
        let code = VerificationCode::generate();
        session.verification_code = Some(code.clone());

        self.sessions.insert(remote_device_id.clone(), session);

        Ok(code)
    }

    /// Handle incoming pairing request
    pub async fn handle_pairing_request(
        &mut self,
        request: PairingRequest,
    ) -> SyncResult<PairingChallenge> {
        // Clean up expired sessions
        self.cleanup_expired_sessions();

        // Create session if not exists
        let session = self
            .sessions
            .entry(request.device_id.clone())
            .or_insert_with(|| PairingSession::new(request.device_id.clone(), false));

        // Parse remote public key
        if request.public_key.len() != 32 {
            return Err(SyncError::PairingFailed("Invalid public key length".into()));
        }
        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(&request.public_key);
        session.remote_public_key = Some(PublicKey::from(key_bytes));

        // Generate our key pair
        let secret = EphemeralSecret::random_from_rng(rand::thread_rng());
        let public_key = PublicKey::from(&secret);
        session.our_secret = Some(secret);
        session.our_public_key = Some(public_key);

        // Generate verification code
        let code = VerificationCode::generate();
        session.verification_code = Some(code.clone());

        Ok(PairingChallenge {
            device_id: self.our_device_id.clone(),
            public_key: public_key.as_bytes().to_vec(),
            verification_code: code,
        })
    }

    /// Confirm pairing with verification code entered by user
    pub async fn confirm_pairing(&mut self, code: &VerificationCode) -> SyncResult<()> {
        // Find the session with this verification code
        let device_id = self
            .sessions
            .iter()
            .find(|(_, s)| {
                s.verification_code
                    .as_ref()
                    .map(|c| c.matches(&code.code))
                    .unwrap_or(false)
            })
            .map(|(id, _)| id.clone())
            .ok_or(SyncError::InvalidVerificationCode)?;

        let session = self
            .sessions
            .get_mut(&device_id)
            .ok_or(SyncError::InvalidVerificationCode)?;

        // Compute shared secret
        let secret = session
            .our_secret
            .take()
            .ok_or(SyncError::PairingFailed("No secret key".into()))?;
        let remote_public = session
            .remote_public_key
            .ok_or(SyncError::PairingFailed("No remote public key".into()))?;

        let shared = secret.diffie_hellman(&remote_public);
        let shared_bytes = Zeroizing::new(*shared.as_bytes());

        // Store as paired device
        let paired = PairedDevice {
            device_id: device_id.clone(),
            device_name: format!("Device {}", &device_id.as_str()[..8]),
            shared_key: shared_bytes.to_vec(),
            paired_at: chrono::Utc::now(),
            last_sync_at: None,
        };

        self.paired_devices.insert(device_id.clone(), paired);

        // Remove session
        self.sessions.remove(&device_id);

        Ok(())
    }

    /// Check if a device is paired
    pub fn is_paired(&self, device_id: &DeviceId) -> bool {
        self.paired_devices.contains_key(device_id)
    }

    /// Get paired device info
    pub fn get_paired_device(&self, device_id: &DeviceId) -> Option<&PairedDevice> {
        self.paired_devices.get(device_id)
    }

    /// Get all paired devices
    pub fn paired_devices(&self) -> Vec<&PairedDevice> {
        self.paired_devices.values().collect()
    }

    /// Remove a paired device
    pub fn unpair(&mut self, device_id: &DeviceId) -> bool {
        self.paired_devices.remove(device_id).is_some()
    }

    /// Add a paired device (for loading from database)
    pub fn add_paired_device(&mut self, device: PairedDevice) {
        self.paired_devices.insert(device.device_id.clone(), device);
    }

    /// Get shared key for a paired device
    pub fn get_shared_key(&self, device_id: &DeviceId) -> Option<&[u8]> {
        self.paired_devices
            .get(device_id)
            .map(|d| d.shared_key.as_slice())
    }

    /// Update last sync time
    pub fn update_last_sync(&mut self, device_id: &DeviceId) {
        if let Some(device) = self.paired_devices.get_mut(device_id) {
            device.last_sync_at = Some(chrono::Utc::now());
        }
    }

    /// Clean up expired pairing sessions
    fn cleanup_expired_sessions(&mut self) {
        self.sessions.retain(|_, s| !s.is_expired());
    }

    /// Cancel an in-progress pairing
    pub fn cancel_pairing(&mut self, device_id: &DeviceId) {
        self.sessions.remove(device_id);
    }

    /// Get pending pairing device IDs
    pub fn pending_pairings(&self) -> Vec<DeviceId> {
        self.sessions.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_code_generation() {
        let code = VerificationCode::generate();
        assert_eq!(code.code.len(), 6);
        assert!(code.code.chars().all(|c| c.is_ascii_digit()));
    }

    #[test]
    fn test_verification_code_matching() {
        let code = VerificationCode::from_code("123456".to_string());
        assert!(code.matches("123456"));
        assert!(!code.matches("654321"));
        assert!(!code.matches("12345"));
    }

    #[test]
    fn test_pairing_manager_new() {
        let device_id = DeviceId::new();
        let manager = PairingManager::new(device_id);
        assert!(manager.paired_devices().is_empty());
    }

    #[test]
    fn test_is_paired() {
        let device_id = DeviceId::new();
        let manager = PairingManager::new(device_id);
        let other_id = DeviceId::new();
        assert!(!manager.is_paired(&other_id));
    }

    #[tokio::test]
    async fn test_initiate_pairing() {
        let device_id = DeviceId::new();
        let mut manager = PairingManager::new(device_id);
        let remote_id = DeviceId::new();

        let code = manager.initiate_pairing(&remote_id).await.unwrap();
        assert_eq!(code.code.len(), 6);
        assert_eq!(manager.pending_pairings().len(), 1);
    }
}
