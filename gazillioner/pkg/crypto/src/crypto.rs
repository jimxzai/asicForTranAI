//! Cryptographic utilities for key derivation and encryption

use argon2::{
    password_hash::{rand_core::OsRng, PasswordHasher, SaltString},
    Argon2, Params, Version,
};
use ring::rand::{SecureRandom, SystemRandom};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

/// Key derivation parameters
pub struct KeyDerivation {
    /// Device root key (stored securely in device)
    device_key: [u8; 32],
    /// Salt for key derivation
    salt: [u8; 32],
}

impl KeyDerivation {
    /// Create new key derivation with random device key
    pub fn new() -> Result<Self> {
        let rng = SystemRandom::new();
        let mut device_key = [0u8; 32];
        let mut salt = [0u8; 32];

        rng.fill(&mut device_key)
            .map_err(|_| Error::KeyDerivation("Failed to generate device key".into()))?;
        rng.fill(&mut salt)
            .map_err(|_| Error::KeyDerivation("Failed to generate salt".into()))?;

        Ok(Self { device_key, salt })
    }

    /// Create from existing keys (for recovery)
    pub fn from_keys(device_key: [u8; 32], salt: [u8; 32]) -> Self {
        Self { device_key, salt }
    }

    /// Derive database encryption key from PIN
    ///
    /// Uses Argon2id with secure parameters for PIN stretching
    pub fn derive_db_key(&self, pin: &str) -> Result<[u8; 32]> {
        // Validate PIN format (6 digits)
        if !Self::validate_pin(pin) {
            return Err(Error::InvalidPinFormat);
        }

        // Combine device key with PIN
        let mut combined = Vec::with_capacity(self.device_key.len() + pin.len());
        combined.extend_from_slice(&self.device_key);
        combined.extend_from_slice(pin.as_bytes());

        // Use Argon2id with strong parameters
        // These parameters are tuned for ~1 second derivation time
        let params = Params::new(
            65536,  // 64 MiB memory
            3,      // 3 iterations
            4,      // 4 parallel lanes
            Some(32), // 32 byte output
        )
        .map_err(|e| Error::KeyDerivation(format!("Invalid Argon2 params: {}", e)))?;

        let argon2 = Argon2::new(argon2::Algorithm::Argon2id, Version::V0x13, params);

        let salt_string = SaltString::encode_b64(&self.salt)
            .map_err(|e| Error::KeyDerivation(format!("Salt encoding failed: {}", e)))?;

        let hash = argon2
            .hash_password(&combined, &salt_string)
            .map_err(|e| Error::KeyDerivation(format!("Argon2 hashing failed: {}", e)))?;

        let hash_bytes = hash.hash.ok_or_else(|| {
            Error::KeyDerivation("No hash output".into())
        })?;

        let mut key = [0u8; 32];
        key.copy_from_slice(hash_bytes.as_bytes());
        Ok(key)
    }

    /// Validate PIN format (6 digits)
    pub fn validate_pin(pin: &str) -> bool {
        pin.len() == 6 && pin.chars().all(|c| c.is_ascii_digit())
    }

    /// Generate a secure random PIN
    pub fn generate_pin() -> Result<String> {
        let rng = SystemRandom::new();
        let mut bytes = [0u8; 4];
        rng.fill(&mut bytes)
            .map_err(|_| Error::KeyDerivation("Failed to generate random bytes".into()))?;

        // Convert to 6-digit number
        let num = u32::from_le_bytes(bytes) % 1_000_000;
        Ok(format!("{:06}", num))
    }

    /// Get device key bytes (for backup)
    pub fn device_key(&self) -> &[u8; 32] {
        &self.device_key
    }

    /// Get salt bytes (for backup)
    pub fn salt(&self) -> &[u8; 32] {
        &self.salt
    }

    /// Convert key to hex string for SQLCipher
    /// SQLCipher raw key format requires double quotes: PRAGMA key = "x'...'";
    pub fn key_to_hex(key: &[u8; 32]) -> String {
        format!("\"x'{}'\"", hex::encode(key))
    }
}

/// HMAC utilities for audit log integrity
pub mod hmac {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    type HmacSha256 = Hmac<Sha256>;

    /// Generate HMAC for audit log entry
    pub fn generate(key: &[u8], data: &[u8]) -> [u8; 32] {
        let mut mac = HmacSha256::new_from_slice(key)
            .expect("HMAC can take key of any size");
        mac.update(data);
        let result = mac.finalize();
        let mut output = [0u8; 32];
        output.copy_from_slice(&result.into_bytes());
        output
    }

    /// Verify HMAC for audit log entry
    pub fn verify(key: &[u8], data: &[u8], expected: &[u8; 32]) -> bool {
        let computed = generate(key, data);
        // Constant-time comparison
        ring::constant_time::verify_slices_are_equal(&computed, expected).is_ok()
    }
}

/// Generate a random ID
pub fn generate_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Hash data with SHA-256
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pin_validation() {
        assert!(KeyDerivation::validate_pin("123456"));
        assert!(KeyDerivation::validate_pin("000000"));
        assert!(!KeyDerivation::validate_pin("12345"));
        assert!(!KeyDerivation::validate_pin("1234567"));
        assert!(!KeyDerivation::validate_pin("12345a"));
        assert!(!KeyDerivation::validate_pin(""));
    }

    #[test]
    fn test_key_derivation() {
        let kd = KeyDerivation::new().unwrap();
        let key1 = kd.derive_db_key("123456").unwrap();
        let key2 = kd.derive_db_key("123456").unwrap();
        let key3 = kd.derive_db_key("654321").unwrap();

        // Same PIN should produce same key
        assert_eq!(key1, key2);
        // Different PIN should produce different key
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_hmac() {
        let key = b"test-key-for-hmac";
        let data = b"audit log entry data";

        let mac = hmac::generate(key, data);
        assert!(hmac::verify(key, data, &mac));

        // Tampered data should fail
        let tampered = b"tampered audit log";
        assert!(!hmac::verify(key, tampered, &mac));
    }

    #[test]
    fn test_generate_pin() {
        for _ in 0..100 {
            let pin = KeyDerivation::generate_pin().unwrap();
            assert!(KeyDerivation::validate_pin(&pin));
        }
    }
}
