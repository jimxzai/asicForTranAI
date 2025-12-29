//! Solana wallet implementation
//!
//! Solana uses Ed25519 for signing, with BIP44 derivation path:
//! m/44'/501'/account'/change'
//!
//! Key differences from EVM:
//! - Uses Ed25519 instead of secp256k1
//! - Addresses are base58-encoded public keys (32 bytes)
//! - No concept of gas - uses "compute units" and priority fees

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Sha512, Digest};
use zeroize::Zeroize;

use crate::error::{WalletError, Result};
use crate::hd::{ExtendedPrivateKey, DerivationPath};
use crate::Network;

/// Solana address
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolanaAddress {
    pub address: String,
    pub derivation_path: String,
    pub index: u32,
    pub created_at: DateTime<Utc>,
}

/// Solana wallet
pub struct SolanaWallet {
    /// Master extended key
    master_key: ExtendedPrivateKey,
    /// Network (mainnet or devnet)
    network: Network,
    /// Generated addresses
    addresses: Vec<SolanaAddress>,
    /// Next address index
    next_index: u32,
}

impl SolanaWallet {
    /// Create a new Solana wallet
    pub fn new(master_key: ExtendedPrivateKey, network: Network) -> Result<Self> {
        if !network.is_solana() {
            return Err(WalletError::InvalidNetwork(format!(
                "Expected Solana network, got {:?}",
                network
            )));
        }

        Ok(Self {
            master_key,
            network,
            addresses: Vec::new(),
            next_index: 0,
        })
    }

    /// Get the derivation path for an index
    /// Solana uses: m/44'/501'/account'/0' (hardened derivation)
    fn derivation_path(&self, index: u32) -> String {
        format!("m/44'/501'/{}'/0'", index)
    }

    /// Derive Ed25519 keypair from seed using SLIP-0010
    fn derive_ed25519_keypair(&self, index: u32) -> Result<Ed25519Keypair> {
        // Get the seed from master key
        let path = self.derivation_path(index);

        // For Solana, we use a simplified derivation
        // In production, use proper SLIP-0010 Ed25519 derivation
        let derived = self.master_key.derive_path(&DerivationPath::from_string(&path)?)?;

        // Use the private key bytes to seed Ed25519
        // This is a simplified version - real implementation would use proper SLIP-0010
        let seed_bytes = derived.secret_key_bytes();

        // Ed25519 seed is 32 bytes
        let mut ed_seed = [0u8; 32];
        ed_seed.copy_from_slice(&seed_bytes[..32]);

        // Generate Ed25519 keypair from seed
        // Using sha512 to expand the seed as per Ed25519 spec
        let mut hasher = Sha512::new();
        hasher.update(&ed_seed);
        let hash = hasher.finalize();

        let mut secret = [0u8; 32];
        secret.copy_from_slice(&hash[..32]);

        // Clamp the secret key as per Ed25519 spec
        secret[0] &= 248;
        secret[31] &= 127;
        secret[31] |= 64;

        // Derive public key (simplified - use proper ed25519 in production)
        // For now, we'll use a placeholder that shows the structure
        let public = derive_ed25519_public(&secret)?;

        ed_seed.zeroize();

        Ok(Ed25519Keypair { secret, public })
    }

    /// Generate a new address
    pub fn new_address(&mut self) -> Result<SolanaAddress> {
        let index = self.next_index;
        self.next_index += 1;

        let keypair = self.derive_ed25519_keypair(index)?;

        // Solana address is base58-encoded public key
        let address = bs58::encode(&keypair.public).into_string();

        let sol_address = SolanaAddress {
            address,
            derivation_path: self.derivation_path(index),
            index,
            created_at: Utc::now(),
        };

        self.addresses.push(sol_address.clone());
        Ok(sol_address)
    }

    /// Get address at specific index
    pub fn get_address(&self, index: u32) -> Result<SolanaAddress> {
        let keypair = self.derive_ed25519_keypair(index)?;
        let address = bs58::encode(&keypair.public).into_string();

        Ok(SolanaAddress {
            address,
            derivation_path: self.derivation_path(index),
            index,
            created_at: Utc::now(),
        })
    }

    /// Get all generated addresses
    pub fn get_all_addresses(&self) -> Vec<SolanaAddress> {
        self.addresses.clone()
    }

    /// Sign a Solana transaction
    pub fn sign_transaction(
        &self,
        tx: &SolanaTransaction,
        from_index: u32,
    ) -> Result<SignedSolanaTransaction> {
        let keypair = self.derive_ed25519_keypair(from_index)?;
        let from_address = bs58::encode(&keypair.public).into_string();

        // Sign the message using Ed25519
        let signature = sign_ed25519(&keypair.secret, &tx.message)?;
        let signature_b58 = bs58::encode(&signature).into_string();

        Ok(SignedSolanaTransaction {
            signature: signature_b58,
            from: from_address,
            to: tx.to.clone(),
            lamports: tx.lamports,
            recent_blockhash: tx.recent_blockhash.clone(),
        })
    }

    /// Sign an arbitrary message
    pub fn sign_message(&self, message: &[u8], from_index: u32) -> Result<Vec<u8>> {
        let keypair = self.derive_ed25519_keypair(from_index)?;
        sign_ed25519(&keypair.secret, message)
    }

    /// Get the network
    pub fn network(&self) -> Network {
        self.network
    }
}

/// Ed25519 keypair
struct Ed25519Keypair {
    secret: [u8; 32],
    public: [u8; 32],
}

impl Drop for Ed25519Keypair {
    fn drop(&mut self) {
        self.secret.zeroize();
    }
}

/// Derive Ed25519 public key from secret
/// This is a simplified implementation - use proper Ed25519 library in production
fn derive_ed25519_public(secret: &[u8; 32]) -> Result<[u8; 32]> {
    // In production, use a proper Ed25519 implementation like ed25519-dalek
    // For now, we'll use a deterministic derivation for structure
    let mut hasher = Sha512::new();
    hasher.update(b"ed25519_public:");
    hasher.update(secret);
    let hash = hasher.finalize();

    let mut public = [0u8; 32];
    public.copy_from_slice(&hash[..32]);
    Ok(public)
}

/// Sign message with Ed25519
/// This is a simplified implementation - use proper Ed25519 library in production
fn sign_ed25519(secret: &[u8; 32], message: &[u8]) -> Result<[u8; 64]> {
    // In production, use a proper Ed25519 implementation like ed25519-dalek
    let mut hasher = Sha512::new();
    hasher.update(secret);
    hasher.update(message);
    let hash = hasher.finalize();

    let mut signature = [0u8; 64];
    signature.copy_from_slice(&hash[..64]);
    Ok(signature)
}

/// Solana transaction (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolanaTransaction {
    /// Recipient address (base58)
    pub to: String,
    /// Amount in lamports (1 SOL = 1e9 lamports)
    pub lamports: u64,
    /// Recent blockhash for transaction validity
    pub recent_blockhash: String,
    /// Optional memo
    pub memo: Option<String>,
}

/// Signed Solana transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedSolanaTransaction {
    pub signature: String,
    pub from: String,
    pub to: String,
    pub lamports: u64,
    pub recent_blockhash: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mnemonic;

    fn test_wallet() -> SolanaWallet {
        let mnemonic = Mnemonic::from_phrase(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        ).unwrap();
        let seed = mnemonic.to_seed_no_passphrase();
        let master = ExtendedPrivateKey::from_seed(&seed, Network::Solana).unwrap();
        SolanaWallet::new(master, Network::Solana).unwrap()
    }

    #[test]
    fn test_new_address() {
        let mut wallet = test_wallet();
        let addr = wallet.new_address().unwrap();

        // Solana addresses are base58-encoded (32-44 characters typically)
        assert!(!addr.address.is_empty());
        assert!(addr.derivation_path.contains("501")); // Solana coin type
    }

    #[test]
    fn test_deterministic_addresses() {
        let wallet1 = test_wallet();
        let wallet2 = test_wallet();

        let addr1 = wallet1.get_address(0).unwrap();
        let addr2 = wallet2.get_address(0).unwrap();

        assert_eq!(addr1.address, addr2.address);
    }

    #[test]
    fn test_sign_transaction() {
        let wallet = test_wallet();

        let tx = SolanaTransaction {
            to: "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM".to_string(),
            lamports: 1_000_000_000, // 1 SOL
            recent_blockhash: "EkSnNWid2cvwEVnVx9aBqawnmiCNiDgp3gUdkDPTKN1N".to_string(),
            memo: Some("Test transfer".to_string()),
        };

        let signed = wallet.sign_transaction(&tx, 0).unwrap();
        assert!(!signed.signature.is_empty());
        assert_eq!(signed.to, tx.to);
        assert_eq!(signed.lamports, tx.lamports);
    }
}
