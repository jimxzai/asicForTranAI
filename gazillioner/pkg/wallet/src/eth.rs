//! Ethereum Wallet - EIP-155 Transaction Signing
//!
//! Implements Ethereum address generation and transaction signing
//! with EIP-155 replay protection.

use ethereum_types::{H160, H256, U256};
use keccak_hash::keccak;
use rlp::RlpStream;
use secp256k1::{Secp256k1, Message, SecretKey, ecdsa::RecoverableSignature, ecdsa::RecoveryId};
use serde::{Deserialize, Serialize};

use crate::error::{Result, WalletError};
use crate::hd::{ExtendedPrivateKey, DerivationPath, KeyPair};
use crate::Network;

/// Ethereum address (checksum format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthereumAddress {
    pub address: String,
    pub derivation_path: String,
    pub index: u32,
}

impl EthereumAddress {
    /// Create from public key
    pub fn from_public_key(
        pubkey: &secp256k1::PublicKey,
        path: &DerivationPath,
        index: u32,
    ) -> Result<Self> {
        // Get uncompressed public key (65 bytes), skip first byte (0x04 prefix)
        let pubkey_bytes = pubkey.serialize_uncompressed();
        let pubkey_data = &pubkey_bytes[1..]; // Skip 0x04 prefix

        // Keccak-256 hash
        let hash = keccak(pubkey_data);

        // Take last 20 bytes as address
        let address_bytes: [u8; 20] = hash.as_bytes()[12..].try_into().unwrap();

        // Convert to checksum address (EIP-55)
        let address = Self::to_checksum_address(&address_bytes);

        Ok(Self {
            address,
            derivation_path: path.to_string(),
            index,
        })
    }

    /// Convert to EIP-55 checksum address
    fn to_checksum_address(addr: &[u8; 20]) -> String {
        let hex_addr = hex::encode(addr);
        let hash = keccak(hex_addr.as_bytes());
        let hash_hex = hex::encode(hash.as_bytes());

        let mut result = String::with_capacity(42);
        result.push_str("0x");

        for (i, c) in hex_addr.chars().enumerate() {
            let hash_char = hash_hex.chars().nth(i).unwrap();
            if hash_char >= '8' {
                result.push(c.to_ascii_uppercase());
            } else {
                result.push(c);
            }
        }

        result
    }

    /// Parse address string
    pub fn parse(address: &str) -> Result<H160> {
        let addr = address.strip_prefix("0x").unwrap_or(address);
        if addr.len() != 40 {
            return Err(WalletError::InvalidAddress("Invalid length".into()));
        }

        let bytes = hex::decode(addr)
            .map_err(|e| WalletError::InvalidAddress(format!("Invalid hex: {}", e)))?;

        Ok(H160::from_slice(&bytes))
    }

    /// Validate checksum (EIP-55)
    pub fn validate_checksum(address: &str) -> bool {
        let addr = address.strip_prefix("0x").unwrap_or(address);
        if addr.len() != 40 {
            return false;
        }

        let lowercase = addr.to_lowercase();
        let hash = keccak(lowercase.as_bytes());
        let hash_hex = hex::encode(hash.as_bytes());

        for (i, (c, h)) in addr.chars().zip(hash_hex.chars()).enumerate() {
            if c.is_alphabetic() {
                let should_upper = h >= '8';
                let is_upper = c.is_uppercase();
                if should_upper != is_upper {
                    return false;
                }
            }
        }

        true
    }
}

/// Ethereum transaction types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TxType {
    /// Legacy transaction (pre-EIP-1559)
    Legacy,
    /// EIP-2930 access list transaction
    AccessList,
    /// EIP-1559 fee market transaction
    EIP1559,
}

/// Unsigned Ethereum transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthereumTransaction {
    /// Transaction type
    pub tx_type: TxType,
    /// Chain ID (EIP-155)
    pub chain_id: u64,
    /// Transaction nonce
    pub nonce: u64,
    /// Gas price (for Legacy/AccessList) in wei
    pub gas_price: Option<U256>,
    /// Max fee per gas (for EIP-1559) in wei
    pub max_fee_per_gas: Option<U256>,
    /// Max priority fee per gas (for EIP-1559) in wei
    pub max_priority_fee_per_gas: Option<U256>,
    /// Gas limit
    pub gas_limit: u64,
    /// Recipient address
    pub to: String,
    /// Value in wei
    pub value: U256,
    /// Transaction data
    pub data: Vec<u8>,
}

impl EthereumTransaction {
    /// Create a simple ETH transfer (Legacy)
    pub fn transfer(
        chain_id: u64,
        nonce: u64,
        to: &str,
        value_wei: U256,
        gas_price: U256,
        gas_limit: u64,
    ) -> Self {
        Self {
            tx_type: TxType::Legacy,
            chain_id,
            nonce,
            gas_price: Some(gas_price),
            max_fee_per_gas: None,
            max_priority_fee_per_gas: None,
            gas_limit,
            to: to.to_string(),
            value: value_wei,
            data: vec![],
        }
    }

    /// Create an EIP-1559 transfer
    pub fn transfer_eip1559(
        chain_id: u64,
        nonce: u64,
        to: &str,
        value_wei: U256,
        max_fee_per_gas: U256,
        max_priority_fee_per_gas: U256,
        gas_limit: u64,
    ) -> Self {
        Self {
            tx_type: TxType::EIP1559,
            chain_id,
            nonce,
            gas_price: None,
            max_fee_per_gas: Some(max_fee_per_gas),
            max_priority_fee_per_gas: Some(max_priority_fee_per_gas),
            gas_limit,
            to: to.to_string(),
            value: value_wei,
            data: vec![],
        }
    }

    /// Encode for signing (returns hash to sign)
    fn encode_for_signing(&self) -> Result<H256> {
        match self.tx_type {
            TxType::Legacy => self.encode_legacy_for_signing(),
            TxType::EIP1559 => self.encode_eip1559_for_signing(),
            _ => Err(WalletError::InvalidTransaction("Unsupported tx type".into())),
        }
    }

    fn encode_legacy_for_signing(&self) -> Result<H256> {
        let to = EthereumAddress::parse(&self.to)?;
        let gas_price = self.gas_price.ok_or_else(||
            WalletError::InvalidTransaction("Missing gas_price for legacy tx".into())
        )?;

        let mut stream = RlpStream::new_list(9);
        stream.append(&self.nonce);
        stream.append(&gas_price);
        stream.append(&self.gas_limit);
        stream.append(&to);
        stream.append(&self.value);
        stream.append(&self.data);
        stream.append(&self.chain_id);
        stream.append(&0u8);
        stream.append(&0u8);

        Ok(keccak(stream.out()))
    }

    fn encode_eip1559_for_signing(&self) -> Result<H256> {
        let to = EthereumAddress::parse(&self.to)?;
        let max_fee = self.max_fee_per_gas.ok_or_else(||
            WalletError::InvalidTransaction("Missing max_fee_per_gas".into())
        )?;
        let max_priority = self.max_priority_fee_per_gas.ok_or_else(||
            WalletError::InvalidTransaction("Missing max_priority_fee_per_gas".into())
        )?;

        let mut stream = RlpStream::new_list(9);
        stream.append(&self.chain_id);
        stream.append(&self.nonce);
        stream.append(&max_priority);
        stream.append(&max_fee);
        stream.append(&self.gas_limit);
        stream.append(&to);
        stream.append(&self.value);
        stream.append(&self.data);
        stream.append_list::<Vec<u8>, Vec<u8>>(&vec![]); // access list

        let mut prefixed = vec![0x02u8];
        prefixed.extend(stream.out());

        Ok(keccak(&prefixed))
    }
}

/// Signed Ethereum transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedEthereumTransaction {
    pub raw_hex: String,
    pub hash: String,
}

/// Ethereum wallet for managing addresses and signing
pub struct EthereumWallet {
    master_key: ExtendedPrivateKey,
    network: Network,
    next_index: u32,
}

impl EthereumWallet {
    /// Create wallet from master extended private key
    pub fn new(master_key: ExtendedPrivateKey, network: Network) -> Result<Self> {
        if !matches!(network, Network::Ethereum | Network::EthereumGoerli) {
            return Err(WalletError::NetworkMismatch {
                expected: "Ethereum or EthereumGoerli".into(),
                actual: format!("{:?}", network),
            });
        }

        Ok(Self {
            master_key,
            network,
            next_index: 0,
        })
    }

    /// Generate new address
    pub fn new_address(&mut self) -> Result<EthereumAddress> {
        let index = self.next_index;
        self.next_index += 1;
        self.get_address(index)
    }

    /// Get address at specific index
    pub fn get_address(&self, index: u32) -> Result<EthereumAddress> {
        let path = DerivationPath::ethereum(0, index);
        let keypair = KeyPair::from_xpriv(&self.master_key, &path)?;

        EthereumAddress::from_public_key(
            keypair.public_key(),
            &path,
            index,
        )
    }

    /// Get chain ID for current network
    pub fn chain_id(&self) -> u64 {
        match self.network {
            Network::Ethereum => 1,
            Network::EthereumGoerli => 5,
            _ => 1,
        }
    }

    /// Sign a transaction
    pub fn sign_transaction(
        &self,
        tx: &EthereumTransaction,
        from_index: u32,
    ) -> Result<SignedEthereumTransaction> {
        let secp = Secp256k1::new();

        // Derive key for signing
        let path = DerivationPath::ethereum(0, from_index);
        let keypair = KeyPair::from_xpriv(&self.master_key, &path)?;

        // Get hash to sign
        let hash = tx.encode_for_signing()?;
        let msg = Message::from_digest(hash.0);

        // Sign with recovery (needed for Ethereum)
        let (rec_id, sig) = secp
            .sign_ecdsa_recoverable(&msg, keypair.secret_key())
            .serialize_compact();

        // Calculate v value based on transaction type
        let v = match tx.tx_type {
            TxType::Legacy => {
                // EIP-155: v = chain_id * 2 + 35 + recovery_id
                tx.chain_id * 2 + 35 + rec_id.to_i32() as u64
            }
            TxType::EIP1559 => {
                // EIP-1559: v = recovery_id (0 or 1)
                rec_id.to_i32() as u64
            }
            _ => return Err(WalletError::InvalidTransaction("Unsupported tx type".into())),
        };

        // Extract r and s from signature
        let r = U256::from_big_endian(&sig[0..32]);
        let s = U256::from_big_endian(&sig[32..64]);

        // Encode signed transaction
        let raw_hex = self.encode_signed_transaction(tx, v, r, s)?;
        let tx_hash = keccak(&hex::decode(&raw_hex).unwrap());

        Ok(SignedEthereumTransaction {
            raw_hex: format!("0x{}", raw_hex),
            hash: format!("0x{}", hex::encode(tx_hash.as_bytes())),
        })
    }

    fn encode_signed_transaction(
        &self,
        tx: &EthereumTransaction,
        v: u64,
        r: U256,
        s: U256,
    ) -> Result<String> {
        let to = EthereumAddress::parse(&tx.to)?;

        match tx.tx_type {
            TxType::Legacy => {
                let gas_price = tx.gas_price.unwrap();

                let mut stream = RlpStream::new_list(9);
                stream.append(&tx.nonce);
                stream.append(&gas_price);
                stream.append(&tx.gas_limit);
                stream.append(&to);
                stream.append(&tx.value);
                stream.append(&tx.data);
                stream.append(&v);
                stream.append(&r);
                stream.append(&s);

                Ok(hex::encode(stream.out()))
            }
            TxType::EIP1559 => {
                let max_fee = tx.max_fee_per_gas.unwrap();
                let max_priority = tx.max_priority_fee_per_gas.unwrap();

                let mut stream = RlpStream::new_list(12);
                stream.append(&tx.chain_id);
                stream.append(&tx.nonce);
                stream.append(&max_priority);
                stream.append(&max_fee);
                stream.append(&tx.gas_limit);
                stream.append(&to);
                stream.append(&tx.value);
                stream.append(&tx.data);
                stream.append_list::<Vec<u8>, Vec<u8>>(&vec![]); // access list
                stream.append(&v);
                stream.append(&r);
                stream.append(&s);

                let mut result = vec![0x02u8];
                result.extend(stream.out());
                Ok(hex::encode(result))
            }
            _ => Err(WalletError::InvalidTransaction("Unsupported tx type".into())),
        }
    }

    /// Get network
    pub fn network(&self) -> Network {
        self.network
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnemonic::Mnemonic;

    fn test_wallet() -> EthereumWallet {
        let mnemonic = Mnemonic::from_phrase(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        ).unwrap();
        let seed = mnemonic.to_seed_no_passphrase();
        let master = ExtendedPrivateKey::from_seed(&seed, Network::Ethereum).unwrap();
        EthereumWallet::new(master, Network::Ethereum).unwrap()
    }

    #[test]
    fn test_generate_address() {
        let mut wallet = test_wallet();
        let addr = wallet.new_address().unwrap();

        // Should be a 0x address
        assert!(addr.address.starts_with("0x"));
        assert_eq!(addr.address.len(), 42);
    }

    #[test]
    fn test_address_derivation_consistency() {
        let wallet = test_wallet();

        let addr1 = wallet.get_address(0).unwrap();
        let addr2 = wallet.get_address(0).unwrap();

        // Same index should produce same address
        assert_eq!(addr1.address, addr2.address);
    }

    #[test]
    fn test_checksum_validation() {
        // Valid checksum
        assert!(EthereumAddress::validate_checksum(
            "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
        ));

        // Invalid checksum (wrong case)
        assert!(!EthereumAddress::validate_checksum(
            "0x5AAEB6053F3E94C9B9A09F33669435E7EF1BEAED"
        ));
    }

    #[test]
    fn test_well_known_address() {
        // Test vector from BIP44 spec
        let mnemonic = Mnemonic::from_phrase(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        ).unwrap();
        let seed = mnemonic.to_seed_no_passphrase();
        let master = ExtendedPrivateKey::from_seed(&seed, Network::Ethereum).unwrap();
        let wallet = EthereumWallet::new(master, Network::Ethereum).unwrap();

        let addr = wallet.get_address(0).unwrap();
        // This is the expected address for this mnemonic
        assert_eq!(addr.address.to_lowercase(), "0x9858effd232b4033e47d90003d41ec34ecaeda94");
    }
}
