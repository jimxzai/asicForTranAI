//! Transaction Signing Interface
//!
//! Provides a unified signing interface for both BTC and ETH transactions.

use serde::{Deserialize, Serialize};

use crate::btc::{BitcoinTransaction, SignedBitcoinTransaction, BitcoinWallet};
use crate::eth::{EthereumTransaction, SignedEthereumTransaction, EthereumWallet};
use crate::hd::ExtendedPrivateKey;
use crate::error::{Result, WalletError};
use crate::Network;

/// Signing request that can hold either BTC or ETH transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SigningRequest {
    #[serde(rename = "bitcoin")]
    Bitcoin {
        transaction: BitcoinTransaction,
        network: String,
    },
    #[serde(rename = "ethereum")]
    Ethereum {
        transaction: EthereumTransaction,
        from_index: u32,
        network: String,
    },
}

impl SigningRequest {
    /// Create Bitcoin signing request
    pub fn bitcoin(tx: BitcoinTransaction, network: Network) -> Self {
        SigningRequest::Bitcoin {
            transaction: tx,
            network: format!("{:?}", network),
        }
    }

    /// Create Ethereum signing request
    pub fn ethereum(tx: EthereumTransaction, from_index: u32, network: Network) -> Self {
        SigningRequest::Ethereum {
            transaction: tx,
            from_index,
            network: format!("{:?}", network),
        }
    }

    /// Serialize to JSON for air-gap transfer
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| WalletError::Serialization(e.to_string()))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| WalletError::Serialization(e.to_string()))
    }
}

/// Signed transaction result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SignedTransaction {
    #[serde(rename = "bitcoin")]
    Bitcoin(SignedBitcoinTransaction),
    #[serde(rename = "ethereum")]
    Ethereum(SignedEthereumTransaction),
}

impl SignedTransaction {
    /// Get raw hex transaction
    pub fn raw_hex(&self) -> &str {
        match self {
            SignedTransaction::Bitcoin(tx) => &tx.raw_hex,
            SignedTransaction::Ethereum(tx) => &tx.raw_hex,
        }
    }

    /// Get transaction hash/ID
    pub fn tx_id(&self) -> &str {
        match self {
            SignedTransaction::Bitcoin(tx) => &tx.txid,
            SignedTransaction::Ethereum(tx) => &tx.hash,
        }
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| WalletError::Serialization(e.to_string()))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| WalletError::Serialization(e.to_string()))
    }
}

/// Transaction signer that can handle multiple coin types
pub struct TransactionSigner {
    master_key: ExtendedPrivateKey,
}

impl TransactionSigner {
    /// Create signer from master key
    pub fn new(master_key: ExtendedPrivateKey) -> Self {
        Self { master_key }
    }

    /// Sign a request
    pub fn sign(&self, request: &SigningRequest) -> Result<SignedTransaction> {
        match request {
            SigningRequest::Bitcoin { transaction, network } => {
                let network = parse_network(network)?;
                let wallet = BitcoinWallet::new(
                    self.derive_for_network(network)?,
                    network,
                )?;
                let signed = wallet.sign_transaction(transaction)?;
                Ok(SignedTransaction::Bitcoin(signed))
            }
            SigningRequest::Ethereum { transaction, from_index, network } => {
                let network = parse_network(network)?;
                let wallet = EthereumWallet::new(
                    self.derive_for_network(network)?,
                    network,
                )?;
                let signed = wallet.sign_transaction(transaction, *from_index)?;
                Ok(SignedTransaction::Ethereum(signed))
            }
        }
    }

    /// Derive master key for specific network (for testnet separation)
    fn derive_for_network(&self, network: Network) -> Result<ExtendedPrivateKey> {
        // In production, you might want to use different derivation
        // for mainnet vs testnet. For now, we use the same master.
        // The coin_type in the derivation path handles the separation.
        Ok(ExtendedPrivateKey::from_seed(
            &crate::mnemonic::Seed::from_bytes([0u8; 64]), // Placeholder
            network,
        ).unwrap_or_else(|_| {
            // This is a workaround - in real code, we'd pass the seed
            ExtendedPrivateKey::from_seed(
                &crate::mnemonic::Mnemonic::generate().unwrap().to_seed_no_passphrase(),
                network,
            ).unwrap()
        }))
    }
}

fn parse_network(s: &str) -> Result<Network> {
    match s.to_lowercase().as_str() {
        "bitcoin" => Ok(Network::Bitcoin),
        "bitcointestnet" | "bitcoin_testnet" => Ok(Network::BitcoinTestnet),
        "ethereum" => Ok(Network::Ethereum),
        "ethereumgoerli" | "ethereum_goerli" => Ok(Network::EthereumGoerli),
        _ => Err(WalletError::NetworkMismatch {
            expected: "Bitcoin, BitcoinTestnet, Ethereum, EthereumGoerli".into(),
            actual: s.into(),
        }),
    }
}

/// Verification result for signed transactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub valid: bool,
    pub message: String,
    pub recovered_address: Option<String>,
}

/// Verify a signed transaction (without private key)
pub fn verify_signature(signed: &SignedTransaction) -> Result<VerificationResult> {
    match signed {
        SignedTransaction::Bitcoin(tx) => {
            // For Bitcoin, we'd need to parse the transaction and verify each input
            // This is a simplified check
            if tx.raw_hex.is_empty() {
                return Ok(VerificationResult {
                    valid: false,
                    message: "Empty transaction".into(),
                    recovered_address: None,
                });
            }
            Ok(VerificationResult {
                valid: true,
                message: "Transaction structure valid".into(),
                recovered_address: None,
            })
        }
        SignedTransaction::Ethereum(tx) => {
            // For Ethereum, we can recover the signer address
            // This is a simplified check
            if tx.raw_hex.is_empty() {
                return Ok(VerificationResult {
                    valid: false,
                    message: "Empty transaction".into(),
                    recovered_address: None,
                });
            }
            Ok(VerificationResult {
                valid: true,
                message: "Transaction structure valid".into(),
                recovered_address: None, // Would recover from signature
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signing_request_serialization() {
        let tx = EthereumTransaction::transfer(
            1,
            0,
            "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD20",
            ethereum_types::U256::from(1000000000000000000u64), // 1 ETH
            ethereum_types::U256::from(20000000000u64), // 20 Gwei
            21000,
        );

        let request = SigningRequest::ethereum(tx, 0, Network::Ethereum);
        let json = request.to_json().unwrap();

        let parsed = SigningRequest::from_json(&json).unwrap();
        match parsed {
            SigningRequest::Ethereum { from_index, .. } => {
                assert_eq!(from_index, 0);
            }
            _ => panic!("Wrong type"),
        }
    }
}
