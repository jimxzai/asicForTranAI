//! Generic EVM L2 wallet support
//!
//! This module provides wallet functionality for EVM-compatible L2 chains:
//! - Polygon (MATIC)
//! - Arbitrum One
//! - Base
//! - Optimism
//!
//! All EVM L2s use the same address derivation as Ethereum (BIP44 coin type 60)
//! and the same signing algorithm (secp256k1 + keccak256).

use serde::{Deserialize, Serialize};

use crate::eth::{EthereumWallet, EthereumAddress, EthereumTransaction, SignedEthTransaction};
use crate::hd::ExtendedPrivateKey;
use crate::error::{WalletError, Result};
use crate::Network;

/// EVM chain configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvmChain {
    pub network: Network,
    pub chain_id: u64,
    pub name: &'static str,
    pub symbol: &'static str,
    pub explorer_url: &'static str,
    pub rpc_url: &'static str,
}

impl EvmChain {
    /// Ethereum Mainnet
    pub const ETHEREUM: EvmChain = EvmChain {
        network: Network::Ethereum,
        chain_id: 1,
        name: "Ethereum",
        symbol: "ETH",
        explorer_url: "https://etherscan.io",
        rpc_url: "https://eth.llamarpc.com",
    };

    /// Polygon Mainnet
    pub const POLYGON: EvmChain = EvmChain {
        network: Network::Polygon,
        chain_id: 137,
        name: "Polygon",
        symbol: "MATIC",
        explorer_url: "https://polygonscan.com",
        rpc_url: "https://polygon-rpc.com",
    };

    /// Arbitrum One
    pub const ARBITRUM: EvmChain = EvmChain {
        network: Network::Arbitrum,
        chain_id: 42161,
        name: "Arbitrum One",
        symbol: "ETH",
        explorer_url: "https://arbiscan.io",
        rpc_url: "https://arb1.arbitrum.io/rpc",
    };

    /// Base Mainnet
    pub const BASE: EvmChain = EvmChain {
        network: Network::Base,
        chain_id: 8453,
        name: "Base",
        symbol: "ETH",
        explorer_url: "https://basescan.org",
        rpc_url: "https://mainnet.base.org",
    };

    /// Optimism Mainnet
    pub const OPTIMISM: EvmChain = EvmChain {
        network: Network::Optimism,
        chain_id: 10,
        name: "Optimism",
        symbol: "ETH",
        explorer_url: "https://optimistic.etherscan.io",
        rpc_url: "https://mainnet.optimism.io",
    };

    /// Get chain by network
    pub fn from_network(network: Network) -> Option<Self> {
        match network {
            Network::Ethereum => Some(Self::ETHEREUM),
            Network::Polygon => Some(Self::POLYGON),
            Network::Arbitrum => Some(Self::ARBITRUM),
            Network::Base => Some(Self::BASE),
            Network::Optimism => Some(Self::OPTIMISM),
            _ => None,
        }
    }

    /// Get all supported mainnet chains
    pub fn all_mainnets() -> Vec<Self> {
        vec![
            Self::ETHEREUM,
            Self::POLYGON,
            Self::ARBITRUM,
            Self::BASE,
            Self::OPTIMISM,
        ]
    }
}

/// Generic EVM wallet that can work with any EVM-compatible chain
pub struct EvmWallet {
    /// The underlying Ethereum wallet (same key derivation)
    inner: EthereumWallet,
    /// Active chain configuration
    chain: EvmChain,
}

impl EvmWallet {
    /// Create a new EVM wallet for a specific chain
    pub fn new(master_key: ExtendedPrivateKey, chain: EvmChain) -> Result<Self> {
        // Use the same derivation as Ethereum since all EVM chains share coin type 60
        let inner = EthereumWallet::new(master_key, Network::Ethereum)?;
        Ok(Self { inner, chain })
    }

    /// Get the active chain
    pub fn chain(&self) -> &EvmChain {
        &self.chain
    }

    /// Switch to a different chain (same addresses, different chain ID)
    pub fn switch_chain(&mut self, chain: EvmChain) {
        self.chain = chain;
    }

    /// Generate a new address (same address works on all EVM chains)
    pub fn new_address(&mut self) -> Result<EthereumAddress> {
        self.inner.new_address()
    }

    /// Get address at index
    pub fn get_address(&self, index: u32) -> Result<EthereumAddress> {
        self.inner.get_address(index)
    }

    /// Get all generated addresses
    pub fn get_all_addresses(&self) -> Vec<EthereumAddress> {
        self.inner.get_all_addresses()
    }

    /// Sign a transaction for the active chain
    pub fn sign_transaction(
        &self,
        tx: &EvmTransaction,
        from_index: u32,
    ) -> Result<SignedEvmTransaction> {
        // Override chain ID with active chain
        let eth_tx = EthereumTransaction {
            to: tx.to.clone(),
            value: tx.value.clone(),
            data: tx.data.clone(),
            nonce: tx.nonce,
            gas_limit: tx.gas_limit,
            max_fee_per_gas: tx.max_fee_per_gas,
            max_priority_fee_per_gas: tx.max_priority_fee_per_gas,
            chain_id: self.chain.chain_id,
        };

        let signed = self.inner.sign_transaction(&eth_tx, from_index)?;

        Ok(SignedEvmTransaction {
            chain: self.chain,
            raw_tx: signed.raw_tx,
            tx_hash: signed.tx_hash,
            from: signed.from,
            to: tx.to.clone(),
            value: tx.value.clone(),
        })
    }

    /// Get the master fingerprint
    pub fn fingerprint(&self) -> [u8; 4] {
        self.inner.fingerprint()
    }
}

/// EVM transaction (same structure as Ethereum)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmTransaction {
    pub to: String,
    pub value: String,
    #[serde(default)]
    pub data: Vec<u8>,
    pub nonce: u64,
    pub gas_limit: u64,
    pub max_fee_per_gas: u64,
    pub max_priority_fee_per_gas: u64,
}

/// Signed EVM transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedEvmTransaction {
    pub chain: EvmChainInfo,
    pub raw_tx: String,
    pub tx_hash: String,
    pub from: String,
    pub to: String,
    pub value: String,
}

/// Chain info for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmChainInfo {
    pub chain_id: u64,
    pub name: String,
    pub symbol: String,
}

impl From<EvmChain> for EvmChainInfo {
    fn from(chain: EvmChain) -> Self {
        Self {
            chain_id: chain.chain_id,
            name: chain.name.to_string(),
            symbol: chain.symbol.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mnemonic;

    fn test_wallet(chain: EvmChain) -> EvmWallet {
        let mnemonic = Mnemonic::from_phrase(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        ).unwrap();
        let seed = mnemonic.to_seed_no_passphrase();
        let master = ExtendedPrivateKey::from_seed(&seed, Network::Ethereum).unwrap();
        EvmWallet::new(master, chain).unwrap()
    }

    #[test]
    fn test_polygon_wallet() {
        let mut wallet = test_wallet(EvmChain::POLYGON);
        let addr = wallet.new_address().unwrap();

        // Same address as Ethereum (both use coin type 60)
        assert!(addr.address.starts_with("0x"));
        assert_eq!(wallet.chain().chain_id, 137);
    }

    #[test]
    fn test_chain_switching() {
        let mut wallet = test_wallet(EvmChain::ETHEREUM);
        assert_eq!(wallet.chain().chain_id, 1);

        wallet.switch_chain(EvmChain::ARBITRUM);
        assert_eq!(wallet.chain().chain_id, 42161);

        // Addresses remain the same
        let addr = wallet.get_address(0).unwrap();
        assert!(addr.address.starts_with("0x"));
    }

    #[test]
    fn test_all_chains_same_address() {
        let mnemonic = Mnemonic::from_phrase(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        ).unwrap();
        let seed = mnemonic.to_seed_no_passphrase();

        let mut addresses = Vec::new();

        for chain in EvmChain::all_mainnets() {
            let master = ExtendedPrivateKey::from_seed(&seed, Network::Ethereum).unwrap();
            let mut wallet = EvmWallet::new(master, chain).unwrap();
            let addr = wallet.new_address().unwrap();
            addresses.push(addr.address.clone());
        }

        // All addresses should be identical
        let first = &addresses[0];
        for addr in &addresses[1..] {
            assert_eq!(first, addr);
        }
    }
}
