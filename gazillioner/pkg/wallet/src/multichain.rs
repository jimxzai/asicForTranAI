//! Multi-chain wallet interface
//!
//! Provides a unified interface for managing wallets across multiple chains:
//! - Bitcoin (BTC)
//! - Ethereum (ETH) and EVM L2s
//! - Solana (SOL)

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::btc::{BitcoinWallet, BitcoinAddress, BitcoinTransaction};
use crate::eth::{EthereumWallet, EthereumAddress, EthereumTransaction};
use crate::evm::{EvmWallet, EvmChain, EvmTransaction};
use crate::sol::{SolanaWallet, SolanaAddress, SolanaTransaction};
use crate::hd::ExtendedPrivateKey;
use crate::mnemonic::Mnemonic;
use crate::error::{WalletError, Result};
use crate::Network;

/// Unified chain address
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainAddress {
    pub network: String,
    pub address: String,
    pub derivation_path: String,
    pub index: u32,
    pub created_at: DateTime<Utc>,
}

impl From<BitcoinAddress> for ChainAddress {
    fn from(addr: BitcoinAddress) -> Self {
        Self {
            network: "Bitcoin".to_string(),
            address: addr.address,
            derivation_path: addr.derivation_path,
            index: addr.index,
            created_at: Utc::now(),
        }
    }
}

impl From<EthereumAddress> for ChainAddress {
    fn from(addr: EthereumAddress) -> Self {
        Self {
            network: "Ethereum".to_string(),
            address: addr.address,
            derivation_path: addr.derivation_path,
            index: addr.index,
            created_at: Utc::now(),
        }
    }
}

impl From<SolanaAddress> for ChainAddress {
    fn from(addr: SolanaAddress) -> Self {
        Self {
            network: "Solana".to_string(),
            address: addr.address,
            derivation_path: addr.derivation_path,
            index: addr.index,
            created_at: addr.created_at,
        }
    }
}

/// Chain balance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainBalance {
    pub network: Network,
    pub symbol: String,
    pub balance: String,
    pub usd_value: Option<f64>,
}

/// Multi-chain wallet supporting BTC, ETH, EVM L2s, and Solana
pub struct MultiChainWallet {
    /// Bitcoin wallet
    btc: Option<BitcoinWallet>,
    /// Ethereum wallet
    eth: Option<EthereumWallet>,
    /// EVM L2 wallets (keyed by chain ID)
    evm_l2s: HashMap<u64, EvmWallet>,
    /// Solana wallet
    sol: Option<SolanaWallet>,
    /// Master fingerprint
    fingerprint: [u8; 4],
    /// Whether the wallet is initialized
    initialized: bool,
}

impl MultiChainWallet {
    /// Create a new multi-chain wallet from mnemonic
    pub fn from_mnemonic(mnemonic: &Mnemonic, passphrase: &str) -> Result<Self> {
        let seed = mnemonic.to_seed(passphrase);

        // Create master keys for each chain family
        let btc_master = ExtendedPrivateKey::from_seed(&seed, Network::Bitcoin)?;
        let eth_master = ExtendedPrivateKey::from_seed(&seed, Network::Ethereum)?;
        let sol_master = ExtendedPrivateKey::from_seed(&seed, Network::Solana)?;

        // Get fingerprint from BTC master
        let fingerprint = btc_master.fingerprint();

        // Create wallets
        let btc = BitcoinWallet::new(btc_master, Network::Bitcoin)?;
        let eth = EthereumWallet::new(eth_master.clone(), Network::Ethereum)?;
        let sol = SolanaWallet::new(sol_master, Network::Solana)?;

        // Create EVM L2 wallets
        let mut evm_l2s = HashMap::new();
        for chain in [EvmChain::POLYGON, EvmChain::ARBITRUM, EvmChain::BASE, EvmChain::OPTIMISM] {
            let l2_master = ExtendedPrivateKey::from_seed(&seed, Network::Ethereum)?;
            let l2_wallet = EvmWallet::new(l2_master, chain)?;
            evm_l2s.insert(chain.chain_id, l2_wallet);
        }

        Ok(Self {
            btc: Some(btc),
            eth: Some(eth),
            evm_l2s,
            sol: Some(sol),
            fingerprint,
            initialized: true,
        })
    }

    /// Get wallet fingerprint
    pub fn fingerprint(&self) -> [u8; 4] {
        self.fingerprint
    }

    /// Check if wallet is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    // ========================================================================
    // Bitcoin operations
    // ========================================================================

    /// Get Bitcoin wallet
    pub fn btc(&self) -> Result<&BitcoinWallet> {
        self.btc.as_ref().ok_or(WalletError::NotInitialized)
    }

    /// Get mutable Bitcoin wallet
    pub fn btc_mut(&mut self) -> Result<&mut BitcoinWallet> {
        self.btc.as_mut().ok_or(WalletError::NotInitialized)
    }

    /// Generate new BTC receive address
    pub fn btc_new_address(&mut self) -> Result<ChainAddress> {
        let addr = self.btc_mut()?.new_receive_address()?;
        Ok(addr.into())
    }

    /// Sign BTC transaction
    pub fn btc_sign(&self, tx: &BitcoinTransaction) -> Result<crate::btc::SignedBitcoinTransaction> {
        self.btc()?.sign_transaction(tx)
    }

    // ========================================================================
    // Ethereum operations
    // ========================================================================

    /// Get Ethereum wallet
    pub fn eth(&self) -> Result<&EthereumWallet> {
        self.eth.as_ref().ok_or(WalletError::NotInitialized)
    }

    /// Get mutable Ethereum wallet
    pub fn eth_mut(&mut self) -> Result<&mut EthereumWallet> {
        self.eth.as_mut().ok_or(WalletError::NotInitialized)
    }

    /// Generate new ETH address
    pub fn eth_new_address(&mut self) -> Result<ChainAddress> {
        let addr = self.eth_mut()?.new_address()?;
        Ok(addr.into())
    }

    /// Sign ETH transaction
    pub fn eth_sign(
        &self,
        tx: &EthereumTransaction,
        from_index: u32,
    ) -> Result<crate::eth::SignedEthereumTransaction> {
        self.eth()?.sign_transaction(tx, from_index)
    }

    // ========================================================================
    // EVM L2 operations
    // ========================================================================

    /// Get EVM L2 wallet by chain ID
    pub fn evm_l2(&self, chain_id: u64) -> Result<&EvmWallet> {
        self.evm_l2s
            .get(&chain_id)
            .ok_or(WalletError::InvalidNetwork(format!("Chain ID {} not supported", chain_id)))
    }

    /// Get mutable EVM L2 wallet
    pub fn evm_l2_mut(&mut self, chain_id: u64) -> Result<&mut EvmWallet> {
        self.evm_l2s
            .get_mut(&chain_id)
            .ok_or(WalletError::InvalidNetwork(format!("Chain ID {} not supported", chain_id)))
    }

    /// Generate new address for EVM L2 (same as ETH address)
    pub fn evm_l2_new_address(&mut self, chain_id: u64) -> Result<ChainAddress> {
        let wallet = self.evm_l2_mut(chain_id)?;
        let chain_name = wallet.chain().name.to_string();
        let addr = wallet.new_address()?;
        Ok(ChainAddress {
            network: chain_name,
            address: addr.address,
            derivation_path: addr.derivation_path,
            index: addr.index,
            created_at: Utc::now(),
        })
    }

    /// Sign EVM L2 transaction
    pub fn evm_l2_sign(
        &self,
        chain_id: u64,
        tx: &EvmTransaction,
        from_index: u32,
    ) -> Result<crate::evm::SignedEvmTransaction> {
        self.evm_l2(chain_id)?.sign_transaction(tx, from_index)
    }

    // ========================================================================
    // Solana operations
    // ========================================================================

    /// Get Solana wallet
    pub fn sol(&self) -> Result<&SolanaWallet> {
        self.sol.as_ref().ok_or(WalletError::NotInitialized)
    }

    /// Get mutable Solana wallet
    pub fn sol_mut(&mut self) -> Result<&mut SolanaWallet> {
        self.sol.as_mut().ok_or(WalletError::NotInitialized)
    }

    /// Generate new SOL address
    pub fn sol_new_address(&mut self) -> Result<ChainAddress> {
        let addr = self.sol_mut()?.new_address()?;
        Ok(addr.into())
    }

    /// Sign SOL transaction
    pub fn sol_sign(
        &self,
        tx: &SolanaTransaction,
        from_index: u32,
    ) -> Result<crate::sol::SignedSolanaTransaction> {
        self.sol()?.sign_transaction(tx, from_index)
    }

    // ========================================================================
    // Cross-chain utilities
    // ========================================================================

    /// Get all addresses across all chains
    pub fn get_all_addresses(&self) -> Vec<ChainAddress> {
        let mut addresses = Vec::new();

        // BTC addresses
        if let Some(btc) = &self.btc {
            for addr in btc.get_all_addresses() {
                addresses.push(addr.into());
            }
        }

        // ETH addresses
        if let Some(eth) = &self.eth {
            for addr in eth.get_all_addresses() {
                addresses.push(addr.into());
            }
        }

        // SOL addresses
        if let Some(sol) = &self.sol {
            for addr in sol.get_all_addresses() {
                addresses.push(addr.into());
            }
        }

        addresses
    }

    /// Get supported networks
    pub fn supported_networks(&self) -> Vec<Network> {
        let mut networks = Vec::new();

        if self.btc.is_some() {
            networks.push(Network::Bitcoin);
        }
        if self.eth.is_some() {
            networks.push(Network::Ethereum);
        }
        for chain_id in self.evm_l2s.keys() {
            match *chain_id {
                137 => networks.push(Network::Polygon),
                42161 => networks.push(Network::Arbitrum),
                8453 => networks.push(Network::Base),
                10 => networks.push(Network::Optimism),
                _ => {}
            }
        }
        if self.sol.is_some() {
            networks.push(Network::Solana);
        }

        networks
    }

    /// Get primary address for each chain (index 0)
    pub fn get_primary_addresses(&self) -> Result<HashMap<Network, ChainAddress>> {
        let mut addresses = HashMap::new();

        // BTC
        if let Some(btc) = &self.btc {
            let addr = btc.get_address(false, 0)?;
            addresses.insert(Network::Bitcoin, addr.into());
        }

        // ETH (same address for all EVM chains)
        if let Some(eth) = &self.eth {
            let addr = eth.get_address(0)?;
            addresses.insert(Network::Ethereum, addr.clone().into());

            // EVM L2s share the same address
            for (chain_id, _) in &self.evm_l2s {
                let network = match *chain_id {
                    137 => Network::Polygon,
                    42161 => Network::Arbitrum,
                    8453 => Network::Base,
                    10 => Network::Optimism,
                    _ => continue,
                };
                addresses.insert(network, ChainAddress {
                    network: network.name().to_string(),
                    address: addr.address.clone(),
                    derivation_path: addr.derivation_path.clone(),
                    index: addr.index,
                    created_at: Utc::now(),
                });
            }
        }

        // SOL
        if let Some(sol) = &self.sol {
            let addr = sol.get_address(0)?;
            addresses.insert(Network::Solana, addr.into());
        }

        Ok(addresses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_wallet() -> MultiChainWallet {
        let mnemonic = Mnemonic::from_phrase(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        ).unwrap();
        MultiChainWallet::from_mnemonic(&mnemonic, "").unwrap()
    }

    #[test]
    fn test_multi_chain_initialization() {
        let wallet = test_wallet();
        assert!(wallet.is_initialized());

        let networks = wallet.supported_networks();
        assert!(networks.contains(&Network::Bitcoin));
        assert!(networks.contains(&Network::Ethereum));
        assert!(networks.contains(&Network::Solana));
        assert!(networks.contains(&Network::Polygon));
    }

    #[test]
    fn test_generate_addresses_all_chains() {
        let mut wallet = test_wallet();

        let btc = wallet.btc_new_address().unwrap();
        assert!(btc.address.starts_with("bc1"));

        let eth = wallet.eth_new_address().unwrap();
        assert!(eth.address.starts_with("0x"));

        let sol = wallet.sol_new_address().unwrap();
        assert!(!sol.address.is_empty());

        let polygon = wallet.evm_l2_new_address(137).unwrap();
        assert!(polygon.address.starts_with("0x"));
        // EVM L2 shares address with ETH
        assert_eq!(polygon.address, eth.address);
    }

    #[test]
    fn test_primary_addresses() {
        let wallet = test_wallet();
        let primaries = wallet.get_primary_addresses().unwrap();

        assert!(primaries.contains_key(&Network::Bitcoin));
        assert!(primaries.contains_key(&Network::Ethereum));
        assert!(primaries.contains_key(&Network::Solana));
        assert!(primaries.contains_key(&Network::Polygon));

        // EVM chains share address
        let eth_addr = &primaries.get(&Network::Ethereum).unwrap().address;
        let polygon_addr = &primaries.get(&Network::Polygon).unwrap().address;
        assert_eq!(eth_addr, polygon_addr);
    }

    #[test]
    fn test_fingerprint() {
        let wallet = test_wallet();
        let fingerprint = wallet.fingerprint();
        // Fingerprint should be consistent
        assert_ne!(fingerprint, [0u8; 4]);
    }
}
