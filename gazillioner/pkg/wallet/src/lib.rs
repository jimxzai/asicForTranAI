//! Gazillioner Wallet - Multi-Chain Cold Storage
//!
//! This crate provides a secure, air-gapped capable wallet for multiple chains.
//!
//! Supported Chains:
//! - Bitcoin (BTC) - Native SegWit (BIP84)
//! - Ethereum (ETH) - EIP-155, EIP-1559
//! - EVM L2s - Polygon, Arbitrum, Base, Optimism
//! - Solana (SOL) - Ed25519 signing
//!
//! Features:
//! - BIP39 mnemonic generation and recovery (24 words)
//! - BIP32/BIP44 HD key derivation
//! - SLIP-0044 coin type standards
//! - Air-gapped signing via QR codes
//! - Zeroization of sensitive data

use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

pub mod mnemonic;
pub mod hd;
pub mod btc;
pub mod eth;
pub mod evm;
pub mod sol;
pub mod signing;
pub mod airgap;
pub mod error;
pub mod ffi;
pub mod multichain;

pub use mnemonic::Mnemonic;
pub use hd::{ExtendedPrivateKey, ExtendedPublicKey, DerivationPath};
pub use btc::{BitcoinWallet, BitcoinAddress, BitcoinTransaction};
pub use eth::{EthereumWallet, EthereumAddress, EthereumTransaction};
pub use evm::{EvmWallet, EvmChain};
pub use sol::{SolanaWallet, SolanaAddress, SolanaTransaction};
pub use signing::{SigningRequest, SignedTransaction};
pub use airgap::{AirGapRequest, AirGapResponse};
pub use multichain::{MultiChainWallet, ChainAddress, ChainBalance};
pub use error::{WalletError, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Supported networks/chains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Network {
    // Bitcoin
    Bitcoin,
    BitcoinTestnet,
    // Ethereum L1
    Ethereum,
    EthereumGoerli,
    EthereumSepolia,
    // EVM L2s
    Polygon,
    PolygonMumbai,
    Arbitrum,
    ArbitrumGoerli,
    Base,
    BaseSepolia,
    Optimism,
    OptimismGoerli,
    // Solana
    Solana,
    SolanaDevnet,
}

// Implement Zeroize for Network (no-op since Network contains no sensitive data)
impl Zeroize for Network {
    fn zeroize(&mut self) {
        // Network is a Copy enum with no sensitive data, nothing to zeroize
    }
}

impl Network {
    pub fn is_mainnet(&self) -> bool {
        matches!(
            self,
            Network::Bitcoin
                | Network::Ethereum
                | Network::Polygon
                | Network::Arbitrum
                | Network::Base
                | Network::Optimism
                | Network::Solana
        )
    }

    /// BIP44 coin type (SLIP-0044)
    pub fn coin_type(&self) -> u32 {
        match self {
            Network::Bitcoin => 0,
            Network::BitcoinTestnet => 1,
            Network::Ethereum
            | Network::EthereumGoerli
            | Network::EthereumSepolia
            | Network::Polygon
            | Network::PolygonMumbai
            | Network::Arbitrum
            | Network::ArbitrumGoerli
            | Network::Base
            | Network::BaseSepolia
            | Network::Optimism
            | Network::OptimismGoerli => 60,
            Network::Solana | Network::SolanaDevnet => 501,
        }
    }

    /// Chain ID for EVM networks
    pub fn chain_id(&self) -> Option<u64> {
        match self {
            Network::Ethereum => Some(1),
            Network::EthereumGoerli => Some(5),
            Network::EthereumSepolia => Some(11155111),
            Network::Polygon => Some(137),
            Network::PolygonMumbai => Some(80001),
            Network::Arbitrum => Some(42161),
            Network::ArbitrumGoerli => Some(421613),
            Network::Base => Some(8453),
            Network::BaseSepolia => Some(84532),
            Network::Optimism => Some(10),
            Network::OptimismGoerli => Some(420),
            _ => None,
        }
    }

    /// Whether this is an EVM-compatible chain
    pub fn is_evm(&self) -> bool {
        self.chain_id().is_some()
    }

    /// Whether this is a Bitcoin-based chain
    pub fn is_bitcoin(&self) -> bool {
        matches!(self, Network::Bitcoin | Network::BitcoinTestnet)
    }

    /// Whether this is Solana
    pub fn is_solana(&self) -> bool {
        matches!(self, Network::Solana | Network::SolanaDevnet)
    }

    /// Network display name
    pub fn name(&self) -> &'static str {
        match self {
            Network::Bitcoin => "Bitcoin",
            Network::BitcoinTestnet => "Bitcoin Testnet",
            Network::Ethereum => "Ethereum",
            Network::EthereumGoerli => "Ethereum Goerli",
            Network::EthereumSepolia => "Ethereum Sepolia",
            Network::Polygon => "Polygon",
            Network::PolygonMumbai => "Polygon Mumbai",
            Network::Arbitrum => "Arbitrum One",
            Network::ArbitrumGoerli => "Arbitrum Goerli",
            Network::Base => "Base",
            Network::BaseSepolia => "Base Sepolia",
            Network::Optimism => "Optimism",
            Network::OptimismGoerli => "Optimism Goerli",
            Network::Solana => "Solana",
            Network::SolanaDevnet => "Solana Devnet",
        }
    }

    /// Native currency symbol
    pub fn symbol(&self) -> &'static str {
        match self {
            Network::Bitcoin | Network::BitcoinTestnet => "BTC",
            Network::Ethereum | Network::EthereumGoerli | Network::EthereumSepolia => "ETH",
            Network::Polygon | Network::PolygonMumbai => "MATIC",
            Network::Arbitrum | Network::ArbitrumGoerli => "ETH",
            Network::Base | Network::BaseSepolia => "ETH",
            Network::Optimism | Network::OptimismGoerli => "ETH",
            Network::Solana | Network::SolanaDevnet => "SOL",
        }
    }
}
