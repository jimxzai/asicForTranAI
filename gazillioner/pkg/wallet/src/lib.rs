//! Gazillioner Wallet - BTC/ETH Cold Storage
//!
//! This crate provides a secure, air-gapped capable wallet for Bitcoin and Ethereum.
//!
//! Features:
//! - BIP39 mnemonic generation and recovery (24 words)
//! - BIP32/BIP44 HD key derivation
//! - BTC transaction signing (SegWit native)
//! - ETH transaction signing (EIP-155)
//! - Air-gapped signing via QR codes
//! - Zeroization of sensitive data

pub mod mnemonic;
pub mod hd;
pub mod btc;
pub mod eth;
pub mod signing;
pub mod airgap;
pub mod error;
pub mod ffi;

pub use mnemonic::Mnemonic;
pub use hd::{ExtendedPrivateKey, ExtendedPublicKey, DerivationPath};
pub use btc::{BitcoinWallet, BitcoinAddress, BitcoinTransaction};
pub use eth::{EthereumWallet, EthereumAddress, EthereumTransaction};
pub use signing::{SigningRequest, SignedTransaction};
pub use airgap::{AirGapRequest, AirGapResponse};
pub use error::{WalletError, Result};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Supported networks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Network {
    Bitcoin,
    BitcoinTestnet,
    Ethereum,
    EthereumGoerli,
}

impl Network {
    pub fn is_mainnet(&self) -> bool {
        matches!(self, Network::Bitcoin | Network::Ethereum)
    }

    pub fn coin_type(&self) -> u32 {
        match self {
            Network::Bitcoin => 0,
            Network::BitcoinTestnet => 1,
            Network::Ethereum | Network::EthereumGoerli => 60,
        }
    }
}
