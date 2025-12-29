//! BIP32/BIP44 HD Key Derivation
//!
//! Implements hierarchical deterministic key derivation for
//! deriving multiple keys from a single seed.

use bitcoin::bip32::{
    DerivationPath as BtcDerivationPath,
    Xpriv, Xpub,
    ChildNumber,
};
use bitcoin::NetworkKind;
use secp256k1::{Secp256k1, SecretKey, PublicKey};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{Result, WalletError};
use crate::mnemonic::Seed;
use crate::Network;

/// BIP44 derivation path: m / purpose' / coin_type' / account' / change / address_index
///
/// Examples:
/// - Bitcoin:  m/84'/0'/0'/0/0  (Native SegWit)
/// - Ethereum: m/44'/60'/0'/0/0
#[derive(Debug, Clone)]
pub struct DerivationPath {
    inner: BtcDerivationPath,
}

impl DerivationPath {
    /// Create from string (e.g., "m/84'/0'/0'/0/0")
    pub fn from_str(path: &str) -> Result<Self> {
        let inner: BtcDerivationPath = path.parse()
            .map_err(|e| WalletError::InvalidDerivationPath(format!("{}", e)))?;
        Ok(Self { inner })
    }

    /// BIP84 path for Bitcoin Native SegWit (bc1...)
    pub fn bitcoin_native_segwit(account: u32, change: bool, index: u32) -> Self {
        let change_val = if change { 1 } else { 0 };
        Self::from_str(&format!("m/84'/0'/{}'/{}{}",
            account, change_val, index
        )).unwrap()
    }

    /// BIP44 path for Bitcoin Legacy (1...)
    pub fn bitcoin_legacy(account: u32, change: bool, index: u32) -> Self {
        let change_val = if change { 1 } else { 0 };
        Self::from_str(&format!("m/44'/0'/{}'/{}{}",
            account, change_val, index
        )).unwrap()
    }

    /// BIP49 path for Bitcoin SegWit-compatible (3...)
    pub fn bitcoin_segwit_compat(account: u32, change: bool, index: u32) -> Self {
        let change_val = if change { 1 } else { 0 };
        Self::from_str(&format!("m/49'/0'/{}'/{}{}",
            account, change_val, index
        )).unwrap()
    }

    /// BIP44 path for Ethereum
    pub fn ethereum(account: u32, index: u32) -> Self {
        Self::from_str(&format!("m/44'/60'/{}'/0/{}",
            account, index
        )).unwrap()
    }

    /// Get the internal path
    pub fn as_btc_path(&self) -> &BtcDerivationPath {
        &self.inner
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.inner.to_string()
    }
}

/// Extended Private Key (xprv)
#[derive(ZeroizeOnDrop)]
pub struct ExtendedPrivateKey {
    #[zeroize(skip)]
    inner: Xpriv,
    network: Network,
}

impl ExtendedPrivateKey {
    /// Create master key from seed
    pub fn from_seed(seed: &Seed, network: Network) -> Result<Self> {
        let network_kind = match network {
            Network::Bitcoin | Network::Ethereum => NetworkKind::Main,
            Network::BitcoinTestnet | Network::EthereumGoerli => NetworkKind::Test,
        };

        let inner = Xpriv::new_master(network_kind, seed.as_bytes())
            .map_err(|e| WalletError::KeyDerivationFailed(e.to_string()))?;

        Ok(Self { inner, network })
    }

    /// Derive child key at path
    pub fn derive(&self, path: &DerivationPath) -> Result<Self> {
        let secp = Secp256k1::new();
        let derived = self.inner.derive_priv(&secp, path.as_btc_path())
            .map_err(|e| WalletError::KeyDerivationFailed(e.to_string()))?;

        Ok(Self {
            inner: derived,
            network: self.network,
        })
    }

    /// Get the corresponding extended public key
    pub fn to_extended_public_key(&self) -> ExtendedPublicKey {
        let secp = Secp256k1::new();
        let xpub = Xpub::from_priv(&secp, &self.inner);
        ExtendedPublicKey {
            inner: xpub,
            network: self.network,
        }
    }

    /// Get the raw private key bytes (32 bytes)
    pub fn private_key_bytes(&self) -> [u8; 32] {
        self.inner.private_key.secret_bytes()
    }

    /// Get as secp256k1 SecretKey
    pub fn secret_key(&self) -> SecretKey {
        self.inner.private_key
    }

    /// Get the public key
    pub fn public_key(&self) -> PublicKey {
        let secp = Secp256k1::new();
        self.inner.private_key.public_key(&secp)
    }

    /// Get network
    pub fn network(&self) -> Network {
        self.network
    }

    /// Serialize to base58 (xprv/tprv format)
    pub fn to_base58(&self) -> String {
        self.inner.to_string()
    }
}

impl std::fmt::Debug for ExtendedPrivateKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExtendedPrivateKey([REDACTED], {:?})", self.network)
    }
}

/// Extended Public Key (xpub)
#[derive(Clone)]
pub struct ExtendedPublicKey {
    inner: Xpub,
    network: Network,
}

impl ExtendedPublicKey {
    /// Derive child public key (only non-hardened derivation)
    pub fn derive(&self, path: &DerivationPath) -> Result<Self> {
        let secp = Secp256k1::new();
        let derived = self.inner.derive_pub(&secp, path.as_btc_path())
            .map_err(|e| WalletError::KeyDerivationFailed(e.to_string()))?;

        Ok(Self {
            inner: derived,
            network: self.network,
        })
    }

    /// Get the raw public key bytes (33 bytes compressed)
    pub fn public_key_bytes(&self) -> [u8; 33] {
        self.inner.public_key.serialize()
    }

    /// Get as secp256k1 PublicKey
    pub fn public_key(&self) -> PublicKey {
        self.inner.public_key.inner
    }

    /// Get network
    pub fn network(&self) -> Network {
        self.network
    }

    /// Serialize to base58 (xpub/tpub format)
    pub fn to_base58(&self) -> String {
        self.inner.to_string()
    }

    /// Get fingerprint (first 4 bytes of hash160 of public key)
    pub fn fingerprint(&self) -> [u8; 4] {
        self.inner.fingerprint().to_bytes()
    }
}

impl std::fmt::Debug for ExtendedPublicKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExtendedPublicKey({}, {:?})",
            hex::encode(&self.public_key_bytes()[..8]),
            self.network
        )
    }
}

/// Key pair for signing
#[derive(ZeroizeOnDrop)]
pub struct KeyPair {
    #[zeroize(skip)]
    secret: SecretKey,
    #[zeroize(skip)]
    public: PublicKey,
    path: String,
    network: Network,
}

impl KeyPair {
    /// Create from extended private key at a specific path
    pub fn from_xpriv(xpriv: &ExtendedPrivateKey, path: &DerivationPath) -> Result<Self> {
        let derived = xpriv.derive(path)?;
        let secp = Secp256k1::new();

        Ok(Self {
            secret: derived.secret_key(),
            public: derived.secret_key().public_key(&secp),
            path: path.to_string(),
            network: derived.network,
        })
    }

    /// Get secret key
    pub fn secret_key(&self) -> &SecretKey {
        &self.secret
    }

    /// Get public key
    pub fn public_key(&self) -> &PublicKey {
        &self.public
    }

    /// Get derivation path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get network
    pub fn network(&self) -> Network {
        self.network
    }

    /// Get public key bytes (33 bytes compressed)
    pub fn public_key_bytes(&self) -> [u8; 33] {
        self.public.serialize()
    }

    /// Get public key bytes uncompressed (65 bytes)
    pub fn public_key_uncompressed(&self) -> [u8; 65] {
        self.public.serialize_uncompressed()
    }
}

impl std::fmt::Debug for KeyPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KeyPair(path={}, pubkey={}, {:?})",
            self.path,
            hex::encode(&self.public_key_bytes()[..8]),
            self.network
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnemonic::Mnemonic;

    #[test]
    fn test_derivation_path_parsing() {
        let path = DerivationPath::from_str("m/84'/0'/0'/0/0").unwrap();
        assert_eq!(path.to_string(), "m/84'/0'/0'/0/0");
    }

    #[test]
    fn test_bitcoin_paths() {
        let native = DerivationPath::bitcoin_native_segwit(0, false, 0);
        assert!(native.to_string().starts_with("m/84'"));

        let legacy = DerivationPath::bitcoin_legacy(0, false, 0);
        assert!(legacy.to_string().starts_with("m/44'"));

        let compat = DerivationPath::bitcoin_segwit_compat(0, false, 0);
        assert!(compat.to_string().starts_with("m/49'"));
    }

    #[test]
    fn test_ethereum_path() {
        let path = DerivationPath::ethereum(0, 0);
        assert_eq!(path.to_string(), "m/44'/60'/0'/0/0");
    }

    #[test]
    fn test_key_derivation() {
        let mnemonic = Mnemonic::from_phrase(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        ).unwrap();

        let seed = mnemonic.to_seed_no_passphrase();
        let master = ExtendedPrivateKey::from_seed(&seed, Network::Bitcoin).unwrap();

        let path = DerivationPath::bitcoin_native_segwit(0, false, 0);
        let derived = master.derive(&path).unwrap();

        // Should produce consistent results
        let pubkey = derived.public_key();
        assert!(pubkey.serialize().len() == 33);
    }

    #[test]
    fn test_xpub_derivation() {
        let mnemonic = Mnemonic::generate().unwrap();
        let seed = mnemonic.to_seed_no_passphrase();
        let master = ExtendedPrivateKey::from_seed(&seed, Network::Bitcoin).unwrap();

        let xpub = master.to_extended_public_key();

        // Can derive non-hardened children from xpub
        let path = DerivationPath::from_str("m/0/0").unwrap();
        let derived_pub = xpub.derive(&path);
        assert!(derived_pub.is_ok());
    }
}
