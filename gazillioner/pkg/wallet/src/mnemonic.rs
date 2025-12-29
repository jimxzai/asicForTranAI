//! BIP39 Mnemonic generation and recovery
//!
//! Implements secure 24-word mnemonic phrase generation using
//! cryptographically secure random number generation.

use bip39::{Language, Mnemonic as Bip39Mnemonic};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{Result, WalletError};

/// BIP39 Mnemonic wrapper with automatic zeroization
#[derive(Clone, ZeroizeOnDrop)]
pub struct Mnemonic {
    #[zeroize(skip)] // bip39::Mnemonic handles its own zeroization
    inner: Bip39Mnemonic,
}

impl Mnemonic {
    /// Generate a new 24-word mnemonic using secure randomness
    pub fn generate() -> Result<Self> {
        // 256 bits of entropy = 24 words
        let mnemonic = Bip39Mnemonic::generate_in(Language::English, 24)
            .map_err(|e| WalletError::InvalidMnemonic(e.to_string()))?;

        Ok(Self { inner: mnemonic })
    }

    /// Generate with specific word count (12, 15, 18, 21, or 24)
    pub fn generate_with_count(word_count: usize) -> Result<Self> {
        if ![12, 15, 18, 21, 24].contains(&word_count) {
            return Err(WalletError::InvalidMnemonic(
                "Word count must be 12, 15, 18, 21, or 24".into()
            ));
        }

        let mnemonic = Bip39Mnemonic::generate_in(Language::English, word_count)
            .map_err(|e| WalletError::InvalidMnemonic(e.to_string()))?;

        Ok(Self { inner: mnemonic })
    }

    /// Recover from existing mnemonic phrase
    pub fn from_phrase(phrase: &str) -> Result<Self> {
        let mnemonic = Bip39Mnemonic::parse_in_normalized(Language::English, phrase)
            .map_err(|e| WalletError::InvalidMnemonic(e.to_string()))?;

        Ok(Self { inner: mnemonic })
    }

    /// Recover from word list
    pub fn from_words(words: &[&str]) -> Result<Self> {
        let phrase = words.join(" ");
        Self::from_phrase(&phrase)
    }

    /// Get the mnemonic phrase as a string
    pub fn phrase(&self) -> String {
        self.inner.to_string()
    }

    /// Get individual words
    pub fn words(&self) -> Vec<&str> {
        self.inner.word_iter().collect()
    }

    /// Get word count
    pub fn word_count(&self) -> usize {
        self.inner.word_count()
    }

    /// Derive seed bytes with optional passphrase (BIP39)
    /// The passphrase provides additional security - different passphrases
    /// produce completely different seeds from the same mnemonic.
    pub fn to_seed(&self, passphrase: &str) -> Seed {
        let seed_bytes = self.inner.to_seed(passphrase);
        Seed { bytes: seed_bytes }
    }

    /// Derive seed without passphrase
    pub fn to_seed_no_passphrase(&self) -> Seed {
        self.to_seed("")
    }

    /// Validate a mnemonic phrase without creating a Mnemonic
    pub fn validate(phrase: &str) -> bool {
        Bip39Mnemonic::parse_in_normalized(Language::English, phrase).is_ok()
    }

    /// Get the language of the mnemonic
    pub fn language(&self) -> Language {
        self.inner.language()
    }
}

impl std::fmt::Debug for Mnemonic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Don't expose mnemonic in debug output
        write!(f, "Mnemonic([REDACTED {} words])", self.word_count())
    }
}

/// BIP39 Seed derived from mnemonic
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct Seed {
    bytes: [u8; 64],
}

impl Seed {
    /// Get the raw seed bytes (512 bits)
    pub fn as_bytes(&self) -> &[u8; 64] {
        &self.bytes
    }

    /// Create from raw bytes
    pub fn from_bytes(bytes: [u8; 64]) -> Self {
        Self { bytes }
    }
}

impl std::fmt::Debug for Seed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Seed([REDACTED])")
    }
}

impl AsRef<[u8]> for Seed {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

/// Utility for secure mnemonic display
pub struct MnemonicDisplay<'a> {
    mnemonic: &'a Mnemonic,
}

impl<'a> MnemonicDisplay<'a> {
    pub fn new(mnemonic: &'a Mnemonic) -> Self {
        Self { mnemonic }
    }

    /// Format as numbered list for backup
    pub fn numbered_list(&self) -> String {
        self.mnemonic
            .words()
            .iter()
            .enumerate()
            .map(|(i, word)| format!("{:2}. {}", i + 1, word))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format as grid (4 columns x 6 rows for 24 words)
    pub fn grid(&self) -> String {
        let words: Vec<_> = self.mnemonic.words();
        let mut result = String::new();

        for row in 0..6 {
            for col in 0..4 {
                let idx = row * 4 + col;
                if idx < words.len() {
                    result.push_str(&format!("{:2}. {:12} ", idx + 1, words[idx]));
                }
            }
            result.push('\n');
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_24_word_mnemonic() {
        let mnemonic = Mnemonic::generate().unwrap();
        assert_eq!(mnemonic.word_count(), 24);
    }

    #[test]
    fn test_generate_12_word_mnemonic() {
        let mnemonic = Mnemonic::generate_with_count(12).unwrap();
        assert_eq!(mnemonic.word_count(), 12);
    }

    #[test]
    fn test_recover_from_phrase() {
        let original = Mnemonic::generate().unwrap();
        let phrase = original.phrase().to_string();

        let recovered = Mnemonic::from_phrase(&phrase).unwrap();
        assert_eq!(original.phrase(), recovered.phrase());
    }

    #[test]
    fn test_seed_derivation() {
        let mnemonic = Mnemonic::from_phrase(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        ).unwrap();

        let seed1 = mnemonic.to_seed("");
        let seed2 = mnemonic.to_seed("password");

        // Different passphrases produce different seeds
        assert_ne!(seed1.as_bytes(), seed2.as_bytes());
    }

    #[test]
    fn test_invalid_mnemonic() {
        let result = Mnemonic::from_phrase("invalid words here");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate() {
        assert!(Mnemonic::validate(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        ));
        assert!(!Mnemonic::validate("invalid"));
    }
}
