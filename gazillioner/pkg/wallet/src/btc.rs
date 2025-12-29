//! Bitcoin Wallet - Native SegWit (BIP84) Support
//!
//! Implements Bitcoin address generation and transaction signing
//! for Native SegWit (bc1...) addresses.

use bitcoin::{
    Address, Network as BtcNetwork, PublicKey as BtcPublicKey,
    ScriptBuf, Sequence, Transaction, TxIn, TxOut, Witness,
    absolute::LockTime,
    transaction::Version,
    sighash::{SighashCache, EcdsaSighashType},
    ecdsa::Signature as BtcSignature,
    CompressedPublicKey,
};
use bitcoin::hashes::Hash;
use secp256k1::{Secp256k1, Message, SecretKey};
use serde::{Deserialize, Serialize};

use crate::error::{Result, WalletError};
use crate::hd::{ExtendedPrivateKey, DerivationPath, KeyPair};
use crate::Network;

/// Bitcoin address types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AddressType {
    /// Native SegWit (bc1q...) - BIP84
    NativeSegwit,
    /// SegWit-compatible (3...) - BIP49
    SegwitCompat,
    /// Legacy (1...) - BIP44
    Legacy,
    /// Taproot (bc1p...) - BIP86
    Taproot,
}

/// Bitcoin address
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitcoinAddress {
    pub address: String,
    pub address_type: AddressType,
    pub derivation_path: String,
    pub index: u32,
    pub is_change: bool,
}

impl BitcoinAddress {
    /// Create from public key (Native SegWit)
    pub fn from_public_key(
        pubkey: &secp256k1::PublicKey,
        network: Network,
        path: &DerivationPath,
        index: u32,
        is_change: bool,
    ) -> Result<Self> {
        let btc_network = match network {
            Network::Bitcoin => BtcNetwork::Bitcoin,
            Network::BitcoinTestnet => BtcNetwork::Testnet,
            _ => return Err(WalletError::NetworkMismatch {
                expected: "Bitcoin".into(),
                actual: format!("{:?}", network),
            }),
        };

        // Create compressed public key
        let compressed = CompressedPublicKey::from_slice(&pubkey.serialize())
            .map_err(|e| WalletError::InvalidPublicKey)?;

        // Generate Native SegWit address (P2WPKH)
        let address = Address::p2wpkh(&compressed, btc_network);

        Ok(Self {
            address: address.to_string(),
            address_type: AddressType::NativeSegwit,
            derivation_path: path.to_string(),
            index,
            is_change,
        })
    }

    /// Parse address string
    pub fn parse(address: &str, network: Network) -> Result<Address<bitcoin::address::NetworkUnchecked>> {
        address.parse()
            .map_err(|e| WalletError::InvalidAddress(format!("{}", e)))
    }
}

/// Unspent transaction output (UTXO)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Utxo {
    pub txid: String,
    pub vout: u32,
    pub amount_sats: u64,
    pub script_pubkey: String,
    pub address: String,
    pub derivation_path: String,
    pub confirmations: u32,
}

/// Transaction output (recipient)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxOutput {
    pub address: String,
    pub amount_sats: u64,
}

/// Unsigned Bitcoin transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitcoinTransaction {
    pub inputs: Vec<Utxo>,
    pub outputs: Vec<TxOutput>,
    pub change_address: Option<String>,
    pub fee_sats: u64,
    pub fee_rate_sat_vb: f64,
}

impl BitcoinTransaction {
    /// Create a new transaction
    pub fn new(
        inputs: Vec<Utxo>,
        outputs: Vec<TxOutput>,
        change_address: Option<String>,
        fee_rate_sat_vb: f64,
    ) -> Result<Self> {
        // Calculate total input
        let total_input: u64 = inputs.iter().map(|u| u.amount_sats).sum();

        // Calculate total output
        let total_output: u64 = outputs.iter().map(|o| o.amount_sats).sum();

        // Estimate transaction size (simplified)
        // P2WPKH: ~68 vbytes per input, ~31 vbytes per output, ~10.5 vbytes overhead
        let estimated_vsize = 10.5 + (inputs.len() as f64 * 68.0) + (outputs.len() as f64 * 31.0);
        let fee_sats = (estimated_vsize * fee_rate_sat_vb).ceil() as u64;

        if total_input < total_output + fee_sats {
            return Err(WalletError::InsufficientFunds {
                needed: total_output + fee_sats,
                available: total_input,
            });
        }

        Ok(Self {
            inputs,
            outputs,
            change_address,
            fee_sats,
            fee_rate_sat_vb,
        })
    }

    /// Get total input amount
    pub fn total_input(&self) -> u64 {
        self.inputs.iter().map(|u| u.amount_sats).sum()
    }

    /// Get total output amount (excluding change)
    pub fn total_output(&self) -> u64 {
        self.outputs.iter().map(|o| o.amount_sats).sum()
    }

    /// Get change amount
    pub fn change_amount(&self) -> u64 {
        self.total_input().saturating_sub(self.total_output() + self.fee_sats)
    }
}

/// Signed Bitcoin transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedBitcoinTransaction {
    pub raw_hex: String,
    pub txid: String,
    pub size_bytes: usize,
    pub vsize: usize,
    pub fee_sats: u64,
}

/// Bitcoin wallet for managing addresses and signing
pub struct BitcoinWallet {
    master_key: ExtendedPrivateKey,
    network: Network,
    next_receive_index: u32,
    next_change_index: u32,
}

impl BitcoinWallet {
    /// Create wallet from master extended private key
    pub fn new(master_key: ExtendedPrivateKey, network: Network) -> Result<Self> {
        if !matches!(network, Network::Bitcoin | Network::BitcoinTestnet) {
            return Err(WalletError::NetworkMismatch {
                expected: "Bitcoin or BitcoinTestnet".into(),
                actual: format!("{:?}", network),
            });
        }

        Ok(Self {
            master_key,
            network,
            next_receive_index: 0,
            next_change_index: 0,
        })
    }

    /// Generate new receive address
    pub fn new_receive_address(&mut self) -> Result<BitcoinAddress> {
        let index = self.next_receive_index;
        self.next_receive_index += 1;
        self.get_address(false, index)
    }

    /// Generate new change address
    pub fn new_change_address(&mut self) -> Result<BitcoinAddress> {
        let index = self.next_change_index;
        self.next_change_index += 1;
        self.get_address(true, index)
    }

    /// Get address at specific index
    pub fn get_address(&self, is_change: bool, index: u32) -> Result<BitcoinAddress> {
        let path = DerivationPath::bitcoin_native_segwit(0, is_change, index);
        let keypair = KeyPair::from_xpriv(&self.master_key, &path)?;

        BitcoinAddress::from_public_key(
            keypair.public_key(),
            self.network,
            &path,
            index,
            is_change,
        )
    }

    /// Sign a transaction
    pub fn sign_transaction(&self, tx: &BitcoinTransaction) -> Result<SignedBitcoinTransaction> {
        let secp = Secp256k1::new();

        let btc_network = match self.network {
            Network::Bitcoin => BtcNetwork::Bitcoin,
            Network::BitcoinTestnet => BtcNetwork::Testnet,
            _ => unreachable!(),
        };

        // Build unsigned transaction
        let mut unsigned_tx = Transaction {
            version: Version::TWO,
            lock_time: LockTime::ZERO,
            input: vec![],
            output: vec![],
        };

        // Add inputs
        for utxo in &tx.inputs {
            let txid: bitcoin::Txid = utxo.txid.parse()
                .map_err(|e| WalletError::InvalidTransaction(format!("Invalid txid: {}", e)))?;

            unsigned_tx.input.push(TxIn {
                previous_output: bitcoin::OutPoint {
                    txid,
                    vout: utxo.vout,
                },
                script_sig: ScriptBuf::new(),
                sequence: Sequence::ENABLE_RBF_NO_LOCKTIME,
                witness: Witness::default(),
            });
        }

        // Add outputs
        for output in &tx.outputs {
            let address: Address<bitcoin::address::NetworkUnchecked> = output.address.parse()
                .map_err(|e| WalletError::InvalidAddress(format!("{}", e)))?;

            let address = address.assume_checked();

            unsigned_tx.output.push(TxOut {
                value: bitcoin::Amount::from_sat(output.amount_sats),
                script_pubkey: address.script_pubkey(),
            });
        }

        // Add change output if needed
        let change_amount = tx.change_amount();
        if change_amount > 546 { // Dust threshold
            if let Some(change_addr) = &tx.change_address {
                let address: Address<bitcoin::address::NetworkUnchecked> = change_addr.parse()
                    .map_err(|e| WalletError::InvalidAddress(format!("{}", e)))?;

                let address = address.assume_checked();

                unsigned_tx.output.push(TxOut {
                    value: bitcoin::Amount::from_sat(change_amount),
                    script_pubkey: address.script_pubkey(),
                });
            }
        }

        // Sign each input
        let mut sighash_cache = SighashCache::new(&unsigned_tx);

        for (i, utxo) in tx.inputs.iter().enumerate() {
            // Derive the key for this input
            let path = DerivationPath::from_str(&utxo.derivation_path)?;
            let keypair = KeyPair::from_xpriv(&self.master_key, &path)?;

            // Create compressed public key
            let compressed = CompressedPublicKey::from_slice(&keypair.public_key_bytes())
                .map_err(|_| WalletError::InvalidPublicKey)?;

            // Create script pubkey for P2WPKH
            let script_pubkey = ScriptBuf::new_p2wpkh(&compressed.wpubkey_hash());

            // Compute sighash
            let sighash = sighash_cache.p2wpkh_signature_hash(
                i,
                &script_pubkey,
                bitcoin::Amount::from_sat(utxo.amount_sats),
                EcdsaSighashType::All,
            ).map_err(|e| WalletError::SigningFailed(format!("Sighash error: {}", e)))?;

            // Sign
            let msg = Message::from_digest(sighash.to_byte_array());
            let sig = secp.sign_ecdsa(&msg, keypair.secret_key());

            // Create Bitcoin signature with sighash type
            let btc_sig = BtcSignature {
                signature: sig,
                sighash_type: EcdsaSighashType::All,
            };

            // Build witness
            let mut witness = Witness::new();
            witness.push(btc_sig.to_vec());
            witness.push(keypair.public_key_bytes());

            // Update transaction with witness
            sighash_cache.witness_mut(i).unwrap().clone_from(&witness);
        }

        let signed_tx = sighash_cache.into_transaction();

        // Serialize
        let raw_hex = bitcoin::consensus::encode::serialize_hex(&signed_tx);
        let txid = signed_tx.compute_txid().to_string();
        let size_bytes = bitcoin::consensus::encode::serialize(&signed_tx).len();
        let vsize = signed_tx.vsize();

        Ok(SignedBitcoinTransaction {
            raw_hex,
            txid,
            size_bytes,
            vsize,
            fee_sats: tx.fee_sats,
        })
    }

    /// Get master fingerprint
    pub fn fingerprint(&self) -> [u8; 4] {
        self.master_key.to_extended_public_key().fingerprint()
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

    fn test_wallet() -> BitcoinWallet {
        let mnemonic = Mnemonic::from_phrase(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        ).unwrap();
        let seed = mnemonic.to_seed_no_passphrase();
        let master = ExtendedPrivateKey::from_seed(&seed, Network::Bitcoin).unwrap();
        BitcoinWallet::new(master, Network::Bitcoin).unwrap()
    }

    #[test]
    fn test_generate_address() {
        let mut wallet = test_wallet();
        let addr = wallet.new_receive_address().unwrap();

        // Should be a bc1 address
        assert!(addr.address.starts_with("bc1q"));
        assert_eq!(addr.address_type, AddressType::NativeSegwit);
    }

    #[test]
    fn test_address_derivation_consistency() {
        let wallet = test_wallet();

        let addr1 = wallet.get_address(false, 0).unwrap();
        let addr2 = wallet.get_address(false, 0).unwrap();

        // Same index should produce same address
        assert_eq!(addr1.address, addr2.address);

        // Different index should produce different address
        let addr3 = wallet.get_address(false, 1).unwrap();
        assert_ne!(addr1.address, addr3.address);
    }

    #[test]
    fn test_change_vs_receive() {
        let wallet = test_wallet();

        let receive = wallet.get_address(false, 0).unwrap();
        let change = wallet.get_address(true, 0).unwrap();

        // Same index but different chains should produce different addresses
        assert_ne!(receive.address, change.address);
    }
}
