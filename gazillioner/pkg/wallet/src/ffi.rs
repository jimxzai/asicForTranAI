//! FFI bindings for Go integration via CGO
//!
//! Provides C-compatible functions for wallet operations.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::mnemonic::Mnemonic;
use crate::hd::{ExtendedPrivateKey, DerivationPath};
use crate::btc::BitcoinWallet;
use crate::eth::EthereumWallet;
use crate::Network;
use crate::error::WalletError;

/// Global wallet state
struct WalletState {
    btc_wallet: Option<BitcoinWallet>,
    eth_wallet: Option<EthereumWallet>,
    mnemonic_backup: Option<String>,
    fingerprint: Option<String>,
}

static WALLET: Lazy<Mutex<WalletState>> = Lazy::new(|| {
    Mutex::new(WalletState {
        btc_wallet: None,
        eth_wallet: None,
        mnemonic_backup: None,
        fingerprint: None,
    })
});

/// Result codes
#[repr(C)]
pub enum WalletResultCode {
    Ok = 0,
    Error = 1,
    NotInitialized = 2,
    InvalidInput = 3,
    InvalidMnemonic = 4,
    AlreadyInitialized = 5,
}

/// String result
#[repr(C)]
pub struct WalletStringResult {
    pub code: WalletResultCode,
    pub data: *mut c_char,
    pub error: *mut c_char,
}

impl WalletStringResult {
    fn ok(data: String) -> Self {
        Self {
            code: WalletResultCode::Ok,
            data: CString::new(data).unwrap().into_raw(),
            error: ptr::null_mut(),
        }
    }

    fn err(code: WalletResultCode, msg: String) -> Self {
        Self {
            code,
            data: ptr::null_mut(),
            error: CString::new(msg).unwrap().into_raw(),
        }
    }
}

/// Initialize wallet with new mnemonic
/// Returns the 24-word mnemonic that MUST be backed up
#[no_mangle]
pub extern "C" fn wallet_init_new() -> WalletStringResult {
    let mut state = WALLET.lock().unwrap();

    if state.btc_wallet.is_some() || state.eth_wallet.is_some() {
        return WalletStringResult::err(
            WalletResultCode::AlreadyInitialized,
            "Wallet already initialized".into(),
        );
    }

    // Generate new mnemonic
    let mnemonic = match Mnemonic::generate() {
        Ok(m) => m,
        Err(e) => return WalletStringResult::err(WalletResultCode::Error, e.to_string()),
    };

    let phrase = mnemonic.phrase().to_string();
    let seed = mnemonic.to_seed_no_passphrase();

    // Create BTC wallet
    let btc_master = match ExtendedPrivateKey::from_seed(&seed, Network::Bitcoin) {
        Ok(k) => k,
        Err(e) => return WalletStringResult::err(WalletResultCode::Error, e.to_string()),
    };

    let btc_wallet = match BitcoinWallet::new(btc_master, Network::Bitcoin) {
        Ok(w) => w,
        Err(e) => return WalletStringResult::err(WalletResultCode::Error, e.to_string()),
    };

    // Create ETH wallet
    let eth_master = match ExtendedPrivateKey::from_seed(&seed, Network::Ethereum) {
        Ok(k) => k,
        Err(e) => return WalletStringResult::err(WalletResultCode::Error, e.to_string()),
    };

    let eth_wallet = match EthereumWallet::new(eth_master, Network::Ethereum) {
        Ok(w) => w,
        Err(e) => return WalletStringResult::err(WalletResultCode::Error, e.to_string()),
    };

    let fingerprint = hex::encode(btc_wallet.fingerprint());

    state.btc_wallet = Some(btc_wallet);
    state.eth_wallet = Some(eth_wallet);
    state.mnemonic_backup = Some(phrase.clone());
    state.fingerprint = Some(fingerprint);

    WalletStringResult::ok(phrase)
}

/// Initialize wallet from existing mnemonic
#[no_mangle]
pub unsafe extern "C" fn wallet_init_from_mnemonic(
    phrase: *const c_char,
    passphrase: *const c_char,
) -> WalletResultCode {
    if phrase.is_null() {
        return WalletResultCode::InvalidInput;
    }

    let phrase_str = match CStr::from_ptr(phrase).to_str() {
        Ok(s) => s,
        Err(_) => return WalletResultCode::InvalidInput,
    };

    let passphrase_str = if passphrase.is_null() {
        ""
    } else {
        match CStr::from_ptr(passphrase).to_str() {
            Ok(s) => s,
            Err(_) => return WalletResultCode::InvalidInput,
        }
    };

    let mut state = WALLET.lock().unwrap();

    if state.btc_wallet.is_some() || state.eth_wallet.is_some() {
        return WalletResultCode::AlreadyInitialized;
    }

    // Parse mnemonic
    let mnemonic = match Mnemonic::from_phrase(phrase_str) {
        Ok(m) => m,
        Err(_) => return WalletResultCode::InvalidMnemonic,
    };

    let seed = mnemonic.to_seed(passphrase_str);

    // Create wallets
    let btc_master = match ExtendedPrivateKey::from_seed(&seed, Network::Bitcoin) {
        Ok(k) => k,
        Err(_) => return WalletResultCode::Error,
    };

    let btc_wallet = match BitcoinWallet::new(btc_master, Network::Bitcoin) {
        Ok(w) => w,
        Err(_) => return WalletResultCode::Error,
    };

    let eth_master = match ExtendedPrivateKey::from_seed(&seed, Network::Ethereum) {
        Ok(k) => k,
        Err(_) => return WalletResultCode::Error,
    };

    let eth_wallet = match EthereumWallet::new(eth_master, Network::Ethereum) {
        Ok(w) => w,
        Err(_) => return WalletResultCode::Error,
    };

    let fingerprint = hex::encode(btc_wallet.fingerprint());

    state.btc_wallet = Some(btc_wallet);
    state.eth_wallet = Some(eth_wallet);
    state.fingerprint = Some(fingerprint);
    // Don't store mnemonic for recovery - user should have it backed up

    WalletResultCode::Ok
}

/// Check if wallet is initialized
#[no_mangle]
pub extern "C" fn wallet_is_initialized() -> bool {
    let state = WALLET.lock().unwrap();
    state.btc_wallet.is_some() && state.eth_wallet.is_some()
}

/// Get wallet fingerprint
#[no_mangle]
pub extern "C" fn wallet_fingerprint() -> WalletStringResult {
    let state = WALLET.lock().unwrap();
    match &state.fingerprint {
        Some(fp) => WalletStringResult::ok(fp.clone()),
        None => WalletStringResult::err(
            WalletResultCode::NotInitialized,
            "Wallet not initialized".into(),
        ),
    }
}

/// Generate new BTC receive address
#[no_mangle]
pub extern "C" fn wallet_btc_new_address() -> WalletStringResult {
    let mut state = WALLET.lock().unwrap();
    let wallet = match state.btc_wallet.as_mut() {
        Some(w) => w,
        None => return WalletStringResult::err(
            WalletResultCode::NotInitialized,
            "Wallet not initialized".into(),
        ),
    };

    match wallet.new_receive_address() {
        Ok(addr) => {
            let json = serde_json::to_string(&addr).unwrap_or_default();
            WalletStringResult::ok(json)
        }
        Err(e) => WalletStringResult::err(WalletResultCode::Error, e.to_string()),
    }
}

/// Get BTC address at index
#[no_mangle]
pub extern "C" fn wallet_btc_get_address(index: u32, is_change: bool) -> WalletStringResult {
    let state = WALLET.lock().unwrap();
    let wallet = match state.btc_wallet.as_ref() {
        Some(w) => w,
        None => return WalletStringResult::err(
            WalletResultCode::NotInitialized,
            "Wallet not initialized".into(),
        ),
    };

    match wallet.get_address(is_change, index) {
        Ok(addr) => {
            let json = serde_json::to_string(&addr).unwrap_or_default();
            WalletStringResult::ok(json)
        }
        Err(e) => WalletStringResult::err(WalletResultCode::Error, e.to_string()),
    }
}

/// Generate new ETH address
#[no_mangle]
pub extern "C" fn wallet_eth_new_address() -> WalletStringResult {
    let mut state = WALLET.lock().unwrap();
    let wallet = match state.eth_wallet.as_mut() {
        Some(w) => w,
        None => return WalletStringResult::err(
            WalletResultCode::NotInitialized,
            "Wallet not initialized".into(),
        ),
    };

    match wallet.new_address() {
        Ok(addr) => {
            let json = serde_json::to_string(&addr).unwrap_or_default();
            WalletStringResult::ok(json)
        }
        Err(e) => WalletStringResult::err(WalletResultCode::Error, e.to_string()),
    }
}

/// Get ETH address at index
#[no_mangle]
pub extern "C" fn wallet_eth_get_address(index: u32) -> WalletStringResult {
    let state = WALLET.lock().unwrap();
    let wallet = match state.eth_wallet.as_ref() {
        Some(w) => w,
        None => return WalletStringResult::err(
            WalletResultCode::NotInitialized,
            "Wallet not initialized".into(),
        ),
    };

    match wallet.get_address(index) {
        Ok(addr) => {
            let json = serde_json::to_string(&addr).unwrap_or_default();
            WalletStringResult::ok(json)
        }
        Err(e) => WalletStringResult::err(WalletResultCode::Error, e.to_string()),
    }
}

/// Sign BTC transaction (JSON input)
#[no_mangle]
pub unsafe extern "C" fn wallet_btc_sign_tx(tx_json: *const c_char) -> WalletStringResult {
    if tx_json.is_null() {
        return WalletStringResult::err(WalletResultCode::InvalidInput, "Null input".into());
    }

    let tx_str = match CStr::from_ptr(tx_json).to_str() {
        Ok(s) => s,
        Err(_) => return WalletStringResult::err(WalletResultCode::InvalidInput, "Invalid UTF-8".into()),
    };

    let tx: crate::btc::BitcoinTransaction = match serde_json::from_str(tx_str) {
        Ok(t) => t,
        Err(e) => return WalletStringResult::err(WalletResultCode::InvalidInput, e.to_string()),
    };

    let state = WALLET.lock().unwrap();
    let wallet = match state.btc_wallet.as_ref() {
        Some(w) => w,
        None => return WalletStringResult::err(
            WalletResultCode::NotInitialized,
            "Wallet not initialized".into(),
        ),
    };

    match wallet.sign_transaction(&tx) {
        Ok(signed) => {
            let json = serde_json::to_string(&signed).unwrap_or_default();
            WalletStringResult::ok(json)
        }
        Err(e) => WalletStringResult::err(WalletResultCode::Error, e.to_string()),
    }
}

/// Sign ETH transaction (JSON input)
#[no_mangle]
pub unsafe extern "C" fn wallet_eth_sign_tx(
    tx_json: *const c_char,
    from_index: u32,
) -> WalletStringResult {
    if tx_json.is_null() {
        return WalletStringResult::err(WalletResultCode::InvalidInput, "Null input".into());
    }

    let tx_str = match CStr::from_ptr(tx_json).to_str() {
        Ok(s) => s,
        Err(_) => return WalletStringResult::err(WalletResultCode::InvalidInput, "Invalid UTF-8".into()),
    };

    let tx: crate::eth::EthereumTransaction = match serde_json::from_str(tx_str) {
        Ok(t) => t,
        Err(e) => return WalletStringResult::err(WalletResultCode::InvalidInput, e.to_string()),
    };

    let state = WALLET.lock().unwrap();
    let wallet = match state.eth_wallet.as_ref() {
        Some(w) => w,
        None => return WalletStringResult::err(
            WalletResultCode::NotInitialized,
            "Wallet not initialized".into(),
        ),
    };

    match wallet.sign_transaction(&tx, from_index) {
        Ok(signed) => {
            let json = serde_json::to_string(&signed).unwrap_or_default();
            WalletStringResult::ok(json)
        }
        Err(e) => WalletStringResult::err(WalletResultCode::Error, e.to_string()),
    }
}

/// Validate mnemonic phrase
#[no_mangle]
pub unsafe extern "C" fn wallet_validate_mnemonic(phrase: *const c_char) -> bool {
    if phrase.is_null() {
        return false;
    }

    match CStr::from_ptr(phrase).to_str() {
        Ok(s) => Mnemonic::validate(s),
        Err(_) => false,
    }
}

/// Close wallet and zeroize sensitive data
#[no_mangle]
pub extern "C" fn wallet_close() -> WalletResultCode {
    let mut state = WALLET.lock().unwrap();
    state.btc_wallet = None;
    state.eth_wallet = None;
    state.mnemonic_backup = None;
    state.fingerprint = None;
    WalletResultCode::Ok
}

/// Free string allocated by wallet functions
#[no_mangle]
pub unsafe extern "C" fn wallet_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Free result struct
#[no_mangle]
pub unsafe extern "C" fn wallet_free_result(result: WalletStringResult) {
    wallet_free_string(result.data);
    wallet_free_string(result.error);
}

/// Get library version
#[no_mangle]
pub extern "C" fn wallet_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}
