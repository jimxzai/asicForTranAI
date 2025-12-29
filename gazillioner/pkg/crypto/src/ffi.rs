//! FFI bindings for Go integration via CGO
//!
//! This module provides C-compatible functions that can be called from Go
//! using CGO. All functions follow the C ABI and handle memory safety.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use crate::crypto::KeyDerivation;
use crate::db::Database;
use crate::error::Error;
use crate::models::*;

/// Global database instance (protected by mutex)
static DATABASE: Lazy<Mutex<Option<Database>>> = Lazy::new(|| Mutex::new(None));

/// Result codes for FFI functions
#[repr(C)]
pub enum ResultCode {
    Ok = 0,
    Error = 1,
    NotInitialized = 2,
    InvalidInput = 3,
    NotFound = 4,
    AuthFailed = 5,
}

/// FFI-safe string result
#[repr(C)]
pub struct StringResult {
    pub code: ResultCode,
    pub data: *mut c_char,
    pub error: *mut c_char,
}

impl StringResult {
    fn ok(data: String) -> Self {
        Self {
            code: ResultCode::Ok,
            data: CString::new(data).unwrap().into_raw(),
            error: ptr::null_mut(),
        }
    }

    fn err(code: ResultCode, msg: String) -> Self {
        Self {
            code,
            data: ptr::null_mut(),
            error: CString::new(msg).unwrap().into_raw(),
        }
    }
}

/// Initialize the database with a key derived from PIN
///
/// # Safety
/// - `db_path` must be a valid null-terminated C string
/// - `pin` must be a valid null-terminated C string with exactly 6 digits
#[no_mangle]
pub unsafe extern "C" fn gazillioner_db_init(
    db_path: *const c_char,
    device_key: *const u8,
    device_key_len: usize,
    salt: *const u8,
    salt_len: usize,
    pin: *const c_char,
) -> ResultCode {
    // Validate inputs
    if db_path.is_null() || device_key.is_null() || salt.is_null() || pin.is_null() {
        return ResultCode::InvalidInput;
    }

    if device_key_len != 32 || salt_len != 32 {
        return ResultCode::InvalidInput;
    }

    let path = match CStr::from_ptr(db_path).to_str() {
        Ok(s) => s,
        Err(_) => return ResultCode::InvalidInput,
    };

    let pin_str = match CStr::from_ptr(pin).to_str() {
        Ok(s) => s,
        Err(_) => return ResultCode::InvalidInput,
    };

    // Create key arrays
    let mut dk = [0u8; 32];
    let mut s = [0u8; 32];
    ptr::copy_nonoverlapping(device_key, dk.as_mut_ptr(), 32);
    ptr::copy_nonoverlapping(salt, s.as_mut_ptr(), 32);

    // Derive encryption key
    let kd = KeyDerivation::from_keys(dk, s);
    let db_key = match kd.derive_db_key(pin_str) {
        Ok(key) => key,
        Err(Error::InvalidPinFormat) => return ResultCode::InvalidInput,
        Err(_) => return ResultCode::AuthFailed,
    };

    // Open database
    let db = match Database::open(path, &db_key) {
        Ok(db) => db,
        Err(_) => return ResultCode::Error,
    };

    // Store in global
    let mut guard = DATABASE.lock().unwrap();
    *guard = Some(db);

    ResultCode::Ok
}

/// Close the database connection
#[no_mangle]
pub extern "C" fn gazillioner_db_close() -> ResultCode {
    let mut guard = DATABASE.lock().unwrap();
    *guard = None;
    ResultCode::Ok
}

/// Check if database is initialized
#[no_mangle]
pub extern "C" fn gazillioner_db_is_initialized() -> bool {
    let guard = DATABASE.lock().unwrap();
    guard.is_some()
}

/// Create a new holding
///
/// # Safety
/// - All string parameters must be valid null-terminated C strings
#[no_mangle]
pub unsafe extern "C" fn gazillioner_holding_create(
    ticker: *const c_char,
    quantity: f64,
    cost_basis: f64,
    notes: *const c_char,
) -> StringResult {
    let guard = DATABASE.lock().unwrap();
    let db = match guard.as_ref() {
        Some(db) => db,
        None => return StringResult::err(ResultCode::NotInitialized, "Database not initialized".into()),
    };

    let ticker_str = match CStr::from_ptr(ticker).to_str() {
        Ok(s) => s,
        Err(_) => return StringResult::err(ResultCode::InvalidInput, "Invalid ticker".into()),
    };

    let notes_str = if notes.is_null() {
        None
    } else {
        CStr::from_ptr(notes).to_str().ok().map(String::from)
    };

    let req = CreateHolding {
        ticker: ticker_str.to_string(),
        quantity,
        cost_basis,
        acquisition_date: None,
        notes: notes_str,
        asset_class: None,
    };

    match db.create_holding(req) {
        Ok(holding) => {
            let json = serde_json::to_string(&holding).unwrap_or_default();
            StringResult::ok(json)
        }
        Err(e) => StringResult::err(ResultCode::Error, e.to_string()),
    }
}

/// List all holdings as JSON array
#[no_mangle]
pub extern "C" fn gazillioner_holding_list() -> StringResult {
    let guard = DATABASE.lock().unwrap();
    let db = match guard.as_ref() {
        Some(db) => db,
        None => return StringResult::err(ResultCode::NotInitialized, "Database not initialized".into()),
    };

    match db.list_holdings() {
        Ok(holdings) => {
            let json = serde_json::to_string(&holdings).unwrap_or_default();
            StringResult::ok(json)
        }
        Err(e) => StringResult::err(ResultCode::Error, e.to_string()),
    }
}

/// Get a holding by ID
///
/// # Safety
/// - `id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn gazillioner_holding_get(id: *const c_char) -> StringResult {
    let guard = DATABASE.lock().unwrap();
    let db = match guard.as_ref() {
        Some(db) => db,
        None => return StringResult::err(ResultCode::NotInitialized, "Database not initialized".into()),
    };

    let id_str = match CStr::from_ptr(id).to_str() {
        Ok(s) => s,
        Err(_) => return StringResult::err(ResultCode::InvalidInput, "Invalid ID".into()),
    };

    match db.get_holding(id_str) {
        Ok(holding) => {
            let json = serde_json::to_string(&holding).unwrap_or_default();
            StringResult::ok(json)
        }
        Err(Error::NotFound { .. }) => StringResult::err(ResultCode::NotFound, "Holding not found".into()),
        Err(e) => StringResult::err(ResultCode::Error, e.to_string()),
    }
}

/// Delete a holding
///
/// # Safety
/// - `id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn gazillioner_holding_delete(id: *const c_char) -> ResultCode {
    let guard = DATABASE.lock().unwrap();
    let db = match guard.as_ref() {
        Some(db) => db,
        None => return ResultCode::NotInitialized,
    };

    let id_str = match CStr::from_ptr(id).to_str() {
        Ok(s) => s,
        Err(_) => return ResultCode::InvalidInput,
    };

    match db.delete_holding(id_str) {
        Ok(_) => ResultCode::Ok,
        Err(Error::NotFound { .. }) => ResultCode::NotFound,
        Err(_) => ResultCode::Error,
    }
}

/// Add ticker to watchlist
///
/// # Safety
/// - All string parameters must be valid null-terminated C strings
#[no_mangle]
pub unsafe extern "C" fn gazillioner_watchlist_add(
    ticker: *const c_char,
    notes: *const c_char,
) -> StringResult {
    let guard = DATABASE.lock().unwrap();
    let db = match guard.as_ref() {
        Some(db) => db,
        None => return StringResult::err(ResultCode::NotInitialized, "Database not initialized".into()),
    };

    let ticker_str = match CStr::from_ptr(ticker).to_str() {
        Ok(s) => s,
        Err(_) => return StringResult::err(ResultCode::InvalidInput, "Invalid ticker".into()),
    };

    let notes_str = if notes.is_null() {
        None
    } else {
        CStr::from_ptr(notes).to_str().ok()
    };

    match db.add_to_watchlist(ticker_str, notes_str) {
        Ok(item) => {
            let json = serde_json::to_string(&item).unwrap_or_default();
            StringResult::ok(json)
        }
        Err(e) => StringResult::err(ResultCode::Error, e.to_string()),
    }
}

/// List watchlist as JSON array
#[no_mangle]
pub extern "C" fn gazillioner_watchlist_list() -> StringResult {
    let guard = DATABASE.lock().unwrap();
    let db = match guard.as_ref() {
        Some(db) => db,
        None => return StringResult::err(ResultCode::NotInitialized, "Database not initialized".into()),
    };

    match db.list_watchlist() {
        Ok(items) => {
            let json = serde_json::to_string(&items).unwrap_or_default();
            StringResult::ok(json)
        }
        Err(e) => StringResult::err(ResultCode::Error, e.to_string()),
    }
}

/// Remove ticker from watchlist
///
/// # Safety
/// - `ticker` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn gazillioner_watchlist_remove(ticker: *const c_char) -> ResultCode {
    let guard = DATABASE.lock().unwrap();
    let db = match guard.as_ref() {
        Some(db) => db,
        None => return ResultCode::NotInitialized,
    };

    let ticker_str = match CStr::from_ptr(ticker).to_str() {
        Ok(s) => s,
        Err(_) => return ResultCode::InvalidInput,
    };

    match db.remove_from_watchlist(ticker_str) {
        Ok(_) => ResultCode::Ok,
        Err(Error::NotFound { .. }) => ResultCode::NotFound,
        Err(_) => ResultCode::Error,
    }
}

/// Validate a PIN format
///
/// # Safety
/// - `pin` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn gazillioner_validate_pin(pin: *const c_char) -> bool {
    if pin.is_null() {
        return false;
    }

    match CStr::from_ptr(pin).to_str() {
        Ok(s) => KeyDerivation::validate_pin(s),
        Err(_) => false,
    }
}

/// Free a string allocated by FFI functions
///
/// # Safety
/// - `ptr` must have been allocated by this library
#[no_mangle]
pub unsafe extern "C" fn gazillioner_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Free a StringResult
///
/// # Safety
/// - `result` must have been created by this library
#[no_mangle]
pub unsafe extern "C" fn gazillioner_free_result(result: StringResult) {
    gazillioner_free_string(result.data);
    gazillioner_free_string(result.error);
}

/// Get library version
#[no_mangle]
pub extern "C" fn gazillioner_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}
