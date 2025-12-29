//! FFI bindings for Go integration
//!
//! Provides C-compatible functions for the Go application to interact
//! with the sync library.

use crate::conflict::Strategy;
use crate::error::SyncError;
use crate::{DeviceId, DeviceInfo, PeerInfo, SyncConfig, SyncManager, SyncState, SyncStats};
use libc::{c_char, c_int, size_t};
use once_cell::sync::OnceCell;
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Mutex;

/// Global sync manager instance
static SYNC_MANAGER: OnceCell<Mutex<SyncManager>> = OnceCell::new();

/// Result codes for FFI functions
#[repr(C)]
pub enum SyncResultCode {
    Ok = 0,
    Error = -1,
    NotInitialized = -2,
    InvalidArgument = -3,
    NotPaired = -4,
    AlreadyPaired = -5,
    Timeout = -6,
    NetworkError = -7,
}

/// Device info for FFI
#[repr(C)]
pub struct FFIDeviceInfo {
    pub device_id: *mut c_char,
    pub device_name: *mut c_char,
    pub platform: *mut c_char,
}

/// Peer info for FFI
#[repr(C)]
pub struct FFIPeerInfo {
    pub device_id: *mut c_char,
    pub device_name: *mut c_char,
    pub address: *mut c_char,
    pub port: u16,
    pub is_paired: c_int,
}

/// Sync stats for FFI
#[repr(C)]
pub struct FFISyncStats {
    pub records_sent: u32,
    pub records_received: u32,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub conflicts_resolved: u32,
    pub duration_ms: u64,
}

/// Initialize the sync manager
///
/// # Safety
/// - `device_id` must be a valid null-terminated C string
/// - `device_name` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn sync_init(
    device_id: *const c_char,
    device_name: *const c_char,
    port: u16,
) -> c_int {
    if device_id.is_null() || device_name.is_null() {
        return SyncResultCode::InvalidArgument as c_int;
    }

    let device_id_str = match CStr::from_ptr(device_id).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return SyncResultCode::InvalidArgument as c_int,
    };

    let device_name_str = match CStr::from_ptr(device_name).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return SyncResultCode::InvalidArgument as c_int,
    };

    let device_info = DeviceInfo::new(DeviceId::from_string(device_id_str), device_name_str);

    let config = SyncConfig {
        device_info,
        listen_port: port,
        enable_discovery: true,
        max_concurrent_syncs: 2,
        sync_timeout_secs: 300,
        conflict_strategy: Strategy::LastWriteWins,
    };

    let manager = SyncManager::new(config);

    if SYNC_MANAGER.set(Mutex::new(manager)).is_err() {
        // Already initialized
        return SyncResultCode::Error as c_int;
    }

    SyncResultCode::Ok as c_int
}

/// Shutdown the sync manager
#[no_mangle]
pub extern "C" fn sync_shutdown() -> c_int {
    // Can't actually remove from OnceCell, but we can stop operations
    if let Some(manager) = SYNC_MANAGER.get() {
        if let Ok(mut m) = manager.lock() {
            // Stop any running operations
            let rt = tokio::runtime::Runtime::new();
            if let Ok(rt) = rt {
                rt.block_on(async {
                    let _ = m.stop_discovery().await;
                });
            }
        }
    }
    SyncResultCode::Ok as c_int
}

/// Get current sync state
#[no_mangle]
pub extern "C" fn sync_get_state() -> c_int {
    if let Some(manager) = SYNC_MANAGER.get() {
        if let Ok(m) = manager.lock() {
            return match m.state() {
                SyncState::Idle => 0,
                SyncState::Discovering => 1,
                SyncState::Connecting => 2,
                SyncState::Pairing => 3,
                SyncState::Syncing => 4,
                SyncState::Completed => 5,
                SyncState::Error => 6,
            };
        }
    }
    -1
}

/// Start mDNS discovery
#[no_mangle]
pub extern "C" fn sync_start_discovery() -> c_int {
    let manager = match SYNC_MANAGER.get() {
        Some(m) => m,
        None => return SyncResultCode::NotInitialized as c_int,
    };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return SyncResultCode::Error as c_int,
    };

    rt.block_on(async {
        let mut m = match manager.lock() {
            Ok(m) => m,
            Err(_) => return SyncResultCode::Error as c_int,
        };

        match m.start_discovery().await {
            Ok(_) => SyncResultCode::Ok as c_int,
            Err(_) => SyncResultCode::Error as c_int,
        }
    })
}

/// Stop mDNS discovery
#[no_mangle]
pub extern "C" fn sync_stop_discovery() -> c_int {
    let manager = match SYNC_MANAGER.get() {
        Some(m) => m,
        None => return SyncResultCode::NotInitialized as c_int,
    };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return SyncResultCode::Error as c_int,
    };

    rt.block_on(async {
        let mut m = match manager.lock() {
            Ok(m) => m,
            Err(_) => return SyncResultCode::Error as c_int,
        };

        match m.stop_discovery().await {
            Ok(_) => SyncResultCode::Ok as c_int,
            Err(_) => SyncResultCode::Error as c_int,
        }
    })
}

/// Get number of discovered peers
#[no_mangle]
pub extern "C" fn sync_get_peer_count() -> c_int {
    let manager = match SYNC_MANAGER.get() {
        Some(m) => m,
        None => return -1,
    };

    let m = match manager.lock() {
        Ok(m) => m,
        Err(_) => return -1,
    };

    m.discovered_peers().len() as c_int
}

/// Get peer info by index
///
/// # Safety
/// - `out_peer` must be a valid pointer to FFIPeerInfo
#[no_mangle]
pub unsafe extern "C" fn sync_get_peer(index: c_int, out_peer: *mut FFIPeerInfo) -> c_int {
    if out_peer.is_null() {
        return SyncResultCode::InvalidArgument as c_int;
    }

    let manager = match SYNC_MANAGER.get() {
        Some(m) => m,
        None => return SyncResultCode::NotInitialized as c_int,
    };

    let m = match manager.lock() {
        Ok(m) => m,
        Err(_) => return SyncResultCode::Error as c_int,
    };

    let peers = m.discovered_peers();
    if index < 0 || index as usize >= peers.len() {
        return SyncResultCode::InvalidArgument as c_int;
    }

    let peer = &peers[index as usize];
    let ffi_peer = &mut *out_peer;

    ffi_peer.device_id = string_to_c_char(&peer.device_id.to_string());
    ffi_peer.device_name = string_to_c_char(&peer.device_name);
    ffi_peer.address = string_to_c_char(
        &peer
            .addresses
            .first()
            .map(|a| a.to_string())
            .unwrap_or_default(),
    );
    ffi_peer.port = peer.port;
    ffi_peer.is_paired = if peer.is_paired { 1 } else { 0 };

    SyncResultCode::Ok as c_int
}

/// Initiate pairing with a peer
///
/// # Safety
/// - `device_id` must be a valid null-terminated C string
/// - `out_code` must be a valid pointer to receive the verification code
#[no_mangle]
pub unsafe extern "C" fn sync_initiate_pairing(
    device_id: *const c_char,
    out_code: *mut *mut c_char,
) -> c_int {
    if device_id.is_null() || out_code.is_null() {
        return SyncResultCode::InvalidArgument as c_int;
    }

    let device_id_str = match CStr::from_ptr(device_id).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return SyncResultCode::InvalidArgument as c_int,
    };

    let manager = match SYNC_MANAGER.get() {
        Some(m) => m,
        None => return SyncResultCode::NotInitialized as c_int,
    };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return SyncResultCode::Error as c_int,
    };

    rt.block_on(async {
        let mut m = match manager.lock() {
            Ok(m) => m,
            Err(_) => return SyncResultCode::Error as c_int,
        };

        // Find peer info
        let peer = m
            .discovered_peers()
            .into_iter()
            .find(|p| p.device_id.as_str() == device_id_str);

        let peer = match peer {
            Some(p) => p,
            None => return SyncResultCode::InvalidArgument as c_int,
        };

        match m.initiate_pairing(&peer).await {
            Ok(code) => {
                *out_code = string_to_c_char(&code.code);
                SyncResultCode::Ok as c_int
            }
            Err(_) => SyncResultCode::Error as c_int,
        }
    })
}

/// Confirm pairing with verification code
///
/// # Safety
/// - `code` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn sync_confirm_pairing(code: *const c_char) -> c_int {
    if code.is_null() {
        return SyncResultCode::InvalidArgument as c_int;
    }

    let code_str = match CStr::from_ptr(code).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return SyncResultCode::InvalidArgument as c_int,
    };

    let manager = match SYNC_MANAGER.get() {
        Some(m) => m,
        None => return SyncResultCode::NotInitialized as c_int,
    };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return SyncResultCode::Error as c_int,
    };

    rt.block_on(async {
        let mut m = match manager.lock() {
            Ok(m) => m,
            Err(_) => return SyncResultCode::Error as c_int,
        };

        let verification_code = crate::pairing::VerificationCode::from_code(code_str);

        match m.confirm_pairing(&verification_code).await {
            Ok(_) => SyncResultCode::Ok as c_int,
            Err(SyncError::InvalidVerificationCode) => SyncResultCode::InvalidArgument as c_int,
            Err(_) => SyncResultCode::Error as c_int,
        }
    })
}

/// Sync with a paired device
///
/// # Safety
/// - `device_id` must be a valid null-terminated C string
/// - `out_stats` must be a valid pointer to FFISyncStats
#[no_mangle]
pub unsafe extern "C" fn sync_with_device(
    device_id: *const c_char,
    out_stats: *mut FFISyncStats,
) -> c_int {
    if device_id.is_null() || out_stats.is_null() {
        return SyncResultCode::InvalidArgument as c_int;
    }

    let device_id_str = match CStr::from_ptr(device_id).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return SyncResultCode::InvalidArgument as c_int,
    };

    let manager = match SYNC_MANAGER.get() {
        Some(m) => m,
        None => return SyncResultCode::NotInitialized as c_int,
    };

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return SyncResultCode::Error as c_int,
    };

    rt.block_on(async {
        let mut m = match manager.lock() {
            Ok(m) => m,
            Err(_) => return SyncResultCode::Error as c_int,
        };

        let device_id = DeviceId::from_string(device_id_str);

        match m.sync_with(&device_id).await {
            Ok(stats) => {
                let ffi_stats = &mut *out_stats;
                ffi_stats.records_sent = stats.records_sent;
                ffi_stats.records_received = stats.records_received;
                ffi_stats.bytes_sent = stats.bytes_sent;
                ffi_stats.bytes_received = stats.bytes_received;
                ffi_stats.conflicts_resolved = stats.conflicts_resolved;
                ffi_stats.duration_ms = stats.duration_ms;
                SyncResultCode::Ok as c_int
            }
            Err(SyncError::NotPaired) => SyncResultCode::NotPaired as c_int,
            Err(_) => SyncResultCode::Error as c_int,
        }
    })
}

/// Free a string allocated by FFI functions
///
/// # Safety
/// - `s` must be a valid pointer returned by an FFI function or null
#[no_mangle]
pub unsafe extern "C" fn sync_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Free peer info allocated by FFI functions
///
/// # Safety
/// - `peer` must be a valid pointer to FFIPeerInfo allocated by FFI or null
#[no_mangle]
pub unsafe extern "C" fn sync_free_peer_info(peer: *mut FFIPeerInfo) {
    if !peer.is_null() {
        let p = &*peer;
        sync_free_string(p.device_id);
        sync_free_string(p.device_name);
        sync_free_string(p.address);
    }
}

/// Helper to convert Rust String to C char*
fn string_to_c_char(s: &str) -> *mut c_char {
    CString::new(s)
        .map(|cs| cs.into_raw())
        .unwrap_or(ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_codes() {
        assert_eq!(SyncResultCode::Ok as c_int, 0);
        assert_eq!(SyncResultCode::Error as c_int, -1);
    }

    #[test]
    fn test_string_conversion() {
        let s = "test string";
        let c_str = string_to_c_char(s);
        assert!(!c_str.is_null());

        unsafe {
            let back = CStr::from_ptr(c_str).to_str().unwrap();
            assert_eq!(back, s);
            sync_free_string(c_str);
        }
    }
}
