//! mDNS service discovery for local network peer detection
//!
//! Uses DNS-SD (DNS Service Discovery) to advertise and find Gazillioner
//! instances on the local network.

use crate::error::{SyncError, SyncResult};
use crate::{DeviceId, DeviceInfo, PeerInfo, SERVICE_NAME};
use chrono::Utc;
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

/// Service discovery manager for finding peers on the local network
pub struct ServiceDiscovery {
    device_info: DeviceInfo,
    port: u16,
    daemon: Option<ServiceDaemon>,
    peers: Arc<RwLock<HashMap<DeviceId, PeerInfo>>>,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl ServiceDiscovery {
    /// Create a new service discovery instance
    pub fn new(device_info: DeviceInfo, port: u16) -> SyncResult<Self> {
        Ok(Self {
            device_info,
            port,
            daemon: None,
            peers: Arc::new(RwLock::new(HashMap::new())),
            shutdown_tx: None,
        })
    }

    /// Start advertising this device and browsing for peers
    pub async fn start(&self) -> SyncResult<()> {
        // Create mDNS daemon
        let daemon = ServiceDaemon::new()
            .map_err(|e| SyncError::Discovery(format!("Failed to create mDNS daemon: {}", e)))?;

        // Register this device as a service
        let instance_name = format!(
            "{}.{}",
            self.device_info.device_id.as_str(),
            SERVICE_NAME
        );

        let mut properties = HashMap::new();
        properties.insert("device_name".to_string(), self.device_info.device_name.clone());
        properties.insert("platform".to_string(), self.device_info.platform.clone());
        properties.insert("app_version".to_string(), self.device_info.app_version.clone());
        properties.insert(
            "protocol_version".to_string(),
            self.device_info.protocol_version.to_string(),
        );

        let service_info = ServiceInfo::new(
            SERVICE_NAME,
            &instance_name,
            &format!("{}.local.", self.device_info.device_id.as_str()),
            "",
            self.port,
            properties,
        )
        .map_err(|e| SyncError::Discovery(format!("Failed to create service info: {}", e)))?;

        daemon
            .register(service_info)
            .map_err(|e| SyncError::Discovery(format!("Failed to register service: {}", e)))?;

        // Browse for other devices
        let receiver = daemon
            .browse(SERVICE_NAME)
            .map_err(|e| SyncError::Discovery(format!("Failed to browse: {}", e)))?;

        // Clone for the background task
        let peers = self.peers.clone();
        let our_device_id = self.device_info.device_id.clone();

        // Spawn background task to handle discovery events
        tokio::spawn(async move {
            loop {
                match receiver.recv() {
                    Ok(event) => {
                        Self::handle_event(event, &peers, &our_device_id);
                    }
                    Err(_) => {
                        // Channel closed, exit
                        break;
                    }
                }
            }
        });

        tracing::info!(
            "Started mDNS discovery for device {}",
            self.device_info.device_id
        );

        Ok(())
    }

    /// Handle mDNS discovery events
    fn handle_event(
        event: ServiceEvent,
        peers: &Arc<RwLock<HashMap<DeviceId, PeerInfo>>>,
        our_device_id: &DeviceId,
    ) {
        match event {
            ServiceEvent::ServiceResolved(info) => {
                // Extract device ID from service name
                let full_name = info.get_fullname();
                let device_id_str = full_name
                    .split('.')
                    .next()
                    .unwrap_or("")
                    .to_string();

                // Skip our own device
                if device_id_str == our_device_id.as_str() {
                    return;
                }

                let device_id = DeviceId::from_string(device_id_str.clone());

                // Get device name from properties
                let device_name = info
                    .get_properties()
                    .get("device_name")
                    .map(|v| v.val_str().to_string())
                    .unwrap_or_else(|| format!("Unknown Device ({})", &device_id_str[..8.min(device_id_str.len())]));

                // Get addresses
                let addresses: Vec<IpAddr> = info
                    .get_addresses()
                    .iter()
                    .cloned()
                    .collect();

                if addresses.is_empty() {
                    tracing::debug!("Discovered device {} but no addresses available", device_id_str);
                    return;
                }

                let peer_info = PeerInfo {
                    device_id: device_id.clone(),
                    device_name,
                    addresses,
                    port: info.get_port(),
                    discovered_at: Utc::now(),
                    is_paired: false, // Will be updated by sync manager
                };

                tracing::info!(
                    "Discovered peer: {} at {:?}:{}",
                    peer_info.device_name,
                    peer_info.addresses,
                    peer_info.port
                );

                // Store the peer
                if let Ok(mut peers) = peers.write() {
                    peers.insert(device_id, peer_info);
                }
            }
            ServiceEvent::ServiceRemoved(_type_name, full_name) => {
                // Extract device ID from service name
                let device_id_str = full_name
                    .split('.')
                    .next()
                    .unwrap_or("")
                    .to_string();

                let device_id = DeviceId::from_string(device_id_str.clone());

                if let Ok(mut peers) = peers.write() {
                    if let Some(peer) = peers.remove(&device_id) {
                        tracing::info!("Peer removed: {}", peer.device_name);
                    }
                }
            }
            ServiceEvent::SearchStarted(_) => {
                tracing::debug!("mDNS search started");
            }
            ServiceEvent::SearchStopped(_) => {
                tracing::debug!("mDNS search stopped");
            }
            _ => {}
        }
    }

    /// Stop discovery
    pub async fn stop(&self) -> SyncResult<()> {
        if let Some(daemon) = &self.daemon {
            daemon
                .shutdown()
                .map_err(|e| SyncError::Discovery(format!("Failed to shutdown daemon: {}", e)))?;
        }
        tracing::info!("Stopped mDNS discovery");
        Ok(())
    }

    /// Get list of discovered peers
    pub fn peers(&self) -> Vec<PeerInfo> {
        self.peers
            .read()
            .map(|p| p.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Get a specific peer by device ID
    pub fn get_peer(&self, device_id: &DeviceId) -> Option<PeerInfo> {
        self.peers
            .read()
            .ok()
            .and_then(|p| p.get(device_id).cloned())
    }

    /// Update paired status for a device
    pub fn set_paired(&self, device_id: &DeviceId, paired: bool) {
        if let Ok(mut peers) = self.peers.write() {
            if let Some(peer) = peers.get_mut(device_id) {
                peer.is_paired = paired;
            }
        }
    }

    /// Clear all discovered peers
    pub fn clear_peers(&self) {
        if let Ok(mut peers) = self.peers.write() {
            peers.clear();
        }
    }
}

impl Drop for ServiceDiscovery {
    fn drop(&mut self) {
        if let Some(daemon) = self.daemon.take() {
            let _ = daemon.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_discovery_new() {
        let device_info = DeviceInfo::new(
            DeviceId::new(),
            "Test Device".to_string(),
        );
        let discovery = ServiceDiscovery::new(device_info, 47392);
        assert!(discovery.is_ok());
    }

    #[test]
    fn test_peers_initially_empty() {
        let device_info = DeviceInfo::new(
            DeviceId::new(),
            "Test Device".to_string(),
        );
        let discovery = ServiceDiscovery::new(device_info, 47392).unwrap();
        assert!(discovery.peers().is_empty());
    }
}
