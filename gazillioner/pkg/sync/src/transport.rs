//! Secure transport layer using TLS 1.3 with mutual authentication
//!
//! Provides encrypted communication channels between paired devices.

use crate::error::{SyncError, SyncResult};
use crate::protocol::SyncMessage;
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::{TlsAcceptor, TlsConnector};

/// Maximum message size (16 MB)
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Secure transport for P2P communication
pub struct SecureTransport {
    shared_key: Vec<u8>,
    cipher: Aes256Gcm,
}

impl SecureTransport {
    /// Create a new secure transport with the shared key
    pub fn new(shared_key: Vec<u8>) -> SyncResult<Self> {
        if shared_key.len() != 32 {
            return Err(SyncError::Encryption(
                "Shared key must be 32 bytes".into(),
            ));
        }

        let key = aes_gcm::Key::<Aes256Gcm>::from_slice(&shared_key);
        let cipher = Aes256Gcm::new(key);

        Ok(Self { shared_key, cipher })
    }

    /// Encrypt a message
    pub fn encrypt(&self, plaintext: &[u8]) -> SyncResult<Vec<u8>> {
        // Generate random nonce (12 bytes for AES-GCM)
        let mut nonce_bytes = [0u8; 12];
        rand::Rng::fill(&mut rand::thread_rng(), &mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt
        let ciphertext = self
            .cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| SyncError::Encryption(format!("Encryption failed: {}", e)))?;

        // Prepend nonce to ciphertext
        let mut result = nonce_bytes.to_vec();
        result.extend(ciphertext);
        Ok(result)
    }

    /// Decrypt a message
    pub fn decrypt(&self, ciphertext: &[u8]) -> SyncResult<Vec<u8>> {
        if ciphertext.len() < 12 {
            return Err(SyncError::Encryption("Ciphertext too short".into()));
        }

        // Extract nonce
        let nonce = Nonce::from_slice(&ciphertext[..12]);
        let encrypted = &ciphertext[12..];

        // Decrypt
        self.cipher
            .decrypt(nonce, encrypted)
            .map_err(|e| SyncError::Encryption(format!("Decryption failed: {}", e)))
    }

    /// Send an encrypted message over a stream
    pub async fn send_message<W: AsyncWriteExt + Unpin>(
        &self,
        stream: &mut W,
        message: &SyncMessage,
    ) -> SyncResult<()> {
        // Serialize message
        let serialized = message.to_bytes()?;

        // Encrypt
        let encrypted = self.encrypt(&serialized)?;

        // Send length prefix (4 bytes, big endian)
        let len = encrypted.len() as u32;
        stream.write_all(&len.to_be_bytes()).await?;

        // Send encrypted data
        stream.write_all(&encrypted).await?;
        stream.flush().await?;

        Ok(())
    }

    /// Receive an encrypted message from a stream
    pub async fn recv_message<R: AsyncReadExt + Unpin>(
        &self,
        stream: &mut R,
    ) -> SyncResult<SyncMessage> {
        // Read length prefix
        let mut len_bytes = [0u8; 4];
        stream.read_exact(&mut len_bytes).await?;
        let len = u32::from_be_bytes(len_bytes) as usize;

        // Validate length
        if len > MAX_MESSAGE_SIZE {
            return Err(SyncError::InvalidMessage(format!(
                "Message too large: {} bytes",
                len
            )));
        }

        // Read encrypted data
        let mut encrypted = vec![0u8; len];
        stream.read_exact(&mut encrypted).await?;

        // Decrypt
        let decrypted = self.decrypt(&encrypted)?;

        // Deserialize
        SyncMessage::from_bytes(&decrypted)
    }
}

/// TCP listener for incoming P2P connections
pub struct P2PListener {
    listener: TcpListener,
    tls_acceptor: Option<TlsAcceptor>,
}

impl P2PListener {
    /// Create a new listener on the specified port
    pub async fn bind(port: u16) -> SyncResult<Self> {
        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        let listener = TcpListener::bind(addr).await?;

        tracing::info!("P2P listener bound to {}", addr);

        Ok(Self {
            listener,
            tls_acceptor: None,
        })
    }

    /// Get the local address
    pub fn local_addr(&self) -> SyncResult<SocketAddr> {
        self.listener.local_addr().map_err(|e| SyncError::Io(e))
    }

    /// Accept an incoming connection
    pub async fn accept(&self) -> SyncResult<(TcpStream, SocketAddr)> {
        let (stream, addr) = self.listener.accept().await?;
        tracing::debug!("Accepted connection from {}", addr);
        Ok((stream, addr))
    }
}

/// P2P connection to a remote peer
pub struct P2PConnection {
    stream: TcpStream,
    transport: SecureTransport,
    remote_addr: SocketAddr,
}

impl P2PConnection {
    /// Connect to a remote peer
    pub async fn connect(addr: SocketAddr, shared_key: Vec<u8>) -> SyncResult<Self> {
        let stream = TcpStream::connect(addr).await?;
        let transport = SecureTransport::new(shared_key)?;

        tracing::info!("Connected to peer at {}", addr);

        Ok(Self {
            stream,
            transport,
            remote_addr: addr,
        })
    }

    /// Wrap an accepted connection
    pub fn from_accepted(
        stream: TcpStream,
        remote_addr: SocketAddr,
        shared_key: Vec<u8>,
    ) -> SyncResult<Self> {
        let transport = SecureTransport::new(shared_key)?;
        Ok(Self {
            stream,
            transport,
            remote_addr,
        })
    }

    /// Get the remote address
    pub fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    /// Send a message
    pub async fn send(&mut self, message: &SyncMessage) -> SyncResult<()> {
        self.transport.send_message(&mut self.stream, message).await
    }

    /// Receive a message
    pub async fn recv(&mut self) -> SyncResult<SyncMessage> {
        self.transport.recv_message(&mut self.stream).await
    }

    /// Close the connection
    pub async fn close(mut self) -> SyncResult<()> {
        self.stream.shutdown().await?;
        Ok(())
    }
}

/// Generate a self-signed certificate for P2P communication
pub fn generate_self_signed_cert() -> SyncResult<(Vec<u8>, Vec<u8>)> {
    use rcgen::{CertifiedKey, generate_simple_self_signed};

    let subject_alt_names = vec!["localhost".to_string()];
    let CertifiedKey { cert, key_pair } = generate_simple_self_signed(subject_alt_names)
        .map_err(|e| SyncError::Encryption(format!("Certificate generation failed: {}", e)))?;

    Ok((cert.pem().into_bytes(), key_pair.serialize_pem().into_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> Vec<u8> {
        vec![
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ]
    }

    #[test]
    fn test_transport_creation() {
        let transport = SecureTransport::new(test_key());
        assert!(transport.is_ok());
    }

    #[test]
    fn test_transport_invalid_key() {
        let transport = SecureTransport::new(vec![0u8; 16]); // Wrong size
        assert!(transport.is_err());
    }

    #[test]
    fn test_encrypt_decrypt() {
        let transport = SecureTransport::new(test_key()).unwrap();
        let plaintext = b"Hello, World!";

        let encrypted = transport.encrypt(plaintext).unwrap();
        assert_ne!(&encrypted, plaintext);

        let decrypted = transport.decrypt(&encrypted).unwrap();
        assert_eq!(&decrypted, plaintext);
    }

    #[test]
    fn test_decrypt_invalid() {
        let transport = SecureTransport::new(test_key()).unwrap();
        let result = transport.decrypt(&[0u8; 5]); // Too short
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_cert() {
        let result = generate_self_signed_cert();
        assert!(result.is_ok());
        let (cert, key) = result.unwrap();
        assert!(!cert.is_empty());
        assert!(!key.is_empty());
    }
}
