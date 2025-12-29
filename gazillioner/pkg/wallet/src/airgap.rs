//! Air-Gapped Signing Support
//!
//! Implements QR code based air-gapped transaction signing for
//! maximum security. Transactions are prepared on an online device,
//! transferred via QR code to the air-gapped signing device,
//! signed, and the signed transaction is transferred back via QR code.

use qrcode::{QrCode, EcLevel};
use qrcode::render::unicode;
use serde::{Deserialize, Serialize};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

use crate::signing::{SigningRequest, SignedTransaction};
use crate::error::{Result, WalletError};

/// Maximum data size for a single QR code (with error correction)
const MAX_QR_BYTES: usize = 2953; // Version 40, Low EC

/// Air-gap request containing transaction data for offline signing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirGapRequest {
    /// Protocol version
    pub version: u8,
    /// Request ID for matching response
    pub request_id: String,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
    /// The signing request
    pub signing_request: SigningRequest,
    /// Optional metadata
    pub metadata: Option<AirGapMetadata>,
}

/// Metadata about the transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirGapMetadata {
    /// Human-readable description
    pub description: Option<String>,
    /// Sender address (for display)
    pub from_address: Option<String>,
    /// Recipient address (for display)
    pub to_address: Option<String>,
    /// Amount in human-readable format
    pub amount: Option<String>,
    /// Fee in human-readable format
    pub fee: Option<String>,
}

impl AirGapRequest {
    /// Create new air-gap request
    pub fn new(signing_request: SigningRequest) -> Self {
        Self {
            version: 1,
            request_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            signing_request,
            metadata: None,
        }
    }

    /// Add metadata for display
    pub fn with_metadata(mut self, metadata: AirGapMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Encode to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| WalletError::Serialization(e.to_string()))
    }

    /// Decode from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| WalletError::Serialization(e.to_string()))
    }

    /// Encode to base64 for QR code
    pub fn to_base64(&self) -> Result<String> {
        let json = self.to_json()?;
        Ok(BASE64.encode(json.as_bytes()))
    }

    /// Decode from base64
    pub fn from_base64(b64: &str) -> Result<Self> {
        let bytes = BASE64.decode(b64)
            .map_err(|e| WalletError::AirGapDecodingFailed(e.to_string()))?;
        let json = String::from_utf8(bytes)
            .map_err(|e| WalletError::AirGapDecodingFailed(e.to_string()))?;
        Self::from_json(&json)
    }

    /// Generate QR code(s) for the request
    /// Returns a list of QR codes if the data needs to be split
    pub fn to_qr_codes(&self) -> Result<Vec<QrCodeData>> {
        let data = self.to_base64()?;

        if data.len() <= MAX_QR_BYTES {
            // Single QR code
            let qr = QrCodeData::new(&data, 0, 1)?;
            Ok(vec![qr])
        } else {
            // Split into multiple QR codes
            let chunks: Vec<&str> = data
                .as_bytes()
                .chunks(MAX_QR_BYTES - 50) // Leave room for header
                .map(|chunk| std::str::from_utf8(chunk).unwrap())
                .collect();

            let total = chunks.len();
            chunks
                .into_iter()
                .enumerate()
                .map(|(i, chunk)| QrCodeData::new(chunk, i, total))
                .collect()
        }
    }
}

/// Air-gap response containing signed transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirGapResponse {
    /// Protocol version
    pub version: u8,
    /// Request ID this is responding to
    pub request_id: String,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
    /// The signed transaction
    pub signed_transaction: SignedTransaction,
    /// Signer fingerprint (for verification)
    pub signer_fingerprint: String,
}

impl AirGapResponse {
    /// Create new air-gap response
    pub fn new(
        request_id: &str,
        signed_transaction: SignedTransaction,
        signer_fingerprint: &str,
    ) -> Self {
        Self {
            version: 1,
            request_id: request_id.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            signed_transaction,
            signer_fingerprint: signer_fingerprint.to_string(),
        }
    }

    /// Encode to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self)
            .map_err(|e| WalletError::Serialization(e.to_string()))
    }

    /// Decode from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| WalletError::Serialization(e.to_string()))
    }

    /// Encode to base64 for QR code
    pub fn to_base64(&self) -> Result<String> {
        let json = self.to_json()?;
        Ok(BASE64.encode(json.as_bytes()))
    }

    /// Decode from base64
    pub fn from_base64(b64: &str) -> Result<Self> {
        let bytes = BASE64.decode(b64)
            .map_err(|e| WalletError::AirGapDecodingFailed(e.to_string()))?;
        let json = String::from_utf8(bytes)
            .map_err(|e| WalletError::AirGapDecodingFailed(e.to_string()))?;
        Self::from_json(&json)
    }

    /// Generate QR code for the response
    pub fn to_qr_code(&self) -> Result<QrCodeData> {
        let data = self.to_base64()?;
        QrCodeData::new(&data, 0, 1)
    }
}

/// QR code data with rendering capability
#[derive(Debug, Clone)]
pub struct QrCodeData {
    /// The data encoded in the QR code
    pub data: String,
    /// Part index (0-based)
    pub part_index: usize,
    /// Total number of parts
    pub total_parts: usize,
    /// The QR code object
    code: QrCode,
}

impl QrCodeData {
    /// Create a new QR code
    pub fn new(data: &str, part_index: usize, total_parts: usize) -> Result<Self> {
        // Add header for multi-part QR codes
        let encoded_data = if total_parts > 1 {
            format!("GP{}of{}:{}", part_index + 1, total_parts, data)
        } else {
            data.to_string()
        };

        let code = QrCode::with_error_correction_level(&encoded_data, EcLevel::M)
            .map_err(|e| WalletError::QrCodeFailed(e.to_string()))?;

        Ok(Self {
            data: encoded_data,
            part_index,
            total_parts,
            code,
        })
    }

    /// Render as Unicode string (for terminal display)
    pub fn render_unicode(&self) -> String {
        self.code
            .render::<unicode::Dense1x2>()
            .dark_color(unicode::Dense1x2::Light)
            .light_color(unicode::Dense1x2::Dark)
            .build()
    }

    /// Render as ASCII art
    pub fn render_ascii(&self) -> String {
        let mut result = String::new();
        let width = self.code.width();

        for y in 0..width {
            for x in 0..width {
                let module = self.code[(x, y)];
                if module == qrcode::Color::Dark {
                    result.push_str("██");
                } else {
                    result.push_str("  ");
                }
            }
            result.push('\n');
        }

        result
    }

    /// Get raw module data (for custom rendering)
    pub fn modules(&self) -> Vec<Vec<bool>> {
        let width = self.code.width();
        (0..width)
            .map(|y| {
                (0..width)
                    .map(|x| self.code[(x, y)] == qrcode::Color::Dark)
                    .collect()
            })
            .collect()
    }
}

/// Reassemble multi-part QR code data
pub struct QrCodeAssembler {
    parts: Vec<Option<String>>,
    total_parts: usize,
}

impl QrCodeAssembler {
    /// Create new assembler for expected number of parts
    pub fn new(total_parts: usize) -> Self {
        Self {
            parts: vec![None; total_parts],
            total_parts,
        }
    }

    /// Add a scanned part
    pub fn add_part(&mut self, data: &str) -> Result<()> {
        // Parse header
        if data.starts_with("GP") {
            let header_end = data.find(':').ok_or_else(||
                WalletError::AirGapDecodingFailed("Invalid multi-part header".into())
            )?;
            let header = &data[2..header_end];
            let parts: Vec<&str> = header.split("of").collect();
            if parts.len() != 2 {
                return Err(WalletError::AirGapDecodingFailed("Invalid part format".into()));
            }

            let part_num: usize = parts[0].parse()
                .map_err(|_| WalletError::AirGapDecodingFailed("Invalid part number".into()))?;
            let total: usize = parts[1].parse()
                .map_err(|_| WalletError::AirGapDecodingFailed("Invalid total".into()))?;

            if total != self.total_parts {
                return Err(WalletError::AirGapDecodingFailed(
                    format!("Part count mismatch: expected {}, got {}", self.total_parts, total)
                ));
            }

            let content = &data[header_end + 1..];
            self.parts[part_num - 1] = Some(content.to_string());
        } else {
            // Single-part QR code
            if self.total_parts != 1 {
                return Err(WalletError::AirGapDecodingFailed(
                    "Expected multi-part QR code".into()
                ));
            }
            self.parts[0] = Some(data.to_string());
        }

        Ok(())
    }

    /// Check if all parts have been received
    pub fn is_complete(&self) -> bool {
        self.parts.iter().all(|p| p.is_some())
    }

    /// Get number of missing parts
    pub fn missing_count(&self) -> usize {
        self.parts.iter().filter(|p| p.is_none()).count()
    }

    /// Assemble the complete data
    pub fn assemble(&self) -> Result<String> {
        if !self.is_complete() {
            return Err(WalletError::AirGapDecodingFailed(
                format!("Missing {} parts", self.missing_count())
            ));
        }

        Ok(self.parts.iter()
            .filter_map(|p| p.as_ref())
            .cloned()
            .collect())
    }
}

/// Display helper for showing transaction details before signing
pub struct TransactionDisplay {
    pub coin: String,
    pub from: String,
    pub to: String,
    pub amount: String,
    pub fee: String,
    pub network: String,
}

impl TransactionDisplay {
    /// Format for terminal display
    pub fn format(&self) -> String {
        format!(
            r#"
╔════════════════════════════════════════════════════════════╗
║                    TRANSACTION DETAILS                     ║
╠════════════════════════════════════════════════════════════╣
║  Network:  {:<47} ║
║  Coin:     {:<47} ║
╠════════════════════════════════════════════════════════════╣
║  From:     {:<47} ║
║  To:       {:<47} ║
╠════════════════════════════════════════════════════════════╣
║  Amount:   {:<47} ║
║  Fee:      {:<47} ║
╚════════════════════════════════════════════════════════════╝
"#,
            self.network, self.coin, self.from, self.to, self.amount, self.fee
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eth::EthereumTransaction;
    use crate::Network;

    #[test]
    fn test_airgap_request_serialization() {
        let tx = EthereumTransaction::transfer(
            1,
            0,
            "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD20",
            ethereum_types::U256::from(1000000000000000000u64),
            ethereum_types::U256::from(20000000000u64),
            21000,
        );

        let request = AirGapRequest::new(
            crate::signing::SigningRequest::ethereum(tx, 0, Network::Ethereum)
        );

        let json = request.to_json().unwrap();
        let parsed = AirGapRequest::from_json(&json).unwrap();

        assert_eq!(parsed.request_id, request.request_id);
    }

    #[test]
    fn test_qr_code_generation() {
        let tx = EthereumTransaction::transfer(
            1,
            0,
            "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD20",
            ethereum_types::U256::from(1000000000000000000u64),
            ethereum_types::U256::from(20000000000u64),
            21000,
        );

        let request = AirGapRequest::new(
            crate::signing::SigningRequest::ethereum(tx, 0, Network::Ethereum)
        );

        let qr_codes = request.to_qr_codes().unwrap();
        assert!(!qr_codes.is_empty());

        // Should render without error
        let _ascii = qr_codes[0].render_ascii();
        let _unicode = qr_codes[0].render_unicode();
    }

    #[test]
    fn test_qr_assembler() {
        let mut assembler = QrCodeAssembler::new(1);
        assert!(!assembler.is_complete());
        assert_eq!(assembler.missing_count(), 1);

        assembler.add_part("test-data").unwrap();
        assert!(assembler.is_complete());

        let result = assembler.assemble().unwrap();
        assert_eq!(result, "test-data");
    }
}
