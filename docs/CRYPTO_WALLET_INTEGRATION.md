# Gazillioner Private - Crypto Wallet Integration

**Bitcoin Cold Wallet + Stablecoin Payments**

The same security architecture that protects your financial AI also protects your crypto.

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GAZILLIONER PRIVATE DEVICE                               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         FINANCIAL AI                                    │ │
│  │   FQ Assessment │ Portfolio Analysis │ AI Coaching │ Planning Tools    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         CRYPTO WALLET                                   │ │
│  │                                                                         │ │
│  │   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────┐  │ │
│  │   │  BITCOIN VAULT  │   │  STABLECOIN     │   │  PAYMENT ENGINE     │  │ │
│  │   │  (Cold Storage) │   │  WALLET         │   │                     │  │ │
│  │   │                 │   │                 │   │  • Pay subscriptions │  │ │
│  │   │  • BTC          │   │  • USDC         │   │  • Receive payments  │  │ │
│  │   │  • Offline sign │   │  • USDT         │   │  • Send to contacts  │  │ │
│  │   │  • Multi-sig    │   │  • DAI          │   │  • Invoice creation  │  │ │
│  │   └─────────────────┘   └─────────────────┘   └─────────────────────┘  │ │
│  │                                                                         │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │   │                    SECURE KEY STORAGE                            │  │ │
│  │   │                                                                   │  │ │
│  │   │   Keys stored in:                                                │  │ │
│  │   │   • Hardware TPM (if available)                                  │  │ │
│  │   │   • Encrypted file (AES-256-GCM)                                 │  │ │
│  │   │   • Optional: External hardware wallet (Ledger/Trezor)           │  │ │
│  │   │                                                                   │  │ │
│  │   │   NEVER exposed:                                                  │  │ │
│  │   │   • Private keys never leave secure enclave                      │  │ │
│  │   │   • Transactions signed locally                                  │  │ │
│  │   │   • No cloud backup of keys                                      │  │ │
│  │   └─────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    BLOCKCHAIN GATEWAY (Read-Only)                       │ │
│  │                                                                         │ │
│  │   ALLOWED (GET only):              BLOCKED:                            │ │
│  │   ✓ Check balances                 ✗ Private keys                      │ │
│  │   ✓ Fetch UTXOs                    ✗ Seed phrases                      │ │
│  │   ✓ Broadcast signed tx            ✗ Wallet metadata                   │ │
│  │   ✓ Get gas prices                 ✗ Transaction history export        │ │
│  │   ✓ Verify confirmations           ✗ Address clustering data           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Bitcoin Cold Wallet Features

### 1. Key Generation (Air-Gapped)

```
┌─────────────────────────────────────────────────────────────┐
│                   KEY GENERATION                             │
│                                                              │
│   1. Disconnect from network (air-gap mode)                 │
│   2. Generate entropy from:                                 │
│      - Hardware RNG (TPM/Secure Enclave)                   │
│      - User dice rolls (optional, for paranoid)            │
│      - Mouse movements + timing                             │
│   3. Create BIP-39 seed phrase (24 words)                  │
│   4. Derive BIP-84 addresses (native SegWit)               │
│   5. Encrypt with user passphrase (Argon2id + AES-256)     │
│   6. Store in secure partition                              │
│                                                              │
│   Keys NEVER touch network. Ever.                           │
└─────────────────────────────────────────────────────────────┘
```

### 2. Transaction Signing Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         BITCOIN TRANSACTION FLOW                          │
│                                                                           │
│   ONLINE PHASE (fetch data only):                                        │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│   │ User enters  │ ───▶ │ Fetch UTXOs  │ ───▶ │ Build PSBT   │          │
│   │ recipient +  │      │ from node    │      │ (unsigned)   │          │
│   │ amount       │      │              │      │              │          │
│   └──────────────┘      └──────────────┘      └──────────────┘          │
│                                                       │                   │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─   │
│                                                       ▼                   │
│   OFFLINE PHASE (air-gapped):                                            │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│   │ Display tx   │ ───▶ │ User confirms│ ───▶ │ Sign PSBT    │          │
│   │ details      │      │ on device    │      │ with cold key│          │
│   │ (verify!)    │      │              │      │              │          │
│   └──────────────┘      └──────────────┘      └──────────────┘          │
│                                                       │                   │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─   │
│                                                       ▼                   │
│   ONLINE PHASE (broadcast only):                                         │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│   │ Signed PSBT  │ ───▶ │ Finalize tx  │ ───▶ │ Broadcast to │          │
│   │              │      │              │      │ Bitcoin node │          │
│   └──────────────┘      └──────────────┘      └──────────────┘          │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3. Multi-Signature Support

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTI-SIG OPTIONS                         │
│                                                              │
│   2-of-3 Setup (Recommended):                               │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐    │
│   │ Key 1         │ │ Key 2         │ │ Key 3         │    │
│   │ Gazillioner   │ │ Hardware      │ │ Paper backup  │    │
│   │ Private       │ │ wallet        │ │ (in safe)     │    │
│   │ Device        │ │ (Ledger)      │ │               │    │
│   └───────────────┘ └───────────────┘ └───────────────┘    │
│                                                              │
│   Benefits:                                                  │
│   • No single point of failure                              │
│   • Device loss doesn't mean fund loss                      │
│   • Extra security for large amounts                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Stablecoin Wallet Features

### Supported Stablecoins

| Token | Network | Use Case |
|-------|---------|----------|
| **USDC** | Ethereum, Base, Solana | Primary payments |
| **USDT** | Ethereum, Tron | High liquidity |
| **DAI** | Ethereum | Decentralized option |
| **PYUSD** | Ethereum | PayPal ecosystem |

### Payment Use Cases

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STABLECOIN USE CASES                              │
│                                                                          │
│   1. PAY FOR GAZILLIONER SERVICES                                       │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  Gazillioner Plus: 10 USDC/month                                 │  │
│   │  Gazillioner Pro:  25 USDC/month                                 │  │
│   │                                                                   │  │
│   │  Benefits:                                                        │  │
│   │  • No credit card fees (save 3%)                                 │  │
│   │  • Privacy (no bank involved)                                    │  │
│   │  • Global (no currency conversion)                               │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   2. RECEIVE PAYMENTS (for advisors/creators)                           │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  Use Case: Financial content creator                             │  │
│   │                                                                   │  │
│   │  • Generate invoice: "Pay 50 USDC for 1hr consultation"         │  │
│   │  • Share payment link or QR code                                 │  │
│   │  • Receive funds directly to device wallet                       │  │
│   │  • Auto-convert to fiat (optional, via ramp)                    │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   3. SEND TO CONTACTS                                                    │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  "Send 100 USDC to mom for birthday"                             │  │
│   │                                                                   │  │
│   │  AI Integration:                                                  │  │
│   │  • "You're sending $100 to [Mom's wallet]"                       │  │
│   │  • "This is 2% of your monthly budget"                           │  │
│   │  • "Confirm with your passphrase"                                │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   4. DCA (Dollar Cost Average) INTO CRYPTO                              │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  Set up recurring:                                                │  │
│   │  • Every Monday, swap 50 USDC → BTC                              │  │
│   │  • Move BTC to cold storage automatically                        │  │
│   │  • AI tracks your average cost basis                             │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Implementation

### Wallet Architecture

```python
# /opt/gazillioner/backend/wallet/wallet_service.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import hashlib
import secrets

class Network(Enum):
    BITCOIN_MAINNET = "bitcoin"
    BITCOIN_TESTNET = "bitcoin_testnet"
    ETHEREUM = "ethereum"
    BASE = "base"
    SOLANA = "solana"

@dataclass
class WalletConfig:
    # Bitcoin
    btc_derivation_path: str = "m/84'/0'/0'"  # BIP-84 native SegWit
    btc_network: Network = Network.BITCOIN_MAINNET

    # Ethereum/EVM
    eth_derivation_path: str = "m/44'/60'/0'/0"  # BIP-44
    preferred_stablecoin: str = "USDC"
    preferred_network: Network = Network.BASE  # Low fees

    # Security
    require_passphrase: bool = True
    auto_lock_minutes: int = 5
    multi_sig_enabled: bool = False

@dataclass
class Transaction:
    tx_id: str
    from_address: str
    to_address: str
    amount: float
    asset: str
    network: Network
    status: str
    timestamp: float
    fee: float

class WalletService:
    """
    Secure wallet service for Gazillioner Private

    Security features:
    - Keys never leave secure enclave
    - All signing done locally
    - Air-gap mode supported
    """

    def __init__(self, config: WalletConfig):
        self.config = config
        self._locked = True
        self._key_store = None  # Loaded from encrypted storage

    # =========================================================================
    # Key Management
    # =========================================================================

    def generate_wallet(self, passphrase: str) -> dict:
        """
        Generate new HD wallet (BIP-39)
        MUST be called in air-gap mode!
        """
        # Generate 256 bits of entropy
        entropy = secrets.token_bytes(32)

        # Convert to mnemonic (24 words)
        from mnemonic import Mnemonic
        mnemo = Mnemonic("english")
        mnemonic = mnemo.to_mnemonic(entropy)

        # Derive master key
        seed = mnemo.to_seed(mnemonic, passphrase)

        # Store encrypted (never plaintext!)
        encrypted_seed = self._encrypt_seed(seed, passphrase)

        return {
            "mnemonic": mnemonic,  # Show once, user must backup
            "encrypted_seed_stored": True,
            "btc_address": self._derive_btc_address(seed),
            "eth_address": self._derive_eth_address(seed)
        }

    def unlock(self, passphrase: str) -> bool:
        """Unlock wallet with passphrase"""
        try:
            self._key_store = self._decrypt_seed(passphrase)
            self._locked = False
            return True
        except Exception:
            return False

    def lock(self):
        """Lock wallet, clear keys from memory"""
        self._key_store = None
        self._locked = True

    # =========================================================================
    # Bitcoin Operations
    # =========================================================================

    def get_btc_balance(self) -> dict:
        """Get Bitcoin balance (requires network)"""
        address = self._get_btc_address()

        # Fetch from node (read-only)
        utxos = self._fetch_utxos(address)

        confirmed = sum(u["value"] for u in utxos if u["confirmations"] >= 1)
        pending = sum(u["value"] for u in utxos if u["confirmations"] == 0)

        return {
            "address": address,
            "confirmed_sats": confirmed,
            "pending_sats": pending,
            "confirmed_btc": confirmed / 100_000_000,
            "pending_btc": pending / 100_000_000
        }

    def create_btc_transaction(
        self,
        to_address: str,
        amount_sats: int,
        fee_rate: int  # sats/vbyte
    ) -> dict:
        """
        Create unsigned Bitcoin transaction (PSBT)
        Returns PSBT for offline signing
        """
        if self._locked:
            raise WalletLocked("Wallet must be unlocked to create transactions")

        # Build PSBT
        utxos = self._select_utxos(amount_sats)
        psbt = self._build_psbt(utxos, to_address, amount_sats, fee_rate)

        return {
            "psbt_base64": psbt,
            "inputs": len(utxos),
            "amount_sats": amount_sats,
            "fee_sats": self._calculate_fee(psbt, fee_rate),
            "requires_signature": True
        }

    def sign_btc_transaction(self, psbt_base64: str) -> str:
        """
        Sign PSBT with cold key
        SHOULD be called in air-gap mode for maximum security!
        """
        if self._locked:
            raise WalletLocked("Wallet must be unlocked to sign")

        # Sign with private key (never exported)
        signed_psbt = self._sign_psbt(psbt_base64, self._key_store)

        return signed_psbt

    def broadcast_btc_transaction(self, signed_psbt: str) -> str:
        """Broadcast signed transaction to network"""
        # Finalize PSBT to raw tx
        raw_tx = self._finalize_psbt(signed_psbt)

        # Broadcast (requires network)
        tx_id = self._broadcast_to_node(raw_tx)

        return tx_id

    # =========================================================================
    # Stablecoin Operations
    # =========================================================================

    def get_stablecoin_balance(self, token: str = "USDC") -> dict:
        """Get stablecoin balance"""
        address = self._get_eth_address()

        balance = self._fetch_token_balance(address, token)

        return {
            "address": address,
            "token": token,
            "balance": balance,
            "network": self.config.preferred_network.value
        }

    def send_stablecoin(
        self,
        to_address: str,
        amount: float,
        token: str = "USDC"
    ) -> dict:
        """Send stablecoin payment"""
        if self._locked:
            raise WalletLocked("Wallet must be unlocked")

        # Build transaction
        tx = self._build_token_transfer(to_address, amount, token)

        # Sign locally
        signed_tx = self._sign_eth_transaction(tx)

        # Broadcast
        tx_hash = self._broadcast_eth_transaction(signed_tx)

        return {
            "tx_hash": tx_hash,
            "amount": amount,
            "token": token,
            "to": to_address,
            "status": "pending"
        }

    def create_payment_request(
        self,
        amount: float,
        token: str = "USDC",
        memo: str = ""
    ) -> dict:
        """Create payment request / invoice"""
        address = self._get_eth_address()

        # Generate payment URI
        uri = f"ethereum:{address}?amount={amount}&token={token}"

        return {
            "address": address,
            "amount": amount,
            "token": token,
            "memo": memo,
            "uri": uri,
            "qr_data": uri,  # For QR code generation
            "expires": None  # No expiration for stablecoins
        }


class WalletLocked(Exception):
    pass
```

### Blockchain Gateway (Whitelist)

```python
# /opt/gazillioner/backend/wallet/blockchain_gateway.py

BLOCKCHAIN_WHITELIST = {
    # Bitcoin
    "blockstream.info": {
        "methods": ["GET"],
        "paths": ["/api/address/", "/api/tx/"],
        "description": "Bitcoin block explorer API"
    },
    "mempool.space": {
        "methods": ["GET", "POST"],  # POST for broadcast only
        "paths": ["/api/address/", "/api/tx", "/api/fees/"],
        "description": "Mempool.space API"
    },

    # Ethereum / EVM
    "eth.llamarpc.com": {
        "methods": ["POST"],  # JSON-RPC
        "paths": ["/"],
        "description": "Ethereum RPC (read + broadcast)"
    },
    "base.llamarpc.com": {
        "methods": ["POST"],
        "paths": ["/"],
        "description": "Base RPC"
    },

    # Price data
    "api.coingecko.com": {
        "methods": ["GET"],
        "paths": ["/api/v3/simple/price"],
        "description": "Crypto prices"
    }
}

# NEVER ALLOWED - data that could leak
BLOCKED_PATTERNS = [
    "*seed*", "*mnemonic*", "*private*", "*secret*",
    "*export*", "*backup*", "*history*"
]
```

---

## Security Model

### Key Protection Layers

```
┌─────────────────────────────────────────────────────────────┐
│                   KEY PROTECTION LAYERS                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Layer 5: Physical Security                                 │
│   ┌──────────────────────────────────────────────────────┐  │
│   │ • Device in secure location                          │  │
│   │ • Optional: safe deposit box for backup              │  │
│   │ • Tamper-evident seals                               │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                              │
│   Layer 4: Encryption at Rest                                │
│   ┌──────────────────────────────────────────────────────┐  │
│   │ • LUKS full-disk encryption                          │  │
│   │ • Seed encrypted with Argon2id + AES-256-GCM         │  │
│   │ • Passphrase required on every unlock                │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                              │
│   Layer 3: Memory Protection                                 │
│   ┌──────────────────────────────────────────────────────┐  │
│   │ • Keys zeroed after use                              │  │
│   │ • No swap (memory locked)                            │  │
│   │ • Auto-lock after 5 minutes                          │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                              │
│   Layer 2: Network Isolation                                 │
│   ┌──────────────────────────────────────────────────────┐  │
│   │ • Whitelist-only outbound                            │  │
│   │ • No cloud backup                                    │  │
│   │ • Air-gap mode for signing                           │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                              │
│   Layer 1: Hardware Security (optional)                      │
│   ┌──────────────────────────────────────────────────────┐  │
│   │ • TPM for key storage                                │  │
│   │ • Secure boot                                        │  │
│   │ • External hardware wallet support                   │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Comparison: Gazillioner Private vs Other Wallets

| Feature | Gazillioner Private | Ledger | Software Wallet |
|---------|---------------------|--------|-----------------|
| Cold storage | ✓ | ✓ | ✗ |
| Air-gap signing | ✓ | ✗ | ✗ |
| Financial AI | ✓ | ✗ | ✗ |
| Portfolio analysis | ✓ | ✗ | Some |
| FQ coaching | ✓ | ✗ | ✗ |
| Stablecoin payments | ✓ | ✓ | ✓ |
| Multi-sig | ✓ | ✓ | Some |
| Open source | ✓ | Partial | Varies |
| Self-custody | ✓ | ✓ | ✓ |
| Price | $599 | $150 | Free |

---

## User Interface

### Wallet Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GAZILLIONER PRIVATE                                    🔒 Locked      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                         WALLET                                   │    │
│  │                                                                  │    │
│  │   Bitcoin (Cold Storage)                                         │    │
│  │   ┌────────────────────────────────────────────────────────┐    │    │
│  │   │  ₿ 0.5432 BTC                         ≈ $52,341        │    │    │
│  │   │  bc1q...xyz                           [Copy] [QR]      │    │    │
│  │   │                                                         │    │    │
│  │   │  [Receive]  [Send]  [History]                          │    │    │
│  │   └────────────────────────────────────────────────────────┘    │    │
│  │                                                                  │    │
│  │   Stablecoins (Hot Wallet)                                       │    │
│  │   ┌────────────────────────────────────────────────────────┐    │    │
│  │   │  💵 1,250.00 USDC                                       │    │    │
│  │   │  0x...abc (Base)                      [Copy] [QR]      │    │    │
│  │   │                                                         │    │    │
│  │   │  [Receive]  [Send]  [Pay Subscription]                 │    │    │
│  │   └────────────────────────────────────────────────────────┘    │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  💡 AI INSIGHT                                                   │    │
│  │                                                                  │    │
│  │  "Your BTC allocation is 45% of portfolio - above your 30%      │    │
│  │   target. Consider rebalancing $10k to stablecoins for your     │    │
│  │   emergency fund goal."                                          │    │
│  │                                                                  │    │
│  │  [Dismiss]  [Create Rebalance Plan]                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Payment Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PAY WITH STABLECOIN                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   To: Gazillioner Plus Subscription                                      │
│                                                                          │
│   Amount: 10.00 USDC                                                     │
│   Network: Base (fee: ~$0.01)                                            │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │   Your balance:  1,250.00 USDC                                  │   │
│   │   After payment: 1,240.00 USDC                                  │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Enter passphrase to confirm:                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ ••••••••••••                                                    │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│                        [Cancel]  [Confirm Payment]                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Revenue Model with Crypto

### Subscription Tiers (Crypto Payment)

| Tier | Monthly (USDC) | Annual (USDC) | Discount |
|------|----------------|---------------|----------|
| Free | $0 | $0 | - |
| Plus | 10 USDC | 100 USDC | 17% off |
| Pro | 25 USDC | 250 USDC | 17% off |
| Advisor | 99 USDC | 990 USDC | 17% off |

### Benefits of Crypto Payments

| Benefit | Value |
|---------|-------|
| No credit card fees | Save 2.9% + $0.30 |
| No chargebacks | Eliminate fraud losses |
| Global customers | No currency conversion |
| Privacy | No bank account required |
| Recurring payments | Smart contract automation |

---

## Roadmap

### Phase 1: Basic Wallet (Q1 2025)
- [ ] BTC receive/display
- [ ] USDC on Base receive/send
- [ ] Basic payment flow

### Phase 2: Cold Storage (Q2 2025)
- [ ] Air-gap transaction signing
- [ ] Multi-sig support
- [ ] Hardware wallet integration

### Phase 3: DeFi Integration (Q3 2025)
- [ ] Swap USDC ↔ BTC (via DEX)
- [ ] Yield on idle stablecoins
- [ ] DCA automation

### Phase 4: Advanced (Q4 2025)
- [ ] Lightning Network
- [ ] Nostr integration
- [ ] Cross-chain bridges

---

*Document Version: 1.0*
*Last Updated: 2025-12-28*
