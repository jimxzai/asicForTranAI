# Gazillioner Backend

Self-Sovereign AI Financial Intelligence Platform

## Architecture

```
gazillioner/
├── cmd/gazillioner/        # Main entry point
├── internal/
│   ├── cmd/                # CLI commands (Cobra)
│   ├── service/            # gRPC service implementations
│   ├── db/                 # Database layer
│   └── tui/                # Terminal UI (Bubbletea)
├── api/proto/v1/           # Protocol Buffer definitions
├── pkg/
│   ├── crypto/             # Rust SQLCipher wrapper
│   └── ffi/                # CGO bindings
└── Makefile
```

## Building

### Prerequisites

- Go 1.22+
- Rust 1.75+
- Protocol Buffers compiler (protoc)
- SQLCipher development files

### Build Commands

```bash
# Build binary
make build

# Build for release (optimized)
make build-release

# Generate protobuf code
make proto

# Build Rust FFI library
cd pkg/crypto && cargo build --release

# Run tests
make test

# Run linter
make lint
```

## CLI Usage

```bash
# Launch interactive TUI
gazillioner

# AI Queries
gazillioner query "Analyze my portfolio risk"

# Portfolio Management
gazillioner portfolio list
gazillioner portfolio add AAPL 100 150.00
gazillioner portfolio import portfolio.csv

# Watchlist
gazillioner watchlist add NVDA

# Market Data
gazillioner market quote AAPL MSFT

# Wallet
gazillioner wallet balance

# System
gazillioner status
gazillioner config show
```

## Security Features

- **SQLCipher**: AES-256 encrypted database
- **Argon2id**: PIN-based key derivation
- **HMAC Integrity**: Audit log verification
- **Air-gapped Design**: No cloud dependencies

## Development

```bash
# Run in development mode
make dev

# Format code
make fmt

# Security audit
make audit
```

## License

Proprietary - Gazillioner Inc.
