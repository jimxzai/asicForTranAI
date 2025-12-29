# Gazillioner Makefile
# Build and development commands for the Gazillioner CLI

.PHONY: all build clean test proto run install lint fmt help

# Build variables
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DATE ?= $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
LDFLAGS := -ldflags "-X main.version=$(VERSION) -X main.commit=$(COMMIT) -X main.date=$(DATE)"

# Go variables
GOCMD := go
GOBUILD := $(GOCMD) build
GOTEST := $(GOCMD) test
GORUN := $(GOCMD) run
GOGET := $(GOCMD) get
GOFMT := gofmt
GOLINT := golangci-lint

# Binary name
BINARY := gazillioner
BINARY_PATH := bin/$(BINARY)

# Proto variables
PROTO_DIR := api/proto
PROTO_OUT := api/v1
PROTOC := protoc

# Rust variables
CARGO := cargo
RUST_LIB := pkg/ffi/libgazillioner

# Default target
all: build

# Build the binary
build:
	@echo "Building $(BINARY)..."
	@mkdir -p bin
	CGO_ENABLED=1 $(GOBUILD) $(LDFLAGS) -o $(BINARY_PATH) ./cmd/gazillioner

# Build for release (optimized, stripped)
build-release:
	@echo "Building $(BINARY) for release..."
	@mkdir -p bin
	CGO_ENABLED=1 $(GOBUILD) $(LDFLAGS) -ldflags="-s -w" -o $(BINARY_PATH) ./cmd/gazillioner

# Build for all platforms
build-all: build-linux build-darwin

build-linux:
	@echo "Building for Linux..."
	@mkdir -p bin
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 $(GOBUILD) $(LDFLAGS) -o bin/$(BINARY)-linux-amd64 ./cmd/gazillioner
	GOOS=linux GOARCH=arm64 CGO_ENABLED=0 $(GOBUILD) $(LDFLAGS) -o bin/$(BINARY)-linux-arm64 ./cmd/gazillioner

build-darwin:
	@echo "Building for macOS..."
	@mkdir -p bin
	GOOS=darwin GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o bin/$(BINARY)-darwin-amd64 ./cmd/gazillioner
	GOOS=darwin GOARCH=arm64 $(GOBUILD) $(LDFLAGS) -o bin/$(BINARY)-darwin-arm64 ./cmd/gazillioner

# Run the application
run:
	$(GORUN) ./cmd/gazillioner $(ARGS)

# Install to GOPATH/bin
install:
	@echo "Installing $(BINARY)..."
	$(GOCMD) install $(LDFLAGS) ./cmd/gazillioner

# Generate protobuf code
proto:
	@echo "Generating protobuf code..."
	@mkdir -p $(PROTO_OUT)
	$(PROTOC) --proto_path=$(PROTO_DIR) \
		--go_out=$(PROTO_OUT) --go_opt=paths=source_relative \
		--go-grpc_out=$(PROTO_OUT) --go-grpc_opt=paths=source_relative \
		$(PROTO_DIR)/v1/*.proto

# Build Rust FFI library
rust:
	@echo "Building Rust FFI library..."
	cd $(RUST_LIB) && $(CARGO) build --release

# Run tests
test:
	@echo "Running tests..."
	$(GOTEST) -v -race -cover ./...

# Run tests with coverage report
test-coverage:
	@echo "Running tests with coverage..."
	$(GOTEST) -v -race -coverprofile=coverage.out ./...
	$(GOCMD) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report: coverage.html"

# Lint code
lint:
	@echo "Linting code..."
	$(GOLINT) run ./...

# Format code
fmt:
	@echo "Formatting code..."
	$(GOFMT) -s -w .

# Check formatting
fmt-check:
	@echo "Checking code formatting..."
	@test -z "$$($(GOFMT) -l .)" || (echo "Code is not formatted. Run 'make fmt'" && exit 1)

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -rf bin/
	rm -f coverage.out coverage.html
	cd $(RUST_LIB) 2>/dev/null && $(CARGO) clean || true

# Download dependencies
deps:
	@echo "Downloading dependencies..."
	$(GOGET) -v ./...
	$(GOCMD) mod tidy

# Verify dependencies
verify:
	@echo "Verifying dependencies..."
	$(GOCMD) mod verify

# Generate all code (proto, etc.)
generate: proto
	@echo "Generating code..."
	$(GOCMD) generate ./...

# Run in development mode with hot reload
dev:
	@echo "Starting development mode..."
	@which air > /dev/null || (echo "Install air: go install github.com/air-verse/air@latest" && exit 1)
	air

# Security audit
audit:
	@echo "Running security audit..."
	$(GOCMD) list -json -m all | nancy sleuth
	govulncheck ./...

# Help
help:
	@echo "Gazillioner Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make build          Build the binary"
	@echo "  make build-release  Build optimized release binary"
	@echo "  make run            Run the application"
	@echo "  make install        Install to GOPATH/bin"
	@echo "  make proto          Generate protobuf code"
	@echo "  make rust           Build Rust FFI library"
	@echo "  make test           Run tests"
	@echo "  make test-coverage  Run tests with coverage report"
	@echo "  make lint           Lint code"
	@echo "  make fmt            Format code"
	@echo "  make clean          Clean build artifacts"
	@echo "  make deps           Download dependencies"
	@echo "  make generate       Generate all code"
	@echo "  make dev            Run in development mode with hot reload"
	@echo "  make audit          Run security audit"
	@echo "  make help           Show this help"
