# Gazillioner — MVP Definition & Scope

**Document Version:** 2.0
**Date:** December 2025
**Target Launch:** Q2 2026

---

## 1. MVP Philosophy

### 1.1 Guiding Principles

Our MVP follows the "skateboard, not wheel" philosophy — we ship a complete, usable product that delivers core value, not a partial feature set that requires future releases to be useful.

```
                    MVP EVOLUTION MODEL

    ❌ WRONG APPROACH (Partial Features)
    ┌─────────────────────────────────────────┐
    │  Sprint 1: Wheel                        │
    │  Sprint 2: Wheel + Axle                 │
    │  Sprint 3: Wheel + Axle + Frame         │
    │  Sprint 4: Complete Car                 │
    │                                         │
    │  Problem: No value until Sprint 4       │
    └─────────────────────────────────────────┘

    ✅ RIGHT APPROACH (Complete Value)
    ┌─────────────────────────────────────────┐
    │  MVP: Skateboard → Complete transport   │
    │  V1.1: Scooter → Better transport       │
    │  V2.0: Bicycle → Even better            │
    │  V3.0: Car → Full vision                │
    │                                         │
    │  Benefit: Value delivered at each stage │
    └─────────────────────────────────────────┘
```

### 1.2 MVP Success Criteria

The MVP is successful if:

1. **Functional:** User can ask AI questions about their portfolio and markets
2. **Self-Sovereign:** Zero data leaves the device; zero external dependencies for core function
3. **Performant:** Query response in <500ms feels instant to users
4. **Usable:** Setup to first insight in <15 minutes
5. **Valuable:** Users prefer it to cloud alternatives for privacy-sensitive queries
6. **Secure:** <100 total dependencies; financial-grade security architecture

### 1.3 What MVP is NOT

- **Not a demo:** This is a shippable product customers pay for
- **Not feature-complete:** We intentionally defer non-essential features
- **Not throwaway:** MVP code is production code, built to scale
- **Not perfect:** We ship with known limitations, clearly documented
- **Not a web app:** Native binaries, not browser-based

---

## 2. MVP Feature Scope

### 2.1 Feature Inclusion Matrix

| Feature | MVP | V1.1 | V2.0 | Rationale |
|---------|-----|------|------|-----------|
| **Setup & Configuration** |
| Initial device setup wizard | ✅ | | | Core requirement |
| Network configuration (Ethernet) | ✅ | | | Core requirement |
| WiFi configuration | ✅ | | | User convenience |
| Security PIN | ✅ | | | Data protection |
| Air-gap mode | | ✅ | | Power user feature |
| Firmware updates | | ✅ | | Can update via USB for MVP |
| Multi-device sync | | | ✅ | Enterprise feature |
| **Portfolio Management** |
| Manual portfolio entry | ✅ | | | Core requirement |
| CSV import | ✅ | | | User convenience |
| Watchlists | ✅ | | | Core requirement |
| Brokerage connection | | ✅ | | Privacy concerns, complexity |
| Crypto wallet tracking | | ✅ | | Secondary market |
| Multi-portfolio | | | ✅ | Power user feature |
| **Market Data** |
| Real-time quotes (delayed) | ✅ | | | Core requirement |
| Historical prices | ✅ | | | Core requirement |
| Performance charts | ✅ | | | Core requirement |
| Allocation breakdown | ✅ | | | Core requirement |
| Risk metrics | | ✅ | | Advanced feature |
| Economic indicators | | ✅ | | Nice to have |
| Sector analysis | | | ✅ | Advanced feature |
| Correlation matrix | | | ✅ | Advanced feature |
| **AI Interface** |
| CLI query interface | ✅ | | | Core requirement |
| TUI interactive mode | ✅ | | | Default experience |
| Portfolio-aware queries | ✅ | | | Core requirement |
| Streaming responses | ✅ | | | UX requirement |
| Conversation history | ✅ | | | UX requirement |
| Verified inference badges | | ✅ | | Differentiation feature |
| Query templates | | ✅ | | User convenience |
| Qt GUI | | | ✅ | Pro version only |
| Voice input | | | ✅ | Accessibility |
| **Wallet (Cold Storage)** |
| BTC receive addresses | ✅ | | | Core crypto feature |
| Balance checking | ✅ | | | Core crypto feature |
| Air-gapped signing | | ✅ | | Advanced security |
| Multi-coin support | | | ✅ | Expansion |
| **Export & Reports** |
| CSV export | ✅ | | | Data portability |
| Conversation export | ✅ | | | Data portability |
| PDF reports | | ✅ | | Professional feature |
| Scheduled reports | | ✅ | | Automation |
| Compliance package | | | ✅ | Enterprise feature |

### 2.2 MVP Feature Details

#### 2.2.1 Setup & Configuration

**Included:**
- First-boot wizard: language, timezone, network, PIN (via TUI)
- Ethernet auto-configuration via DHCP
- WiFi network selection and password entry
- 6-digit PIN with lockout protection
- Device naming for network discovery

**Excluded:**
- Static IP configuration (workaround: use router DHCP reservation)
- Air-gap mode (workaround: disconnect ethernet)
- OTA firmware updates (workaround: USB update package)

#### 2.2.2 Portfolio Management

**Included:**
- Add/edit/delete holdings via CLI or TUI
- Fields: ticker, quantity, cost basis, acquisition date
- Ticker autocomplete and validation
- CSV import with column mapping
- Single watchlist with unlimited items
- Portfolio totals: value, gain/loss, day change

**Excluded:**
- Multiple portfolios (workaround: tag holdings with notes)
- Brokerage sync (workaround: manual entry or CSV)
- Crypto tracking (workaround: manual entry as custom ticker)
- Transaction history (workaround: track in spreadsheet)

#### 2.2.3 Market Data

**Included:**
- Stock quotes from Yahoo Finance (15-min delay)
- ETF and mutual fund support
- Historical daily prices (5 years)
- Portfolio value chart (ASCII in TUI, 1D, 1W, 1M, 3M, 1Y)
- Allocation by sector and asset class
- Simple benchmark comparison (S&P 500)

**Excluded:**
- Real-time quotes (workaround: refresh manually)
- Options data (workaround: N/A)
- Fixed income analytics (workaround: manual entry)
- Risk metrics (workaround: ask AI to calculate)
- Economic indicators (workaround: ask AI)

#### 2.2.4 AI Interface

**Included:**
- CLI mode: `gazillioner query "your question here"`
- TUI mode: Interactive chat with streaming responses
- Portfolio context injection (AI knows your holdings)
- Market data context (AI can reference current prices)
- Conversation history (last 90 days)
- Standard disclaimers for financial topics

**Excluded:**
- Verification badges (V1.1 will show)
- Query templates (workaround: copy/paste)
- Voice input (workaround: type)
- Qt graphical interface (workaround: use TUI)

#### 2.2.5 Export

**Included:**
- Export holdings to CSV
- Export conversations to JSON
- Download via USB or SCP

**Excluded:**
- PDF report generation (workaround: copy to Word)
- Scheduled exports (workaround: manual or cron script)

---

## 3. Technical MVP Specification

### 3.1 Hardware Configuration

**MVP Hardware: Jetson Orin Nano Developer Kit**

| Component | Specification | Rationale |
|-----------|---------------|-----------|
| Platform | NVIDIA Jetson Orin Nano 8GB | Best perf/price for edge AI |
| Storage | 256GB NVMe SSD | Model weights + user data |
| Memory | 8GB unified | Sufficient for 7-13B models |
| Network | Gigabit Ethernet + WiFi 6 | Connectivity options |
| Power | 15W typical, 25W max | Standard adapter |
| Enclosure | Custom aluminum case | Thermal + aesthetics |

**Target MSRP:** $2,499 (includes hardware, software, setup)

**Cost Breakdown:**
| Item | Cost | % of COGS |
|------|------|-----------|
| Jetson Orin Nano | $499 | 52% |
| SSD + accessories | $80 | 8% |
| Enclosure + thermal | $120 | 13% |
| Power supply | $30 | 3% |
| Packaging + manual | $50 | 5% |
| Assembly + QA | $100 | 10% |
| Shipping reserve | $80 | 8% |
| **Total COGS** | **$959** | **100%** |
| **Gross Margin** | **$1,540** | **62%** |

### 3.2 Software Stack (Security-First Design)

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CLI (Go + Cobra)           TUI (Go + Bubbletea)            │
│  ├── Single binary          ├── Terminal UI                 │
│  ├── Scriptable             ├── SSH accessible              │
│  ├── ~15MB size             ├── No browser needed           │
│  └── Zero dependencies      └── Charts/tables in ASCII      │
│                                                              │
│  WHY NOT REACT:                                             │
│  ├── 1,200+ npm dependencies → Attack surface               │
│  ├── Browser vulnerabilities → XSS, CSRF                    │
│  └── Source code exposed → IP theft                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ gRPC / Unix Socket
┌─────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  Go gRPC Server                                             │
│  ├── Portfolio CRUD operations                               │
│  ├── Streaming chat responses                                │
│  ├── Background workers (data refresh)                       │
│  ├── Market gateway (whitelist proxy)                        │
│                                                              │
│  WHY GO (NOT PYTHON):                                       │
│  ├── 15MB binary vs 500MB venv                              │
│  ├── 50ms startup vs 2s                                      │
│  ├── 20MB memory vs 150MB                                    │
│  └── Compiled, not interpreted                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ FFI (CGO)
┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  Rust Wrapper (libgazillioner.so)                           │
│  ├── Memory-safe interface                                   │
│  ├── CUDA initialization                                     │
│  └── Token streaming                                         │
│                                                              │
│  Fortran Core (libquant.a - Static Linked)                  │
│  ├── 3.5-bit quantization (proprietary)                     │
│  ├── Optimized attention                                     │
│  ├── Compiled binary (IP protected)                          │
│  └── Stripped symbols                                        │
│                                                              │
│  Lean 4 Verification                                        │
│  └── Proof certificates for inference                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA LAYER (Rust)                          │
├─────────────────────────────────────────────────────────────┤
│  SQLite + SQLCipher                                         │
│  ├── AES-256 encryption                                      │
│  ├── Key derived from Device Root Key + PIN                  │
│  └── Audit logging with HMAC integrity                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PLATFORM LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  Alpine Linux (Custom - NOT Ubuntu)                         │
│  ├── ~50MB total (vs 500MB Ubuntu)                          │
│  ├── musl libc (simpler, fewer CVEs)                        │
│  ├── Read-only root (dm-verity)                             │
│  ├── LUKS2 encrypted data partition                         │
│  ├── Secure Boot chain                                       │
│  └── No systemd (OpenRC)                                     │
│                                                              │
│  WHY ALPINE (NOT UBUNTU):                                   │
│  ├── 100x smaller attack surface                            │
│  ├── Faster boot (35s vs 45s)                               │
│  └── Less memory (30MB vs 200MB)                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Dependency Comparison

| Metric | Old Stack (React+Python) | New Stack (Go+Rust) |
|--------|-------------------------|---------------------|
| **Total Dependencies** | 1,200+ | ~80 |
| **UI Dependencies** | 800+ (npm) | 10 (Go) |
| **Backend Dependencies** | 200+ (pip) | 30 (Go) |
| **Inference Dependencies** | 100+ | 20 (Rust crates) |
| **OS Packages** | 500+ (Ubuntu) | 50 (Alpine) |
| **Attack Surface** | Large | Minimal |
| **Auditability** | Impossible | Feasible |

### 3.4 Model Selection

**MVP Model: Llama 3 13B with Custom 3.5-bit Quantization**

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Parameters | 13B | — |
| Quantized Size | 5.7GB | Original: 26GB |
| Context Window | 8,192 tokens | Sufficient for portfolio + query |
| Inference Speed | ~30 tok/s | Jetson Orin Nano |
| Financial QA Accuracy | 91.2% | Internal benchmark |
| Memory Usage | 5.5GB | Fits in 8GB unified |

**Why Llama 3 13B:**
- Best quality/size ratio for financial reasoning
- Open weights allow local deployment
- Active community and fine-tuning ecosystem
- Runs comfortably on target hardware

### 3.5 CLI Commands (MVP)

```bash
# AI Queries
gazillioner query "Analyze my portfolio risk"
gazillioner query "Compare AAPL vs MSFT"
gazillioner query "What's my tech sector exposure?"

# Portfolio Management
gazillioner portfolio list
gazillioner portfolio add AAPL 100 150.00
gazillioner portfolio add BTC 0.5 45000.00
gazillioner portfolio import portfolio.csv
gazillioner portfolio export --format csv

# Watchlist
gazillioner watchlist add NVDA
gazillioner watchlist list
gazillioner watchlist remove NVDA

# Market Data
gazillioner market quote AAPL
gazillioner market quote BTC
gazillioner market history AAPL --days 30

# Wallet
gazillioner wallet balance
gazillioner wallet receive --coin btc

# System
gazillioner status
gazillioner config show
gazillioner config set timezone America/New_York

# Interactive Mode (TUI)
gazillioner          # Launches TUI
gazillioner --tui    # Explicit TUI launch
```

---

## 4. MVP Development Timeline

### 4.1 Sprint Plan (2-Week Sprints)

```
┌──────────────────────────────────────────────────────────────────┐
│                    MVP DEVELOPMENT TIMELINE                       │
│                    Total: 20 weeks (5 months)                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PHASE 1: FOUNDATION (Sprints 1-3, 6 weeks)                      │
│  ─────────────────────────────────────────────                   │
│  Sprint 1: Infrastructure                                         │
│    • Alpine base OS image                                         │
│    • Go project structure                                         │
│    • Rust/Fortran build system                                    │
│    • CI/CD pipeline                                               │
│                                                                   │
│  Sprint 2: Device Setup                                          │
│    • TUI setup wizard                                             │
│    • Network configuration                                        │
│    • PIN authentication                                           │
│    • Device discovery (mDNS)                                      │
│                                                                   │
│  Sprint 3: Data Layer                                            │
│    • SQLCipher database                                           │
│    • Portfolio CRUD gRPC                                          │
│    • Watchlist gRPC                                               │
│    • Encryption key derivation                                    │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PHASE 2: PORTFOLIO (Sprints 4-5, 4 weeks)                       │
│  ─────────────────────────────────────────────                   │
│  Sprint 4: Portfolio Management                                   │
│    • CLI portfolio commands                                       │
│    • TUI holdings view                                            │
│    • CSV import with mapping                                      │
│    • Ticker validation                                            │
│                                                                   │
│  Sprint 5: Market Data                                           │
│    • Market gateway (whitelist proxy)                             │
│    • Yahoo Finance integration                                    │
│    • Quote caching layer                                          │
│    • ASCII performance charts (TUI)                               │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PHASE 3: AI CORE (Sprints 6-8, 6 weeks)                         │
│  ─────────────────────────────────────────────                   │
│  Sprint 6: Inference Engine                                       │
│    • Rust wrapper for Fortran core                                │
│    • CGO bindings to Go                                           │
│    • Model loading                                                │
│    • Basic inference API                                          │
│                                                                   │
│  Sprint 7: Chat Interface                                        │
│    • CLI query command                                            │
│    • TUI chat view                                                │
│    • Streaming token display                                      │
│    • Conversation persistence                                     │
│                                                                   │
│  Sprint 8: Context Integration                                   │
│    • Portfolio context injection                                  │
│    • Market data context                                          │
│    • Financial prompt engineering                                 │
│    • Disclaimer generation                                        │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PHASE 4: POLISH (Sprints 9-10, 4 weeks)                         │
│  ─────────────────────────────────────────────                   │
│  Sprint 9: Export & Wallet                                       │
│    • CSV/JSON export                                              │
│    • Basic BTC wallet                                             │
│    • TUI polish and responsiveness                                │
│    • Error handling improvements                                  │
│                                                                   │
│  Sprint 10: Testing & Documentation                              │
│    • End-to-end testing                                           │
│    • Security audit                                               │
│    • User documentation                                           │
│    • Production image creation                                    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Milestone Checkpoints

| Milestone | Sprint | Week | Deliverable | Go/No-Go Criteria |
|-----------|--------|------|-------------|-------------------|
| M1: Bootable | 2 | 4 | Device boots, shows TUI setup | System stable 24hrs |
| M2: Portfolio | 5 | 10 | Can manage portfolio, see charts | All CRUD works |
| M3: AI Chat | 8 | 16 | Can chat with AI about portfolio | <500ms response, accurate |
| M4: Beta | 10 | 20 | Feature-complete MVP | 0 P0 bugs, docs complete |
| M5: Launch | 12 | 24 | Production units shipping | 25 beta units validated |

---

## 5. MVP Risk Assessment

### 5.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model doesn't fit 8GB | Low | High | Already validated; fallback to 7B |
| Inference too slow | Medium | High | Optimize Fortran; reduce context |
| WiFi driver issues | Medium | Medium | Ethernet fallback; known-good chip |
| Power/thermal issues | Low | High | Conservative TDP; quality enclosure |
| CGO complexity | Medium | Medium | Extensive testing; fallback to socket |

### 5.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Inference engine delays | Medium | High | Start early; parallel workflows |
| TUI polish takes longer | High | Medium | Time-box; defer to V1.1 |
| Security audit findings | Medium | Medium | Build secure from day 1 |
| Supply chain delays | Medium | High | Order components early; buffer stock |

### 5.3 Market Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Price too high | Medium | High | Validate with beta users; cost engineering |
| Feature set insufficient | Low | High | User research; flexible architecture |
| Competitor launches first | Low | Medium | Speed to market; differentiation |

---

## 6. MVP Success Metrics

### 6.1 Launch Metrics (First 90 Days)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Units Sold | 100 | Sales records |
| Setup Completion Rate | >90% | Analytics |
| Daily Active Users | >60% | Analytics |
| NPS Score | >40 | Survey |
| Support Tickets/User | <2 | Help desk |
| Return Rate | <5% | Returns records |

### 6.2 Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Inference Latency (p50) | <200ms | Logs |
| Inference Latency (p99) | <500ms | Logs |
| Uptime | >99.5% | Monitoring |
| Error Rate | <1% | Logs |
| Memory Usage | <7GB | Monitoring |
| Boot Time | <40s | Measurement |
| Binary Size | <50MB | Build |

---

## 7. Post-MVP Roadmap Preview

### V1.1 (Q3 2026) — "Pro"
- Air-gap mode with USB data import
- OTA firmware updates
- Risk metrics dashboard
- PDF report generation
- Verified inference badges
- Query templates
- Full BTC/ETH wallet with signing

### V2.0 (Q4 2026) — "Enterprise"
- Multi-portfolio support
- Brokerage connection
- Crypto wallet tracking (multi-chain)
- Qt GUI option
- Voice input
- Compliance export package
- Multi-device sync

### V3.0 (2027) — "Platform"
- Pro hardware tier (RTX 4060, 70B models)
- Custom model fine-tuning
- Developer API access
- White-label licensing
- International markets

---

*This MVP definition is approved for development. Changes require Product Owner sign-off.*
