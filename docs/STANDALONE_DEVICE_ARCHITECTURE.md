# Gazillioner Private - Standalone Device Architecture

**The "Cold Wallet for Financial AI"**

A self-contained hardware device that runs the full Gazillioner platform locally while selectively accessing external market data.

---

## Concept

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     GAZILLIONER PRIVATE DEVICE                          │
│                     ════════════════════════                            │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LOCAL PROCESSING                              │   │
│  │    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │   │
│  │    │   Next.js    │    │   FastAPI    │    │   Fortran    │    │   │
│  │    │   Frontend   │───▶│   Backend    │───▶│   Inference  │    │   │
│  │    │   (Static)   │    │   (Python)   │    │   (3.5-bit)  │    │   │
│  │    └──────────────┘    └──────────────┘    └──────────────┘    │   │
│  │           │                   │                   │             │   │
│  │           └───────────────────┼───────────────────┘             │   │
│  │                               │                                  │   │
│  │                    ┌──────────▼──────────┐                      │   │
│  │                    │   SQLite / JSON     │                      │   │
│  │                    │   Local Database    │                      │   │
│  │                    │   (FQ, Portfolio)   │                      │   │
│  │                    └─────────────────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                     │
│                                   │ SELECTIVE DATA FETCH                │
│                                   │ (Read-only, no user data sent)      │
│                                   ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    DATA GATEWAY (Firewall)                       │   │
│  │                                                                   │   │
│  │   ALLOWED:                    │   BLOCKED:                       │   │
│  │   ✓ Stock prices (Yahoo)      │   ✗ User portfolio data          │   │
│  │   ✓ Market indices            │   ✗ Personal information          │   │
│  │   ✓ Company fundamentals      │   ✗ Transaction history          │   │
│  │   ✓ Economic indicators       │   ✗ AI conversation logs         │   │
│  │   ✓ News headlines            │   ✗ FQ assessment results        │   │
│  │                               │   ✗ ANY uploads                   │   │
│  └───────────────────────────────┴───────────────────────────────────┘   │
│                                   │                                     │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │   INTERNET      │
                          │   (Read-only)   │
                          │                 │
                          │ • Yahoo Finance │
                          │ • Alpha Vantage │
                          │ • FRED (Fed)    │
                          │ • News APIs     │
                          └─────────────────┘
```

---

## Hardware Options

### Option 1: Raspberry Pi 5 + Coral TPU (Budget: ~$200)

```
┌────────────────────────────────────────┐
│         Raspberry Pi 5 8GB             │
│         + Coral USB TPU                │
├────────────────────────────────────────┤
│ CPU: ARM Cortex-A76 (4 cores)          │
│ RAM: 8GB LPDDR4X                       │
│ AI:  Coral Edge TPU (4 TOPS)           │
│ Storage: 256GB NVMe SSD                │
│ Network: Ethernet + WiFi               │
│ Power: 27W USB-C                       │
├────────────────────────────────────────┤
│ Performance:                           │
│ • Inference: ~500ms per query          │
│ • Model: 7B quantized (3.5-bit)        │
│ • Concurrent users: 1                  │
├────────────────────────────────────────┤
│ Cost: ~$200                            │
│ Power: 15W idle, 27W peak              │
└────────────────────────────────────────┘
```

### Option 2: Nvidia Jetson Orin Nano (Budget: ~$500)

```
┌────────────────────────────────────────┐
│       Nvidia Jetson Orin Nano          │
│           (Recommended)                │
├────────────────────────────────────────┤
│ CPU: 6-core ARM Cortex-A78AE           │
│ GPU: 1024 CUDA cores                   │
│ RAM: 8GB unified LPDDR5                │
│ AI:  40 TOPS INT8                      │
│ Storage: 256GB NVMe SSD                │
│ Network: GbE + WiFi 6                  │
│ Power: 15W                             │
├────────────────────────────────────────┤
│ Performance:                           │
│ • Inference: ~50ms per query           │
│ • Model: 13B quantized (3.5-bit)       │
│ • Concurrent users: 3-5                │
├────────────────────────────────────────┤
│ Cost: ~$500                            │
│ Power: 7W idle, 15W peak               │
└────────────────────────────────────────┘
```

### Option 3: Mini PC + RTX 4060 (Budget: ~$800)

```
┌────────────────────────────────────────┐
│     Mini PC + External RTX 4060        │
│         (Power User)                   │
├────────────────────────────────────────┤
│ CPU: Intel i5-13400 / AMD 7600         │
│ GPU: RTX 4060 8GB (external eGPU)      │
│ RAM: 32GB DDR5                         │
│ Storage: 512GB NVMe SSD                │
│ Network: 2.5GbE + WiFi 6E              │
│ Power: 150W                            │
├────────────────────────────────────────┤
│ Performance:                           │
│ • Inference: ~17ms per query           │
│ • Model: 70B quantized (3.5-bit)       │
│ • Concurrent users: 10+                │
├────────────────────────────────────────┤
│ Cost: ~$800                            │
│ Power: 50W idle, 150W peak             │
└────────────────────────────────────────┘
```

---

## Software Stack

### Complete Stack on Device

```
┌─────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 Next.js (Static Export)                   │   │
│  │                                                           │   │
│  │  • FQ Assessment                                          │   │
│  │  • AI Chat Interface                                      │   │
│  │  • Portfolio Tracker                                      │   │
│  │  • Financial Planning Tools                               │   │
│  │  • Knowledge Hub (cached offline)                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                          API LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Backend                        │   │
│  │                                                           │   │
│  │  /v1/chat          - AI chat with verification            │   │
│  │  /v1/fq/analyze    - FQ assessment analysis               │   │
│  │  /v1/portfolio     - Portfolio analysis                   │   │
│  │  /v1/market/quote  - Fetch stock prices (via gateway)     │   │
│  │  /v1/market/news   - Fetch news (via gateway)             │   │
│  │  /health           - System health                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                       INFERENCE LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Fortran 2023 Inference Engine                │   │
│  │                                                           │   │
│  │  • 3.5-bit quantized LLaMA (7B/13B/70B)                  │   │
│  │  • SIMD-optimized matrix operations                       │   │
│  │  • Lean 4 verified computations                           │   │
│  │  • SPARK/Ada safety contracts                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                        DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │    SQLite       │  │   JSON Files    │  │  Market Cache  │   │
│  │                 │  │                 │  │                │   │
│  │  • User profile │  │  • Model weights│  │  • Stock cache │   │
│  │  • FQ history   │  │  • Config       │  │  • News cache  │   │
│  │  • Portfolio    │  │  • UI settings  │  │  • Index data  │   │
│  │  • Chat logs    │  │                 │  │  (24hr TTL)    │   │
│  └─────────────────┘  └─────────────────┘  └────────────────┘   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                      SECURITY LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Data Gateway                            │   │
│  │                                                           │   │
│  │  OUTBOUND WHITELIST (read-only):                         │   │
│  │  ✓ query1.finance.yahoo.com (stock quotes)               │   │
│  │  ✓ api.polygon.io (market data)                          │   │
│  │  ✓ api.stlouisfed.org (FRED economic data)               │   │
│  │  ✓ newsapi.org (headlines only)                          │   │
│  │                                                           │   │
│  │  BLOCKED:                                                 │   │
│  │  ✗ All POST requests with user data                      │   │
│  │  ✗ All file uploads                                      │   │
│  │  ✗ Telemetry / analytics                                 │   │
│  │  ✗ Any non-whitelisted domains                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Stock Quote Example

```
User: "What's NVDA trading at?"
         │
         ▼
┌─────────────────────┐
│    Local Browser    │
│    (localhost)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   FastAPI Backend   │
│   /v1/market/quote  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│    Market Cache     │────▶│  Cache HIT?         │
│    (SQLite/Redis)   │     │  Return cached      │
└─────────────────────┘     └─────────────────────┘
          │                           │
          │ Cache MISS                │
          ▼                           │
┌─────────────────────┐               │
│   Data Gateway      │               │
│   (Outbound filter) │               │
└─────────┬───────────┘               │
          │                           │
          │ GET https://query1.       │
          │ finance.yahoo.com/v8/     │
          │ finance/chart/NVDA        │
          ▼                           │
┌─────────────────────┐               │
│   Yahoo Finance     │               │
│   (Read-only)       │               │
└─────────┬───────────┘               │
          │                           │
          │ {price: 137.42}           │
          ▼                           │
┌─────────────────────┐               │
│   Cache + Return    │◀──────────────┘
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Fortran Engine    │
│   + AI Analysis     │
│   "NVDA is at       │
│   $137.42, up 2.3%" │
└─────────┬───────────┘
          │
          ▼
      User sees
      verified response
```

---

## Directory Structure on Device

```
/opt/gazillioner/
├── frontend/                    # Next.js static export
│   ├── index.html
│   ├── _next/
│   └── public/
│
├── backend/                     # FastAPI + Fortran
│   ├── api/
│   │   ├── inference_api.py
│   │   ├── market_gateway.py   # Data gateway
│   │   └── cache_service.py
│   ├── fortran/
│   │   ├── inference_engine
│   │   └── libquant3p5.so
│   └── requirements.txt
│
├── data/                        # All user data (encrypted)
│   ├── user.db                  # SQLite database
│   ├── portfolio.json
│   ├── fq_history.json
│   └── chat_logs/
│
├── models/                      # AI model weights
│   ├── llama-7b-3p5bit.bin
│   └── embeddings.bin
│
├── cache/                       # Market data cache
│   ├── quotes.db
│   └── news.db
│
├── config/
│   ├── gateway_whitelist.yaml  # Allowed external APIs
│   ├── device.yaml             # Device settings
│   └── security.yaml           # Security policies
│
└── scripts/
    ├── start.sh                # Launch all services
    ├── update_market.sh        # Refresh market cache
    └── backup.sh               # Encrypted backup
```

---

## Implementation: Data Gateway

```python
# /opt/gazillioner/backend/api/market_gateway.py
"""
Secure Data Gateway - Only allows whitelisted read-only requests
"""

import httpx
from typing import Optional
import hashlib
from datetime import datetime, timedelta

# Whitelist of allowed external APIs
ALLOWED_HOSTS = {
    "query1.finance.yahoo.com": {
        "methods": ["GET"],
        "paths": ["/v8/finance/chart/", "/v7/finance/quote"],
        "description": "Stock quotes and charts"
    },
    "api.polygon.io": {
        "methods": ["GET"],
        "paths": ["/v2/aggs/", "/v3/reference/"],
        "description": "Market data"
    },
    "api.stlouisfed.org": {
        "methods": ["GET"],
        "paths": ["/fred/series/observations"],
        "description": "Economic indicators"
    },
    "newsapi.org": {
        "methods": ["GET"],
        "paths": ["/v2/everything", "/v2/top-headlines"],
        "description": "Financial news"
    }
}

# Cache settings
CACHE_TTL = {
    "quote": timedelta(minutes=1),      # Real-time quotes
    "chart": timedelta(hours=1),        # Historical charts
    "fundamentals": timedelta(days=1),  # Company data
    "news": timedelta(hours=4),         # News articles
    "economic": timedelta(days=1)       # FRED data
}


class DataGateway:
    """Secure gateway for external data access"""

    def __init__(self, cache_db: str = "/opt/gazillioner/cache/gateway.db"):
        self.cache_db = cache_db
        self.client = httpx.AsyncClient(timeout=10.0)

    async def fetch(
        self,
        url: str,
        cache_type: str = "quote"
    ) -> Optional[dict]:
        """Fetch data through security gateway"""

        # 1. Validate URL against whitelist
        if not self._is_allowed(url):
            raise SecurityError(f"Blocked: {url} not in whitelist")

        # 2. Check cache
        cache_key = self._cache_key(url)
        cached = await self._get_cache(cache_key)
        if cached:
            return cached

        # 3. Fetch from external API (GET only)
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()

        # 4. Cache result
        ttl = CACHE_TTL.get(cache_type, timedelta(hours=1))
        await self._set_cache(cache_key, data, ttl)

        # 5. Log access (local only, no external telemetry)
        self._log_access(url, "success")

        return data

    def _is_allowed(self, url: str) -> bool:
        """Check if URL is in whitelist"""
        from urllib.parse import urlparse
        parsed = urlparse(url)

        if parsed.hostname not in ALLOWED_HOSTS:
            return False

        config = ALLOWED_HOSTS[parsed.hostname]

        # Only GET allowed
        if "GET" not in config["methods"]:
            return False

        # Check path prefix
        for allowed_path in config["paths"]:
            if parsed.path.startswith(allowed_path):
                return True

        return False

    def _cache_key(self, url: str) -> str:
        """Generate cache key from URL"""
        return hashlib.sha256(url.encode()).hexdigest()[:16]


class SecurityError(Exception):
    """Raised when gateway blocks a request"""
    pass
```

---

## Implementation: Startup Script

```bash
#!/bin/bash
# /opt/gazillioner/scripts/start.sh
# Gazillioner Private - Startup Script

set -e

GAZILLIONER_HOME="/opt/gazillioner"
LOG_DIR="/var/log/gazillioner"

echo "╔═══════════════════════════════════════════════════════╗"
echo "║         GAZILLIONER PRIVATE - Starting...             ║"
echo "║         Your Personal Financial AI                    ║"
echo "╚═══════════════════════════════════════════════════════╝"

# 1. Check hardware
echo "[1/5] Checking hardware..."
if command -v nvidia-smi &> /dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "  ✓ GPU detected: $GPU"
else
    echo "  ℹ No GPU detected, using CPU inference"
fi

# 2. Start Fortran inference engine
echo "[2/5] Loading AI model..."
cd $GAZILLIONER_HOME/backend/fortran
./inference_engine --model ../models/llama-7b-3p5bit.bin &
INFERENCE_PID=$!
sleep 2
if kill -0 $INFERENCE_PID 2>/dev/null; then
    echo "  ✓ Inference engine ready (PID: $INFERENCE_PID)"
else
    echo "  ✗ Failed to start inference engine"
    exit 1
fi

# 3. Start FastAPI backend
echo "[3/5] Starting API backend..."
cd $GAZILLIONER_HOME/backend
source venv/bin/activate
uvicorn api.inference_api:app \
    --host 127.0.0.1 \
    --port 8000 \
    --log-level warning \
    > $LOG_DIR/backend.log 2>&1 &
BACKEND_PID=$!
sleep 2
if curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "  ✓ Backend ready (PID: $BACKEND_PID)"
else
    echo "  ✗ Backend failed to start"
    exit 1
fi

# 4. Start nginx for frontend
echo "[4/5] Starting frontend..."
nginx -c $GAZILLIONER_HOME/config/nginx.conf
echo "  ✓ Frontend ready at http://localhost:3000"

# 5. Update market data cache
echo "[5/5] Refreshing market data..."
$GAZILLIONER_HOME/scripts/update_market.sh &
echo "  ✓ Market data updating in background"

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║         GAZILLIONER PRIVATE - Ready!                  ║"
echo "║                                                       ║"
echo "║   Open in browser: http://localhost:3000              ║"
echo "║                                                       ║"
echo "║   All data stays on this device.                      ║"
echo "║   Only stock prices fetched from internet.            ║"
echo "╚═══════════════════════════════════════════════════════╝"

# Keep running
wait
```

---

## Air-Gap Mode (Maximum Privacy)

For users who want complete isolation:

```
┌─────────────────────────────────────────────────────────────────┐
│                      AIR-GAP MODE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  NETWORK: DISABLED                                               │
│                                                                  │
│  Features available:                                             │
│  ✓ AI chat (offline model)                                      │
│  ✓ FQ assessment                                                │
│  ✓ Portfolio tracking (manual entry)                            │
│  ✓ Financial planning calculators                               │
│  ✓ Knowledge hub (pre-cached)                                   │
│                                                                  │
│  Features unavailable:                                           │
│  ✗ Live stock prices                                            │
│  ✗ Real-time news                                               │
│  ✗ Market data updates                                          │
│                                                                  │
│  Data update method:                                             │
│  1. Connect USB drive to internet-connected computer            │
│  2. Download market data bundle (signed + encrypted)            │
│  3. Transfer USB to Gazillioner Private device                  │
│  4. Import data (signature verified)                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Bill of Materials (Recommended: Jetson Orin Nano)

| Component | Model | Price |
|-----------|-------|-------|
| Compute | Nvidia Jetson Orin Nano 8GB | $250 |
| Storage | Samsung 970 EVO 256GB NVMe | $50 |
| Case | Aluminum heatsink case | $40 |
| Power | Official 45W adapter | $30 |
| microSD | Samsung EVO 64GB (boot) | $15 |
| Ethernet | Cat6 cable | $10 |
| **Total** | | **~$400** |

Optional:
| Component | Model | Price |
|-----------|-------|-------|
| WiFi antenna | Dual-band external | $15 |
| Display | 7" touchscreen | $60 |
| UPS | Mini UPS 30W | $40 |

---

## Pricing Model

| Edition | Hardware | Features | Price |
|---------|----------|----------|-------|
| **Gazillioner Free** | Cloud (web) | Basic FQ, limited chat | $0 |
| **Gazillioner Plus** | Cloud (web) | Full features, verified AI | $9.99/mo |
| **Gazillioner Private** | Device (owned) | Everything local, lifetime | $599 one-time |
| **Gazillioner Pro** | Device + support | Priority support, updates | $899 + $99/yr |

---

## Security Guarantees

| Guarantee | How |
|-----------|-----|
| **No user data leaves device** | Gateway whitelist, outbound firewall |
| **No telemetry** | No analytics, no tracking code |
| **Encrypted storage** | LUKS full-disk encryption |
| **Verified AI** | Lean 4 proofs, SPARK contracts |
| **Open source** | All code auditable |
| **Physical security** | TPM, secure boot, tamper detection |

---

## Next Steps

1. [ ] Prototype on Jetson Orin Nano
2. [ ] Compile Fortran engine for ARM64
3. [ ] Build static Next.js export
4. [ ] Implement data gateway
5. [ ] Test offline mode
6. [ ] Design enclosure
7. [ ] Create installation image
8. [ ] Security audit

---

*Document Version: 1.0*
*Last Updated: 2025-12-28*
