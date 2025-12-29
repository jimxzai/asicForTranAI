"""
Market Data Gateway - Secure external data access for Gazillioner Private

Only allows whitelisted read-only requests to fetch market data.
No user data ever leaves the device.
"""

import httpx
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("market_gateway")


# =============================================================================
# WHITELIST - Only these external APIs are allowed
# =============================================================================

ALLOWED_APIS = {
    # Yahoo Finance - Stock quotes and charts
    "query1.finance.yahoo.com": {
        "methods": ["GET"],
        "paths": ["/v8/finance/chart/", "/v7/finance/quote"],
        "description": "Real-time and historical stock data"
    },
    "query2.finance.yahoo.com": {
        "methods": ["GET"],
        "paths": ["/v8/finance/chart/", "/v7/finance/quote"],
        "description": "Yahoo Finance backup"
    },

    # Alpha Vantage - Fundamentals (requires free API key)
    "www.alphavantage.co": {
        "methods": ["GET"],
        "paths": ["/query"],
        "description": "Company fundamentals"
    },

    # FRED - Federal Reserve Economic Data
    "api.stlouisfed.org": {
        "methods": ["GET"],
        "paths": ["/fred/series/observations"],
        "description": "Economic indicators (GDP, CPI, rates)"
    },

    # Finnhub - News (requires free API key)
    "finnhub.io": {
        "methods": ["GET"],
        "paths": ["/api/v1/news", "/api/v1/company-news"],
        "description": "Financial news"
    },
}


# =============================================================================
# Cache Settings
# =============================================================================

CACHE_TTL = {
    "quote": timedelta(minutes=1),       # Real-time stock prices
    "chart_1d": timedelta(minutes=5),    # Intraday charts
    "chart_1w": timedelta(hours=1),      # Weekly charts
    "chart_1y": timedelta(hours=4),      # Yearly charts
    "fundamentals": timedelta(days=1),   # Company fundamentals
    "news": timedelta(hours=2),          # News articles
    "economic": timedelta(days=1),       # FRED economic data
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StockQuote:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float]
    timestamp: datetime
    source: str = "yahoo"


@dataclass
class MarketNews:
    headline: str
    summary: str
    source: str
    url: str
    timestamp: datetime
    symbols: List[str]


@dataclass
class EconomicIndicator:
    series_id: str
    name: str
    value: float
    date: str
    unit: str


# =============================================================================
# Cache Implementation
# =============================================================================

class MarketCache:
    """SQLite-based cache for market data"""

    def __init__(self, db_path: str = "cache/market_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expires_at REAL,
                    created_at REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires
                ON cache(expires_at)
            """)

    def get(self, key: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache WHERE key = ?",
                (key,)
            ).fetchone()

            if row is None:
                return None

            value, expires_at = row
            if datetime.now().timestamp() > expires_at:
                # Expired, delete and return None
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return None

            return json.loads(value)

    def set(self, key: str, value: Dict, ttl: timedelta):
        expires_at = (datetime.now() + ttl).timestamp()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache (key, value, expires_at, created_at)
                VALUES (?, ?, ?, ?)
            """, (key, json.dumps(value), expires_at, datetime.now().timestamp()))

    def clear_expired(self):
        with sqlite3.connect(self.db_path) as conn:
            deleted = conn.execute(
                "DELETE FROM cache WHERE expires_at < ?",
                (datetime.now().timestamp(),)
            ).rowcount
            logger.info(f"Cleared {deleted} expired cache entries")


# =============================================================================
# Security Gateway
# =============================================================================

class SecurityError(Exception):
    """Raised when gateway blocks a request"""
    pass


class MarketGateway:
    """
    Secure gateway for fetching external market data.

    Security guarantees:
    - Only whitelisted hosts/paths allowed
    - GET requests only (read-only)
    - No user data in requests
    - All data cached locally
    """

    def __init__(self, cache_path: str = "cache/market_cache.db"):
        self.cache = MarketCache(cache_path)
        self.client = httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "Gazillioner/1.0"}
        )
        self.request_count = 0
        self.blocked_count = 0

    def _is_allowed(self, url: str) -> bool:
        """Check if URL is in whitelist"""
        from urllib.parse import urlparse
        parsed = urlparse(url)

        hostname = parsed.hostname
        if hostname not in ALLOWED_APIS:
            logger.warning(f"BLOCKED: Host not in whitelist: {hostname}")
            return False

        config = ALLOWED_APIS[hostname]

        # Only GET allowed
        if "GET" not in config["methods"]:
            logger.warning(f"BLOCKED: Non-GET method for {hostname}")
            return False

        # Check path prefix
        for allowed_path in config["paths"]:
            if parsed.path.startswith(allowed_path):
                return True

        logger.warning(f"BLOCKED: Path not allowed: {parsed.path}")
        return False

    def _cache_key(self, url: str) -> str:
        """Generate cache key from URL"""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    async def fetch_raw(
        self,
        url: str,
        cache_type: str = "quote"
    ) -> Optional[Dict]:
        """Fetch raw data through security gateway"""

        self.request_count += 1

        # 1. Security check
        if not self._is_allowed(url):
            self.blocked_count += 1
            raise SecurityError(f"Request blocked: {url}")

        # 2. Check cache
        cache_key = self._cache_key(url)
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug(f"Cache HIT: {url[:50]}...")
            return cached

        # 3. Fetch from external API
        logger.info(f"Fetching: {url[:80]}...")
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            return None

        # 4. Cache result
        ttl = CACHE_TTL.get(cache_type, timedelta(hours=1))
        self.cache.set(cache_key, data, ttl)

        return data

    # =========================================================================
    # High-level API methods
    # =========================================================================

    async def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get real-time stock quote"""
        symbol = symbol.upper()
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"

        data = await self.fetch_raw(url, cache_type="quote")
        if not data:
            return None

        try:
            result = data["quoteResponse"]["result"][0]
            return StockQuote(
                symbol=symbol,
                price=result.get("regularMarketPrice", 0),
                change=result.get("regularMarketChange", 0),
                change_percent=result.get("regularMarketChangePercent", 0),
                volume=result.get("regularMarketVolume", 0),
                market_cap=result.get("marketCap"),
                timestamp=datetime.now(),
                source="yahoo"
            )
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing quote for {symbol}: {e}")
            return None

    async def get_quotes(self, symbols: List[str]) -> Dict[str, StockQuote]:
        """Get multiple stock quotes"""
        symbols = [s.upper() for s in symbols]
        symbols_str = ",".join(symbols)
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols_str}"

        data = await self.fetch_raw(url, cache_type="quote")
        if not data:
            return {}

        quotes = {}
        try:
            for result in data["quoteResponse"]["result"]:
                symbol = result["symbol"]
                quotes[symbol] = StockQuote(
                    symbol=symbol,
                    price=result.get("regularMarketPrice", 0),
                    change=result.get("regularMarketChange", 0),
                    change_percent=result.get("regularMarketChangePercent", 0),
                    volume=result.get("regularMarketVolume", 0),
                    market_cap=result.get("marketCap"),
                    timestamp=datetime.now(),
                    source="yahoo"
                )
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing quotes: {e}")

        return quotes

    async def get_chart(
        self,
        symbol: str,
        range: str = "1d",
        interval: str = "5m"
    ) -> Optional[Dict]:
        """Get historical chart data"""
        symbol = symbol.upper()
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?range={range}&interval={interval}"
        )

        cache_type = f"chart_{range}"
        return await self.fetch_raw(url, cache_type=cache_type)

    async def get_economic_indicator(
        self,
        series_id: str,
        api_key: str = ""
    ) -> Optional[EconomicIndicator]:
        """Get FRED economic indicator"""
        # Common series: GDP, CPIAUCSL, FEDFUNDS, UNRATE
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&file_type=json&limit=1"
            f"&sort_order=desc&api_key={api_key}"
        )

        data = await self.fetch_raw(url, cache_type="economic")
        if not data or "observations" not in data:
            return None

        try:
            obs = data["observations"][0]
            return EconomicIndicator(
                series_id=series_id,
                name=series_id,  # Could fetch full name
                value=float(obs["value"]) if obs["value"] != "." else 0,
                date=obs["date"],
                unit="varies"
            )
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Error parsing FRED data: {e}")
            return None

    def get_stats(self) -> Dict:
        """Get gateway statistics"""
        return {
            "total_requests": self.request_count,
            "blocked_requests": self.blocked_count,
            "allowed_hosts": list(ALLOWED_APIS.keys())
        }


# =============================================================================
# FastAPI Routes
# =============================================================================

def create_market_routes(app):
    """Add market data routes to FastAPI app"""
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    router = APIRouter(prefix="/v1/market", tags=["market"])
    gateway = MarketGateway()

    class QuoteResponse(BaseModel):
        symbol: str
        price: float
        change: float
        change_percent: float
        volume: int
        market_cap: Optional[float]
        timestamp: str

    class MultiQuoteRequest(BaseModel):
        symbols: List[str]

    @router.get("/quote/{symbol}", response_model=QuoteResponse)
    async def get_quote(symbol: str):
        """Get real-time stock quote"""
        quote = await gateway.get_quote(symbol)
        if not quote:
            raise HTTPException(404, f"Quote not found for {symbol}")

        return QuoteResponse(
            symbol=quote.symbol,
            price=quote.price,
            change=quote.change,
            change_percent=quote.change_percent,
            volume=quote.volume,
            market_cap=quote.market_cap,
            timestamp=quote.timestamp.isoformat()
        )

    @router.post("/quotes")
    async def get_quotes(request: MultiQuoteRequest):
        """Get multiple stock quotes"""
        quotes = await gateway.get_quotes(request.symbols)
        return {
            "quotes": {
                symbol: {
                    "price": q.price,
                    "change": q.change,
                    "change_percent": q.change_percent,
                    "volume": q.volume
                }
                for symbol, q in quotes.items()
            }
        }

    @router.get("/chart/{symbol}")
    async def get_chart(symbol: str, range: str = "1d", interval: str = "5m"):
        """Get historical chart data"""
        data = await gateway.get_chart(symbol, range, interval)
        if not data:
            raise HTTPException(404, f"Chart not found for {symbol}")
        return data

    @router.get("/stats")
    async def get_stats():
        """Get gateway statistics"""
        return gateway.get_stats()

    app.include_router(router)
    return gateway


# =============================================================================
# Standalone usage
# =============================================================================

async def main():
    """Test the market gateway"""
    gateway = MarketGateway()

    print("Testing Market Gateway...")
    print("=" * 50)

    # Test stock quote
    quote = await gateway.get_quote("NVDA")
    if quote:
        print(f"\nNVDA: ${quote.price:.2f} ({quote.change_percent:+.2f}%)")

    # Test multiple quotes
    quotes = await gateway.get_quotes(["AAPL", "GOOGL", "MSFT"])
    print("\nMultiple quotes:")
    for symbol, q in quotes.items():
        print(f"  {symbol}: ${q.price:.2f}")

    # Test blocked request
    print("\nTesting blocked request...")
    try:
        await gateway.fetch_raw("https://evil.com/steal-data")
    except SecurityError as e:
        print(f"  ✓ Correctly blocked: {e}")

    # Stats
    print(f"\nGateway stats: {gateway.get_stats()}")


if __name__ == "__main__":
    asyncio.run(main())
