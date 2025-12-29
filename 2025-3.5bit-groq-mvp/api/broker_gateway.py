"""
Broker Gateway - Secure API access for brokerage integrations

Extends the market_gateway pattern with:
- OAuth token management
- Broker-specific API whitelisting
- Rate limiting per broker
- Position/order normalization
"""

import asyncio
import hashlib
import httpx
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("broker_gateway")

# =============================================================================
# Broker API Whitelists - SECURITY CRITICAL
# Only these domains are allowed for broker API calls
# =============================================================================

BROKER_CONFIGS = {
    "alpaca": {
        "paper": {
            "base_url": "https://paper-api.alpaca.markets",
            "data_url": "https://data.alpaca.markets",
        },
        "live": {
            "base_url": "https://api.alpaca.markets",
            "data_url": "https://data.alpaca.markets",
        },
        "oauth_url": "https://app.alpaca.markets/oauth/authorize",
        "token_url": "https://api.alpaca.markets/oauth/token",
        "allowed_domains": [
            "paper-api.alpaca.markets",
            "api.alpaca.markets",
            "data.alpaca.markets",
            "app.alpaca.markets",
        ],
        "rate_limit": 200,  # requests per minute
    },
    "schwab": {
        "base_url": "https://api.schwabapi.com/trader/v1",
        "oauth_url": "https://api.schwabapi.com/v1/oauth/authorize",
        "token_url": "https://api.schwabapi.com/v1/oauth/token",
        "allowed_domains": [
            "api.schwabapi.com",
        ],
        "rate_limit": 120,  # requests per minute
    },
    "coinbase": {
        "base_url": "https://api.coinbase.com/v2",
        "advanced_base_url": "https://api.coinbase.com/api/v3",
        "oauth_url": "https://www.coinbase.com/oauth/authorize",
        "token_url": "https://api.coinbase.com/oauth/token",
        "allowed_domains": [
            "api.coinbase.com",
            "www.coinbase.com",
        ],
        "rate_limit": 10000,  # requests per hour
        "rate_window": 3600,  # 1 hour in seconds
    },
    "ibkr": {
        # IBKR Client Portal runs locally
        "client_portal": "https://localhost:5000/v1/api",
        "allowed_domains": [
            "localhost",
            "127.0.0.1",
        ],
        "rate_limit": 50,  # requests per minute
    },
}


class BrokerType(str, Enum):
    ALPACA = "alpaca"
    SCHWAB = "schwab"
    COINBASE = "coinbase"
    IBKR = "ibkr"


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, requests_per_window: int, window_seconds: int = 60):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.requests: List[datetime] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request slot is available"""
        async with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window_seconds)

            # Remove old requests
            self.requests = [r for r in self.requests if r > cutoff]

            if len(self.requests) >= self.requests_per_window:
                # Calculate wait time
                oldest = self.requests[0]
                wait_time = (oldest + timedelta(seconds=self.window_seconds) - now).total_seconds()
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    # Clean up again after waiting
                    now = datetime.now()
                    cutoff = now - timedelta(seconds=self.window_seconds)
                    self.requests = [r for r in self.requests if r > cutoff]

            self.requests.append(now)


# =============================================================================
# Normalized Data Models
# =============================================================================


@dataclass
class NormalizedPosition:
    """Normalized position across all brokers"""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float
    side: str  # "long" or "short"
    asset_class: str  # "stock", "crypto", "etf", etc.
    broker: str
    account_id: str


@dataclass
class NormalizedOrder:
    """Normalized order across all brokers"""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit", etc.
    quantity: float
    filled_qty: float
    limit_price: Optional[float]
    status: str
    created_at: datetime
    broker: str


@dataclass
class NormalizedAccount:
    """Normalized account info across all brokers"""
    account_id: str
    broker: str
    account_type: str
    currency: str
    cash: float
    buying_power: float
    portfolio_value: float
    day_trade_count: int = 0


# =============================================================================
# Base Broker Adapter
# =============================================================================


class BaseBrokerAdapter:
    """Base class for broker-specific adapters"""

    broker_type: BrokerType
    config: Dict[str, Any]

    def __init__(self, paper: bool = True):
        self.paper = paper
        self.config = BROKER_CONFIGS.get(self.broker_type.value, {})
        self.rate_limiter = RateLimiter(
            self.config.get("rate_limit", 100),
            self.config.get("rate_window", 60),
        )
        self.client = httpx.AsyncClient(timeout=30.0)
        self._credentials: Dict[str, str] = {}

    def set_credentials(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> None:
        """Set API credentials"""
        if api_key:
            self._credentials["api_key"] = api_key
        if api_secret:
            self._credentials["api_secret"] = api_secret
        if access_token:
            self._credentials["access_token"] = access_token

    def _validate_url(self, url: str) -> bool:
        """Validate URL is in the allowed domain whitelist"""
        parsed = urlparse(url)
        allowed = self.config.get("allowed_domains", [])
        return parsed.netloc in allowed

    async def _request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make an authenticated, rate-limited request"""
        # Validate URL against whitelist
        if not self._validate_url(url):
            raise SecurityError(f"URL not in allowed domains: {url}")

        # Apply rate limiting
        await self.rate_limiter.acquire()

        # Make request
        response = await self.client.request(method, url, headers=headers, **kwargs)

        # Handle errors
        if response.status_code == 429:
            raise RateLimitError("Rate limited by broker API")
        if response.status_code == 401:
            raise AuthenticationError("Invalid or expired credentials")
        if response.status_code == 403:
            raise AuthorizationError("Access denied")

        response.raise_for_status()
        return response.json()

    async def get_account(self) -> NormalizedAccount:
        """Get account info - to be implemented by subclasses"""
        raise NotImplementedError

    async def get_positions(self) -> List[NormalizedPosition]:
        """Get positions - to be implemented by subclasses"""
        raise NotImplementedError

    async def get_orders(self, status: str = "open") -> List[NormalizedOrder]:
        """Get orders - to be implemented by subclasses"""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the HTTP client"""
        await self.client.aclose()


# =============================================================================
# Alpaca Adapter
# =============================================================================


class AlpacaAdapter(BaseBrokerAdapter):
    """Alpaca Markets API adapter"""

    broker_type = BrokerType.ALPACA

    def __init__(self, paper: bool = True):
        super().__init__(paper)
        env = "paper" if paper else "live"
        self.base_url = self.config[env]["base_url"]
        self.data_url = self.config[env]["data_url"]

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if "access_token" in self._credentials:
            return {"Authorization": f"Bearer {self._credentials['access_token']}"}
        return {
            "APCA-API-KEY-ID": self._credentials.get("api_key", ""),
            "APCA-API-SECRET-KEY": self._credentials.get("api_secret", ""),
        }

    async def get_account(self) -> NormalizedAccount:
        """Get Alpaca account info"""
        data = await self._request(
            "GET",
            f"{self.base_url}/v2/account",
            headers=self._get_headers(),
        )
        return NormalizedAccount(
            account_id=data["account_number"],
            broker="alpaca",
            account_type="margin" if data.get("multiplier", 1) > 1 else "cash",
            currency=data.get("currency", "USD"),
            cash=float(data.get("cash", 0)),
            buying_power=float(data.get("buying_power", 0)),
            portfolio_value=float(data.get("portfolio_value", 0)),
            day_trade_count=int(data.get("daytrade_count", 0)),
        )

    async def get_positions(self) -> List[NormalizedPosition]:
        """Get Alpaca positions"""
        data = await self._request(
            "GET",
            f"{self.base_url}/v2/positions",
            headers=self._get_headers(),
        )
        return [
            NormalizedPosition(
                symbol=p["symbol"],
                quantity=float(p["qty"]),
                avg_cost=float(p["avg_entry_price"]),
                current_price=float(p["current_price"]),
                market_value=float(p["market_value"]),
                unrealized_pl=float(p["unrealized_pl"]),
                unrealized_pl_pct=float(p["unrealized_plpc"]) * 100,
                side=p["side"],
                asset_class=p.get("asset_class", "us_equity"),
                broker="alpaca",
                account_id=p.get("account_id", ""),
            )
            for p in data
        ]

    async def get_orders(self, status: str = "open") -> List[NormalizedOrder]:
        """Get Alpaca orders"""
        params = {"status": status}
        data = await self._request(
            "GET",
            f"{self.base_url}/v2/orders",
            headers=self._get_headers(),
            params=params,
        )
        return [
            NormalizedOrder(
                order_id=o["id"],
                symbol=o["symbol"],
                side=o["side"],
                order_type=o["type"],
                quantity=float(o["qty"]),
                filled_qty=float(o.get("filled_qty", 0)),
                limit_price=float(o["limit_price"]) if o.get("limit_price") else None,
                status=o["status"],
                created_at=datetime.fromisoformat(o["created_at"].replace("Z", "+00:00")),
                broker="alpaca",
            )
            for o in data
        ]

    @staticmethod
    def get_oauth_url(client_id: str, redirect_uri: str, state: str) -> str:
        """Generate OAuth authorization URL"""
        return (
            f"{BROKER_CONFIGS['alpaca']['oauth_url']}"
            f"?response_type=code"
            f"&client_id={client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&state={state}"
            f"&scope=account:write%20trading"
        )

    async def exchange_code(
        self, code: str, client_id: str, client_secret: str, redirect_uri: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        response = await self.client.post(
            BROKER_CONFIGS["alpaca"]["token_url"],
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
            },
        )
        response.raise_for_status()
        return response.json()


# =============================================================================
# Coinbase Adapter
# =============================================================================


class CoinbaseAdapter(BaseBrokerAdapter):
    """Coinbase API adapter"""

    broker_type = BrokerType.COINBASE

    def __init__(self):
        super().__init__(paper=False)  # Coinbase doesn't have paper trading
        self.base_url = self.config["base_url"]

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        return {"Authorization": f"Bearer {self._credentials.get('access_token', '')}"}

    async def get_account(self) -> NormalizedAccount:
        """Get Coinbase account info"""
        data = await self._request(
            "GET",
            f"{self.base_url}/accounts",
            headers=self._get_headers(),
        )
        # Aggregate all account balances
        total_value = 0.0
        for account in data.get("data", []):
            balance = float(account.get("balance", {}).get("amount", 0))
            # Convert to USD if possible
            native = account.get("native_balance", {})
            if native:
                total_value += float(native.get("amount", 0))

        return NormalizedAccount(
            account_id=data.get("data", [{}])[0].get("id", ""),
            broker="coinbase",
            account_type="crypto",
            currency="USD",
            cash=total_value,
            buying_power=total_value,
            portfolio_value=total_value,
        )

    async def get_positions(self) -> List[NormalizedPosition]:
        """Get Coinbase positions (crypto balances)"""
        data = await self._request(
            "GET",
            f"{self.base_url}/accounts",
            headers=self._get_headers(),
        )
        positions = []
        for account in data.get("data", []):
            balance = float(account.get("balance", {}).get("amount", 0))
            if balance > 0:
                native = account.get("native_balance", {})
                market_value = float(native.get("amount", 0)) if native else 0
                positions.append(
                    NormalizedPosition(
                        symbol=account.get("currency", {}).get("code", ""),
                        quantity=balance,
                        avg_cost=0,  # Coinbase doesn't provide cost basis
                        current_price=market_value / balance if balance > 0 else 0,
                        market_value=market_value,
                        unrealized_pl=0,
                        unrealized_pl_pct=0,
                        side="long",
                        asset_class="crypto",
                        broker="coinbase",
                        account_id=account.get("id", ""),
                    )
                )
        return positions

    @staticmethod
    def get_oauth_url(client_id: str, redirect_uri: str, state: str) -> str:
        """Generate OAuth authorization URL"""
        return (
            f"{BROKER_CONFIGS['coinbase']['oauth_url']}"
            f"?response_type=code"
            f"&client_id={client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&state={state}"
            f"&scope=wallet:accounts:read,wallet:transactions:read"
        )


# =============================================================================
# Schwab Adapter (Stub)
# =============================================================================


class SchwabAdapter(BaseBrokerAdapter):
    """Charles Schwab API adapter"""

    broker_type = BrokerType.SCHWAB

    def __init__(self):
        super().__init__(paper=False)
        self.base_url = self.config["base_url"]

    # TODO: Implement Schwab-specific methods


# =============================================================================
# IBKR Adapter (Stub)
# =============================================================================


class IBKRAdapter(BaseBrokerAdapter):
    """Interactive Brokers Client Portal adapter"""

    broker_type = BrokerType.IBKR

    def __init__(self):
        super().__init__(paper=False)
        self.base_url = self.config["client_portal"]

    # TODO: Implement IBKR-specific methods


# =============================================================================
# Exceptions
# =============================================================================


class BrokerError(Exception):
    """Base broker error"""
    pass


class SecurityError(BrokerError):
    """Security violation (e.g., URL not in whitelist)"""
    pass


class RateLimitError(BrokerError):
    """Rate limit exceeded"""
    pass


class AuthenticationError(BrokerError):
    """Authentication failed"""
    pass


class AuthorizationError(BrokerError):
    """Authorization denied"""
    pass


# =============================================================================
# Factory
# =============================================================================


def get_adapter(broker_type: BrokerType, paper: bool = True) -> BaseBrokerAdapter:
    """Get the appropriate adapter for a broker type"""
    adapters = {
        BrokerType.ALPACA: lambda: AlpacaAdapter(paper=paper),
        BrokerType.COINBASE: CoinbaseAdapter,
        BrokerType.SCHWAB: SchwabAdapter,
        BrokerType.IBKR: IBKRAdapter,
    }
    factory = adapters.get(broker_type)
    if not factory:
        raise ValueError(f"Unknown broker type: {broker_type}")
    return factory()
