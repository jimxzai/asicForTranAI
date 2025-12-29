package broker

import (
	"time"
)

// BrokerType identifies the brokerage
type BrokerType string

const (
	BrokerAlpaca   BrokerType = "alpaca"
	BrokerIBKR     BrokerType = "ibkr"
	BrokerSchwab   BrokerType = "schwab"
	BrokerCoinbase BrokerType = "coinbase"
)

// ConnectionStatus represents broker connection state
type ConnectionStatus string

const (
	StatusDisconnected  ConnectionStatus = "disconnected"
	StatusConnecting    ConnectionStatus = "connecting"
	StatusConnected     ConnectionStatus = "connected"
	StatusError         ConnectionStatus = "error"
	StatusTokenExpired  ConnectionStatus = "token_expired"
)

// Account represents a brokerage account
type Account struct {
	ID              string
	BrokerType      BrokerType
	AccountNumber   string
	AccountType     string // "margin", "cash", "ira", etc.
	Currency        string
	Status          string
	CreatedAt       time.Time

	// Permissions
	TradingEnabled   bool
	OptionsEnabled   bool
	CryptoEnabled    bool
	MarginEnabled    bool
	PatternDayTrader bool
}

// Balances represents account balances
type Balances struct {
	Cash           float64
	BuyingPower    float64
	PortfolioValue float64
	DayTradeCount  int
	Currency       string

	// Margin-specific
	MarginUsed      float64
	MarginAvailable float64

	// Crypto-specific (for Coinbase)
	CryptoBalances map[string]float64

	AsOf time.Time
}

// Position represents a holding at the broker
type Position struct {
	Symbol          string
	Quantity        float64
	AvgCost         float64
	CurrentPrice    float64
	MarketValue     float64
	UnrealizedPL    float64
	UnrealizedPLPct float64
	Side            string // "long" or "short"
	AssetClass      AssetClass
	Exchange        string
}

// Order represents a trade order
type Order struct {
	ID            string
	ClientOrderID string
	Symbol        string
	Side          OrderSide
	Type          OrderType
	TimeInForce   TimeInForce
	Quantity      float64
	FilledQty     float64
	Price         float64 // Limit price
	StopPrice     float64 // Stop price
	Status        OrderStatus
	CreatedAt     time.Time
	FilledAt      *time.Time
	AvgFillPrice  float64
	Commission    float64
}

// OrderRequest for placing new orders
type OrderRequest struct {
	Symbol      string
	Side        OrderSide
	Type        OrderType
	TimeInForce TimeInForce
	Quantity    float64
	LimitPrice  *float64
	StopPrice   *float64
	Extended    bool // Extended hours
}

// OrderFilter for querying orders
type OrderFilter struct {
	Status string // "open", "closed", "all"
	Limit  int
	After  *time.Time
}

// OAuthTokens holds OAuth credentials
type OAuthTokens struct {
	AccessToken  string
	RefreshToken string
	TokenType    string
	ExpiresAt    time.Time
	Scope        string
}

// BrokerConnection represents a stored broker connection
type BrokerConnection struct {
	ID           string
	BrokerType   BrokerType
	DisplayName  string
	AccountID    string
	Status       ConnectionStatus
	PaperTrading bool
	ConnectedAt  time.Time
	TokenExpires *time.Time
	ErrorMessage string
}

// AssetClass categorizes tradeable assets
type AssetClass string

const (
	AssetStock  AssetClass = "stock"
	AssetETF    AssetClass = "etf"
	AssetCrypto AssetClass = "crypto"
	AssetOption AssetClass = "option"
	AssetForex  AssetClass = "forex"
	AssetFuture AssetClass = "future"
)

// OrderSide indicates buy or sell
type OrderSide string

const (
	OrderBuy  OrderSide = "buy"
	OrderSell OrderSide = "sell"
)

// OrderType defines order execution type
type OrderType string

const (
	OrderMarket       OrderType = "market"
	OrderLimit        OrderType = "limit"
	OrderStop         OrderType = "stop"
	OrderStopLimit    OrderType = "stop_limit"
	OrderTrailingStop OrderType = "trailing_stop"
)

// OrderStatus tracks order lifecycle
type OrderStatus string

const (
	OrderStatusNew       OrderStatus = "new"
	OrderStatusAccepted  OrderStatus = "accepted"
	OrderStatusPending   OrderStatus = "pending"
	OrderStatusFilled    OrderStatus = "filled"
	OrderStatusPartial   OrderStatus = "partially_filled"
	OrderStatusCanceled  OrderStatus = "canceled"
	OrderStatusRejected  OrderStatus = "rejected"
	OrderStatusExpired   OrderStatus = "expired"
)

// TimeInForce specifies order duration
type TimeInForce string

const (
	TIFDay TimeInForce = "day"
	TIFGTC TimeInForce = "gtc" // Good-til-canceled
	TIFIOC TimeInForce = "ioc" // Immediate-or-cancel
	TIFFOK TimeInForce = "fok" // Fill-or-kill
)

// String returns display name for broker type
func (b BrokerType) String() string {
	switch b {
	case BrokerAlpaca:
		return "Alpaca"
	case BrokerIBKR:
		return "Interactive Brokers"
	case BrokerSchwab:
		return "Charles Schwab"
	case BrokerCoinbase:
		return "Coinbase"
	default:
		return string(b)
	}
}

// SupportedAssetClasses returns asset classes supported by each broker
func (b BrokerType) SupportedAssetClasses() []AssetClass {
	switch b {
	case BrokerAlpaca:
		return []AssetClass{AssetStock, AssetETF, AssetCrypto}
	case BrokerIBKR:
		return []AssetClass{AssetStock, AssetETF, AssetOption, AssetForex, AssetFuture}
	case BrokerSchwab:
		return []AssetClass{AssetStock, AssetETF, AssetOption}
	case BrokerCoinbase:
		return []AssetClass{AssetCrypto}
	default:
		return []AssetClass{}
	}
}
