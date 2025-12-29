package alpaca

import "time"

// Alpaca API response types

// alpacaAccount represents the Alpaca account response
type alpacaAccount struct {
	ID                    string  `json:"id"`
	AccountNumber         string  `json:"account_number"`
	Status                string  `json:"status"`
	Currency              string  `json:"currency"`
	Cash                  string  `json:"cash"`
	PortfolioValue        string  `json:"portfolio_value"`
	PatternDayTrader      bool    `json:"pattern_day_trader"`
	TradeSuspendedByUser  bool    `json:"trade_suspended_by_user"`
	TradingBlocked        bool    `json:"trading_blocked"`
	TransfersBlocked      bool    `json:"transfers_blocked"`
	AccountBlocked        bool    `json:"account_blocked"`
	ShortingEnabled       bool    `json:"shorting_enabled"`
	LongMarketValue       string  `json:"long_market_value"`
	ShortMarketValue      string  `json:"short_market_value"`
	Equity                string  `json:"equity"`
	LastEquity            string  `json:"last_equity"`
	Multiplier            string  `json:"multiplier"`
	BuyingPower           string  `json:"buying_power"`
	InitialMargin         string  `json:"initial_margin"`
	MaintenanceMargin     string  `json:"maintenance_margin"`
	SMA                   string  `json:"sma"`
	DaytradingBuyingPower string  `json:"daytrading_buying_power"`
	LastMaintenanceMargin string  `json:"last_maintenance_margin"`
	DaytradeCount         int     `json:"daytrade_count"`
	RegtBuyingPower       string  `json:"regt_buying_power"`
	PendingTransferIn     string  `json:"pending_transfer_in"`
	PendingTransferOut    string  `json:"pending_transfer_out"`
	CreatedAt             string  `json:"created_at"`
}

// alpacaPosition represents an Alpaca position response
type alpacaPosition struct {
	AssetID           string `json:"asset_id"`
	Symbol            string `json:"symbol"`
	Exchange          string `json:"exchange"`
	AssetClass        string `json:"asset_class"`
	AssetMarginable   bool   `json:"asset_marginable"`
	Qty               string `json:"qty"`
	AvgEntryPrice     string `json:"avg_entry_price"`
	Side              string `json:"side"`
	MarketValue       string `json:"market_value"`
	CostBasis         string `json:"cost_basis"`
	UnrealizedPL      string `json:"unrealized_pl"`
	UnrealizedPLPC    string `json:"unrealized_plpc"`
	UnrealizedIntradayPL   string `json:"unrealized_intraday_pl"`
	UnrealizedIntradayPLPC string `json:"unrealized_intraday_plpc"`
	CurrentPrice      string `json:"current_price"`
	LastdayPrice      string `json:"lastday_price"`
	ChangeToday       string `json:"change_today"`
	QtyAvailable      string `json:"qty_available"`
}

// alpacaOrder represents an Alpaca order response
type alpacaOrder struct {
	ID             string     `json:"id"`
	ClientOrderID  string     `json:"client_order_id"`
	CreatedAt      time.Time  `json:"created_at"`
	UpdatedAt      time.Time  `json:"updated_at"`
	SubmittedAt    time.Time  `json:"submitted_at"`
	FilledAt       *time.Time `json:"filled_at"`
	ExpiredAt      *time.Time `json:"expired_at"`
	CanceledAt     *time.Time `json:"canceled_at"`
	FailedAt       *time.Time `json:"failed_at"`
	ReplacedAt     *time.Time `json:"replaced_at"`
	ReplacedBy     *string    `json:"replaced_by"`
	Replaces       *string    `json:"replaces"`
	AssetID        string     `json:"asset_id"`
	Symbol         string     `json:"symbol"`
	AssetClass     string     `json:"asset_class"`
	Notional       *string    `json:"notional"`
	Qty            string     `json:"qty"`
	FilledQty      string     `json:"filled_qty"`
	FilledAvgPrice *string    `json:"filled_avg_price"`
	OrderClass     string     `json:"order_class"`
	OrderType      string     `json:"order_type"`
	Type           string     `json:"type"`
	Side           string     `json:"side"`
	TimeInForce    string     `json:"time_in_force"`
	LimitPrice     *string    `json:"limit_price"`
	StopPrice      *string    `json:"stop_price"`
	Status         string     `json:"status"`
	ExtendedHours  bool       `json:"extended_hours"`
	Legs           []alpacaOrder `json:"legs,omitempty"`
	TrailPercent   *string    `json:"trail_percent"`
	TrailPrice     *string    `json:"trail_price"`
	HWM            *string    `json:"hwm"`
}

// alpacaOrderRequest represents the request body for creating an order
type alpacaOrderRequest struct {
	Symbol        string  `json:"symbol"`
	Qty           string  `json:"qty,omitempty"`
	Notional      string  `json:"notional,omitempty"`
	Side          string  `json:"side"`
	Type          string  `json:"type"`
	TimeInForce   string  `json:"time_in_force"`
	LimitPrice    string  `json:"limit_price,omitempty"`
	StopPrice     string  `json:"stop_price,omitempty"`
	TrailPrice    string  `json:"trail_price,omitempty"`
	TrailPercent  string  `json:"trail_percent,omitempty"`
	ExtendedHours bool    `json:"extended_hours,omitempty"`
	ClientOrderID string  `json:"client_order_id,omitempty"`
	OrderClass    string  `json:"order_class,omitempty"`
}

// alpacaOAuthToken represents the OAuth token response
type alpacaOAuthToken struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	Scope        string `json:"scope"`
}

// alpacaError represents an error response from Alpaca
type alpacaError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// Clock represents market clock information
type alpacaClock struct {
	Timestamp string `json:"timestamp"`
	IsOpen    bool   `json:"is_open"`
	NextOpen  string `json:"next_open"`
	NextClose string `json:"next_close"`
}

// alpacaAsset represents asset information
type alpacaAsset struct {
	ID           string `json:"id"`
	Class        string `json:"class"`
	Exchange     string `json:"exchange"`
	Symbol       string `json:"symbol"`
	Name         string `json:"name"`
	Status       string `json:"status"`
	Tradable     bool   `json:"tradable"`
	Marginable   bool   `json:"marginable"`
	Shortable    bool   `json:"shortable"`
	EasyToBorrow bool   `json:"easy_to_borrow"`
	Fractionable bool   `json:"fractionable"`
}

// alpacaBar represents OHLCV data
type alpacaBar struct {
	Timestamp  time.Time `json:"t"`
	Open       float64   `json:"o"`
	High       float64   `json:"h"`
	Low        float64   `json:"l"`
	Close      float64   `json:"c"`
	Volume     uint64    `json:"v"`
	TradeCount uint64    `json:"n"`
	VWAP       float64   `json:"vw"`
}

// alpacaQuote represents a quote
type alpacaQuote struct {
	Timestamp   time.Time `json:"t"`
	AskExchange string    `json:"ax"`
	AskPrice    float64   `json:"ap"`
	AskSize     uint32    `json:"as"`
	BidExchange string    `json:"bx"`
	BidPrice    float64   `json:"bp"`
	BidSize     uint32    `json:"bs"`
	Conditions  []string  `json:"c"`
	Tape        string    `json:"z"`
}

// alpacaTrade represents a trade
type alpacaTrade struct {
	Timestamp  time.Time `json:"t"`
	Exchange   string    `json:"x"`
	Price      float64   `json:"p"`
	Size       uint32    `json:"s"`
	ID         int64     `json:"i"`
	Conditions []string  `json:"c"`
	Tape       string    `json:"z"`
}
