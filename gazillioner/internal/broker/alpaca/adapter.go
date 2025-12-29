package alpaca

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gazillioner/gazillioner/internal/broker"
)

const (
	// API URLs
	PaperBaseURL = "https://paper-api.alpaca.markets"
	LiveBaseURL  = "https://api.alpaca.markets"
	DataURL      = "https://data.alpaca.markets"
	OAuthURL     = "https://app.alpaca.markets/oauth/authorize"
	TokenURL     = "https://api.alpaca.markets/oauth/token"

	// Rate limiting
	RequestsPerMinute = 200
)

// Adapter implements broker.BrokerService for Alpaca
type Adapter struct {
	mu sync.RWMutex

	// Configuration
	paperTrading bool
	baseURL      string

	// Authentication
	apiKey      string
	apiSecret   string
	accessToken string

	// OAuth config (for OAuth flow)
	clientID     string
	clientSecret string
	redirectURI  string

	// Token management
	tokenExpiry time.Time

	// State
	status    broker.ConnectionStatus
	accountID string

	// HTTP client
	client *http.Client

	// Rate limiting
	rateLimiter *rateLimiter
}

// Config holds Alpaca adapter configuration
type Config struct {
	PaperTrading bool
	APIKey       string
	APISecret    string
	ClientID     string
	ClientSecret string
	RedirectURI  string
}

// NewAdapter creates a new Alpaca adapter
func NewAdapter(cfg Config) *Adapter {
	baseURL := LiveBaseURL
	if cfg.PaperTrading {
		baseURL = PaperBaseURL
	}

	return &Adapter{
		paperTrading: cfg.PaperTrading,
		baseURL:      baseURL,
		apiKey:       cfg.APIKey,
		apiSecret:    cfg.APISecret,
		clientID:     cfg.ClientID,
		clientSecret: cfg.ClientSecret,
		redirectURI:  cfg.RedirectURI,
		status:       broker.StatusDisconnected,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		rateLimiter: newRateLimiter(RequestsPerMinute),
	}
}

// Connect establishes connection to Alpaca
func (a *Adapter) Connect(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.status == broker.StatusConnected {
		return broker.ErrAlreadyConnected
	}

	a.status = broker.StatusConnecting

	// Verify credentials by fetching account
	account, err := a.fetchAccount(ctx)
	if err != nil {
		a.status = broker.StatusError
		return err
	}

	a.accountID = account.AccountNumber
	a.status = broker.StatusConnected
	return nil
}

// Disconnect closes the connection
func (a *Adapter) Disconnect(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.status = broker.StatusDisconnected
	return nil
}

// Status returns the current connection status
func (a *Adapter) Status() broker.ConnectionStatus {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.status
}

// IsConnected returns true if connected
func (a *Adapter) IsConnected() bool {
	return a.Status() == broker.StatusConnected
}

// GetAccount returns account information
func (a *Adapter) GetAccount(ctx context.Context) (*broker.Account, error) {
	if !a.IsConnected() {
		return nil, broker.ErrNotConnected
	}

	account, err := a.fetchAccount(ctx)
	if err != nil {
		return nil, err
	}

	return &broker.Account{
		ID:               account.ID,
		BrokerType:       broker.BrokerAlpaca,
		AccountNumber:    account.AccountNumber,
		AccountType:      account.AccountType(),
		Currency:         account.Currency,
		Status:           account.Status,
		TradingEnabled:   account.TradingBlocked == false,
		CryptoEnabled:    account.CryptoStatus == "ACTIVE",
		MarginEnabled:    account.Multiplier > 1,
		PatternDayTrader: account.PatternDayTrader,
	}, nil
}

// GetBalances returns account balances
func (a *Adapter) GetBalances(ctx context.Context) (*broker.Balances, error) {
	if !a.IsConnected() {
		return nil, broker.ErrNotConnected
	}

	account, err := a.fetchAccount(ctx)
	if err != nil {
		return nil, err
	}

	return &broker.Balances{
		Cash:           account.Cash,
		BuyingPower:    account.BuyingPower,
		PortfolioValue: account.PortfolioValue,
		DayTradeCount:  account.DaytradeCount,
		Currency:       account.Currency,
		AsOf:           time.Now(),
	}, nil
}

// GetPositions returns all positions
func (a *Adapter) GetPositions(ctx context.Context) ([]broker.Position, error) {
	if !a.IsConnected() {
		return nil, broker.ErrNotConnected
	}

	positions, err := a.fetchPositions(ctx)
	if err != nil {
		return nil, err
	}

	result := make([]broker.Position, len(positions))
	for i, p := range positions {
		result[i] = broker.Position{
			Symbol:          p.Symbol,
			Quantity:        p.Qty,
			AvgCost:         p.AvgEntryPrice,
			CurrentPrice:    p.CurrentPrice,
			MarketValue:     p.MarketValue,
			UnrealizedPL:    p.UnrealizedPL,
			UnrealizedPLPct: p.UnrealizedPLPC * 100,
			Side:            p.Side,
			AssetClass:      parseAssetClass(p.AssetClass),
			Exchange:        p.Exchange,
		}
	}

	return result, nil
}

// GetPosition returns a specific position
func (a *Adapter) GetPosition(ctx context.Context, symbol string) (*broker.Position, error) {
	if !a.IsConnected() {
		return nil, broker.ErrNotConnected
	}

	p, err := a.fetchPosition(ctx, symbol)
	if err != nil {
		return nil, err
	}

	return &broker.Position{
		Symbol:          p.Symbol,
		Quantity:        p.Qty,
		AvgCost:         p.AvgEntryPrice,
		CurrentPrice:    p.CurrentPrice,
		MarketValue:     p.MarketValue,
		UnrealizedPL:    p.UnrealizedPL,
		UnrealizedPLPct: p.UnrealizedPLPC * 100,
		Side:            p.Side,
		AssetClass:      parseAssetClass(p.AssetClass),
		Exchange:        p.Exchange,
	}, nil
}

// GetOrders returns orders matching filter
func (a *Adapter) GetOrders(ctx context.Context, filter broker.OrderFilter) ([]broker.Order, error) {
	if !a.IsConnected() {
		return nil, broker.ErrNotConnected
	}

	orders, err := a.fetchOrders(ctx, filter.Status, filter.Limit)
	if err != nil {
		return nil, err
	}

	result := make([]broker.Order, len(orders))
	for i, o := range orders {
		result[i] = broker.Order{
			ID:            o.ID,
			ClientOrderID: o.ClientOrderID,
			Symbol:        o.Symbol,
			Side:          broker.OrderSide(o.Side),
			Type:          broker.OrderType(o.Type),
			TimeInForce:   broker.TimeInForce(o.TimeInForce),
			Quantity:      o.Qty,
			FilledQty:     o.FilledQty,
			Price:         o.LimitPrice,
			StopPrice:     o.StopPrice,
			Status:        broker.OrderStatus(o.Status),
			CreatedAt:     o.CreatedAt,
			AvgFillPrice:  o.FilledAvgPrice,
		}
		if o.FilledAt != nil {
			result[i].FilledAt = o.FilledAt
		}
	}

	return result, nil
}

// GetOrder returns a specific order
func (a *Adapter) GetOrder(ctx context.Context, orderID string) (*broker.Order, error) {
	if !a.IsConnected() {
		return nil, broker.ErrNotConnected
	}

	o, err := a.fetchOrder(ctx, orderID)
	if err != nil {
		return nil, err
	}

	order := &broker.Order{
		ID:            o.ID,
		ClientOrderID: o.ClientOrderID,
		Symbol:        o.Symbol,
		Side:          broker.OrderSide(o.Side),
		Type:          broker.OrderType(o.Type),
		TimeInForce:   broker.TimeInForce(o.TimeInForce),
		Quantity:      o.Qty,
		FilledQty:     o.FilledQty,
		Price:         o.LimitPrice,
		StopPrice:     o.StopPrice,
		Status:        broker.OrderStatus(o.Status),
		CreatedAt:     o.CreatedAt,
		AvgFillPrice:  o.FilledAvgPrice,
	}
	if o.FilledAt != nil {
		order.FilledAt = o.FilledAt
	}

	return order, nil
}

// PlaceOrder places a new order (Phase 2)
func (a *Adapter) PlaceOrder(ctx context.Context, req broker.OrderRequest) (*broker.Order, error) {
	if !a.IsConnected() {
		return nil, broker.ErrNotConnected
	}

	// Build order request
	orderReq := alpacaOrderRequest{
		Symbol:      req.Symbol,
		Qty:         req.Quantity,
		Side:        string(req.Side),
		Type:        string(req.Type),
		TimeInForce: string(req.TimeInForce),
	}
	if req.LimitPrice != nil {
		orderReq.LimitPrice = *req.LimitPrice
	}
	if req.StopPrice != nil {
		orderReq.StopPrice = *req.StopPrice
	}
	if req.Extended {
		orderReq.ExtendedHours = true
	}

	o, err := a.submitOrder(ctx, orderReq)
	if err != nil {
		return nil, err
	}

	return &broker.Order{
		ID:            o.ID,
		ClientOrderID: o.ClientOrderID,
		Symbol:        o.Symbol,
		Side:          broker.OrderSide(o.Side),
		Type:          broker.OrderType(o.Type),
		TimeInForce:   broker.TimeInForce(o.TimeInForce),
		Quantity:      o.Qty,
		Status:        broker.OrderStatus(o.Status),
		CreatedAt:     o.CreatedAt,
	}, nil
}

// CancelOrder cancels an existing order
func (a *Adapter) CancelOrder(ctx context.Context, orderID string) error {
	if !a.IsConnected() {
		return broker.ErrNotConnected
	}

	return a.deleteOrder(ctx, orderID)
}

// Type returns the broker type
func (a *Adapter) Type() broker.BrokerType {
	return broker.BrokerAlpaca
}

// Name returns the broker display name
func (a *Adapter) Name() string {
	if a.paperTrading {
		return "Alpaca (Paper)"
	}
	return "Alpaca"
}

// SupportedAssetClasses returns supported asset classes
func (a *Adapter) SupportedAssetClasses() []broker.AssetClass {
	return []broker.AssetClass{
		broker.AssetStock,
		broker.AssetETF,
		broker.AssetCrypto,
	}
}

// IsPaperTrading returns true if using paper trading
func (a *Adapter) IsPaperTrading() bool {
	return a.paperTrading
}

// OAuth methods

// GetAuthURL returns the OAuth authorization URL
func (a *Adapter) GetAuthURL(state string) string {
	params := url.Values{
		"response_type": {"code"},
		"client_id":     {a.clientID},
		"redirect_uri":  {a.redirectURI},
		"state":         {state},
		"scope":         {"account:write trading"},
	}
	return OAuthURL + "?" + params.Encode()
}

// ExchangeCode exchanges auth code for tokens
func (a *Adapter) ExchangeCode(ctx context.Context, code string) (*broker.OAuthTokens, error) {
	data := url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"client_id":     {a.clientID},
		"client_secret": {a.clientSecret},
		"redirect_uri":  {a.redirectURI},
	}

	req, err := http.NewRequestWithContext(ctx, "POST", TokenURL, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, &broker.APIError{
			StatusCode: resp.StatusCode,
			Message:    string(body),
		}
	}

	var tokenResp struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		TokenType    string `json:"token_type"`
		ExpiresIn    int    `json:"expires_in"`
		Scope        string `json:"scope"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return nil, err
	}

	a.mu.Lock()
	a.accessToken = tokenResp.AccessToken
	a.tokenExpiry = time.Now().Add(time.Duration(tokenResp.ExpiresIn) * time.Second)
	a.mu.Unlock()

	return &broker.OAuthTokens{
		AccessToken:  tokenResp.AccessToken,
		RefreshToken: tokenResp.RefreshToken,
		TokenType:    tokenResp.TokenType,
		ExpiresAt:    a.tokenExpiry,
		Scope:        tokenResp.Scope,
	}, nil
}

// RefreshToken refreshes the OAuth token
func (a *Adapter) RefreshToken(ctx context.Context) error {
	// Alpaca OAuth doesn't support refresh tokens currently
	// Need to re-authenticate
	return broker.ErrTokenRefreshFailed
}

// GetTokenExpiry returns when the token expires
func (a *Adapter) GetTokenExpiry() time.Time {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.tokenExpiry
}

// SetAccessToken sets the OAuth access token
func (a *Adapter) SetAccessToken(token string, expiry time.Time) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.accessToken = token
	a.tokenExpiry = expiry
}

// Internal API methods

func (a *Adapter) doRequest(ctx context.Context, method, path string, body io.Reader) (*http.Response, error) {
	// Rate limiting
	a.rateLimiter.wait()

	url := a.baseURL + path
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, err
	}

	// Set auth headers
	a.mu.RLock()
	if a.accessToken != "" {
		req.Header.Set("Authorization", "Bearer "+a.accessToken)
	} else if a.apiKey != "" {
		req.Header.Set("APCA-API-KEY-ID", a.apiKey)
		req.Header.Set("APCA-API-SECRET-KEY", a.apiSecret)
	}
	a.mu.RUnlock()

	req.Header.Set("Content-Type", "application/json")

	return a.client.Do(req)
}

func (a *Adapter) fetchAccount(ctx context.Context) (*alpacaAccount, error) {
	resp, err := a.doRequest(ctx, "GET", "/v2/account", nil)
	if err != nil {
		return nil, broker.NewBrokerError(broker.BrokerAlpaca, "GetAccount", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, a.handleErrorResponse(resp, "GetAccount")
	}

	var account alpacaAccount
	if err := json.NewDecoder(resp.Body).Decode(&account); err != nil {
		return nil, err
	}

	return &account, nil
}

func (a *Adapter) fetchPositions(ctx context.Context) ([]alpacaPosition, error) {
	resp, err := a.doRequest(ctx, "GET", "/v2/positions", nil)
	if err != nil {
		return nil, broker.NewBrokerError(broker.BrokerAlpaca, "GetPositions", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, a.handleErrorResponse(resp, "GetPositions")
	}

	var positions []alpacaPosition
	if err := json.NewDecoder(resp.Body).Decode(&positions); err != nil {
		return nil, err
	}

	return positions, nil
}

func (a *Adapter) fetchPosition(ctx context.Context, symbol string) (*alpacaPosition, error) {
	resp, err := a.doRequest(ctx, "GET", "/v2/positions/"+symbol, nil)
	if err != nil {
		return nil, broker.NewBrokerError(broker.BrokerAlpaca, "GetPosition", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, broker.ErrPositionNotFound
	}
	if resp.StatusCode != http.StatusOK {
		return nil, a.handleErrorResponse(resp, "GetPosition")
	}

	var position alpacaPosition
	if err := json.NewDecoder(resp.Body).Decode(&position); err != nil {
		return nil, err
	}

	return &position, nil
}

func (a *Adapter) fetchOrders(ctx context.Context, status string, limit int) ([]alpacaOrder, error) {
	path := "/v2/orders"
	if status != "" || limit > 0 {
		params := url.Values{}
		if status != "" {
			params.Set("status", status)
		}
		if limit > 0 {
			params.Set("limit", fmt.Sprintf("%d", limit))
		}
		path += "?" + params.Encode()
	}

	resp, err := a.doRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, broker.NewBrokerError(broker.BrokerAlpaca, "GetOrders", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, a.handleErrorResponse(resp, "GetOrders")
	}

	var orders []alpacaOrder
	if err := json.NewDecoder(resp.Body).Decode(&orders); err != nil {
		return nil, err
	}

	return orders, nil
}

func (a *Adapter) fetchOrder(ctx context.Context, orderID string) (*alpacaOrder, error) {
	resp, err := a.doRequest(ctx, "GET", "/v2/orders/"+orderID, nil)
	if err != nil {
		return nil, broker.NewBrokerError(broker.BrokerAlpaca, "GetOrder", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, broker.ErrOrderNotFound
	}
	if resp.StatusCode != http.StatusOK {
		return nil, a.handleErrorResponse(resp, "GetOrder")
	}

	var order alpacaOrder
	if err := json.NewDecoder(resp.Body).Decode(&order); err != nil {
		return nil, err
	}

	return &order, nil
}

func (a *Adapter) submitOrder(ctx context.Context, req alpacaOrderRequest) (*alpacaOrder, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := a.doRequest(ctx, "POST", "/v2/orders", strings.NewReader(string(body)))
	if err != nil {
		return nil, broker.NewBrokerError(broker.BrokerAlpaca, "PlaceOrder", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return nil, a.handleErrorResponse(resp, "PlaceOrder")
	}

	var order alpacaOrder
	if err := json.NewDecoder(resp.Body).Decode(&order); err != nil {
		return nil, err
	}

	return &order, nil
}

func (a *Adapter) deleteOrder(ctx context.Context, orderID string) error {
	resp, err := a.doRequest(ctx, "DELETE", "/v2/orders/"+orderID, nil)
	if err != nil {
		return broker.NewBrokerError(broker.BrokerAlpaca, "CancelOrder", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return broker.ErrOrderNotFound
	}
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		return a.handleErrorResponse(resp, "CancelOrder")
	}

	return nil
}

func (a *Adapter) handleErrorResponse(resp *http.Response, op string) error {
	body, _ := io.ReadAll(resp.Body)

	var errResp struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	}
	json.Unmarshal(body, &errResp)

	apiErr := &broker.APIError{
		StatusCode: resp.StatusCode,
		Code:       fmt.Sprintf("%d", errResp.Code),
		Message:    errResp.Message,
	}

	if resp.StatusCode == 401 || resp.StatusCode == 403 {
		return broker.NewBrokerError(broker.BrokerAlpaca, op, broker.ErrInvalidCredentials).
			WithCode(apiErr.Code).WithMessage(apiErr.Message)
	}
	if resp.StatusCode == 429 {
		return broker.ErrRateLimited
	}

	return broker.NewBrokerError(broker.BrokerAlpaca, op, apiErr)
}

func parseAssetClass(s string) broker.AssetClass {
	switch strings.ToLower(s) {
	case "us_equity":
		return broker.AssetStock
	case "crypto":
		return broker.AssetCrypto
	default:
		return broker.AssetStock
	}
}

// Rate limiter
type rateLimiter struct {
	mu       sync.Mutex
	requests []time.Time
	limit    int
}

func newRateLimiter(requestsPerMinute int) *rateLimiter {
	return &rateLimiter{
		requests: make([]time.Time, 0),
		limit:    requestsPerMinute,
	}
}

func (r *rateLimiter) wait() {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-time.Minute)

	// Remove old requests
	valid := r.requests[:0]
	for _, t := range r.requests {
		if t.After(cutoff) {
			valid = append(valid, t)
		}
	}
	r.requests = valid

	// Check if at limit
	if len(r.requests) >= r.limit {
		waitTime := r.requests[0].Add(time.Minute).Sub(now)
		if waitTime > 0 {
			time.Sleep(waitTime)
		}
	}

	r.requests = append(r.requests, now)
}
