package broker

import (
	"context"
	"sync"
	"time"
)

// BrokerService defines the unified interface for all brokerages
type BrokerService interface {
	// Connection management
	Connect(ctx context.Context) error
	Disconnect(ctx context.Context) error
	Status() ConnectionStatus
	IsConnected() bool

	// Account information
	GetAccount(ctx context.Context) (*Account, error)
	GetBalances(ctx context.Context) (*Balances, error)

	// Position management
	GetPositions(ctx context.Context) ([]Position, error)
	GetPosition(ctx context.Context, symbol string) (*Position, error)

	// Order management (read-only first)
	GetOrders(ctx context.Context, filter OrderFilter) ([]Order, error)
	GetOrder(ctx context.Context, orderID string) (*Order, error)

	// Trading (Phase 2)
	PlaceOrder(ctx context.Context, req OrderRequest) (*Order, error)
	CancelOrder(ctx context.Context, orderID string) error

	// Metadata
	Type() BrokerType
	Name() string
	SupportedAssetClasses() []AssetClass
	IsPaperTrading() bool
}

// OAuthBroker interface for brokers using OAuth
type OAuthBroker interface {
	BrokerService
	GetAuthURL(state string) string
	ExchangeCode(ctx context.Context, code string) (*OAuthTokens, error)
	RefreshToken(ctx context.Context) error
	GetTokenExpiry() time.Time
}

// APIKeyBroker interface for brokers using API keys
type APIKeyBroker interface {
	BrokerService
	SetCredentials(apiKey, apiSecret string) error
}

// Manager manages multiple broker connections
type Manager struct {
	mu          sync.RWMutex
	connections map[string]BrokerService
	store       CredentialStore
}

// CredentialStore interface for encrypted credential storage
type CredentialStore interface {
	SaveConnection(conn *BrokerConnection, tokens *OAuthTokens) error
	LoadConnection(id string) (*BrokerConnection, *OAuthTokens, error)
	ListConnections() ([]BrokerConnection, error)
	DeleteConnection(id string) error
	UpdateTokens(id string, tokens *OAuthTokens) error
}

// NewManager creates a new broker manager
func NewManager(store CredentialStore) *Manager {
	return &Manager{
		connections: make(map[string]BrokerService),
		store:       store,
	}
}

// AddConnection adds a broker connection
func (m *Manager) AddConnection(id string, broker BrokerService) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.connections[id] = broker
}

// GetConnection retrieves a broker connection by ID
func (m *Manager) GetConnection(id string) (BrokerService, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	broker, ok := m.connections[id]
	return broker, ok
}

// RemoveConnection removes a broker connection
func (m *Manager) RemoveConnection(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if broker, ok := m.connections[id]; ok {
		// Disconnect if connected
		if broker.IsConnected() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			broker.Disconnect(ctx)
		}
		delete(m.connections, id)
	}
}

// ListConnections returns all active connections
func (m *Manager) ListConnections() []BrokerService {
	m.mu.RLock()
	defer m.mu.RUnlock()
	brokers := make([]BrokerService, 0, len(m.connections))
	for _, b := range m.connections {
		brokers = append(brokers, b)
	}
	return brokers
}

// SyncAllPositions syncs positions from all connected brokers
func (m *Manager) SyncAllPositions(ctx context.Context) (map[string][]Position, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	results := make(map[string][]Position)
	for id, broker := range m.connections {
		if !broker.IsConnected() {
			continue
		}
		positions, err := broker.GetPositions(ctx)
		if err != nil {
			// Log error but continue with other brokers
			continue
		}
		results[id] = positions
	}
	return results, nil
}

// GetTotalBalances aggregates balances from all connected brokers
func (m *Manager) GetTotalBalances(ctx context.Context) (*Balances, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	total := &Balances{
		Currency:       "USD",
		CryptoBalances: make(map[string]float64),
		AsOf:           time.Now(),
	}

	for _, broker := range m.connections {
		if !broker.IsConnected() {
			continue
		}
		balances, err := broker.GetBalances(ctx)
		if err != nil {
			continue
		}
		total.Cash += balances.Cash
		total.BuyingPower += balances.BuyingPower
		total.PortfolioValue += balances.PortfolioValue

		// Merge crypto balances
		for symbol, amount := range balances.CryptoBalances {
			total.CryptoBalances[symbol] += amount
		}
	}

	return total, nil
}
