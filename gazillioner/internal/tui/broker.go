package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// BrokerConnection represents a broker connection for display
type BrokerConnection struct {
	ID           string
	BrokerType   string
	DisplayName  string
	AccountID    string
	Status       string
	PaperTrading bool
	ErrorMessage string
}

// BrokerModel represents the broker connections view
type BrokerModel struct {
	width  int
	height int
	styles Styles

	// State
	connections   []BrokerConnection
	selected      int
	syncing       bool
	awaitingOAuth bool
	oauthURL      string
	errorMsg      string

	// Components
	spinner spinner.Model
}

// NewBrokerModel creates a new broker model
func NewBrokerModel() *BrokerModel {
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("#7C3AED"))

	return &BrokerModel{
		styles:      DefaultStyles(),
		connections: []BrokerConnection{},
		spinner:     s,
	}
}

// SetSize updates the model dimensions
func (m *BrokerModel) SetSize(width, height int) {
	m.width = width
	m.height = height
}

// Init implements tea.Model
func (m *BrokerModel) Init() tea.Cmd {
	return m.spinner.Tick
}

// Update implements tea.Model
func (m *BrokerModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if m.selected > 0 {
				m.selected--
			}
		case "down", "j":
			if m.selected < len(m.connections)-1 {
				m.selected++
			}
		case "a":
			// Add new broker connection
			// TODO: Show broker selection menu
			return m, nil
		case "s":
			// Sync holdings from selected broker
			if len(m.connections) > 0 && m.connections[m.selected].Status == "connected" {
				m.syncing = true
				// TODO: Trigger sync
			}
			return m, nil
		case "d":
			// Disconnect selected broker
			// TODO: Implement disconnect
			return m, nil
		case "r":
			// Refresh connections
			// TODO: Refresh broker statuses
			return m, nil
		case "1":
			// Quick connect Alpaca
			m.awaitingOAuth = true
			m.oauthURL = "https://app.alpaca.markets/oauth/authorize?..."
			// TODO: Start OAuth flow
			return m, nil
		case "2":
			// Quick connect IBKR
			return m, nil
		case "3":
			// Quick connect Schwab
			return m, nil
		case "4":
			// Quick connect Coinbase
			return m, nil
		}

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		cmds = append(cmds, cmd)
	}

	return m, tea.Batch(cmds...)
}

// View implements tea.Model
func (m *BrokerModel) View() string {
	var b strings.Builder

	// Title
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#FFFFFF")).
		MarginBottom(1).
		Render("Brokerage Connections")
	b.WriteString(title)
	b.WriteString("\n\n")

	if len(m.connections) == 0 {
		b.WriteString(m.renderEmptyState())
	} else {
		b.WriteString(m.renderConnections())
	}

	b.WriteString("\n\n")

	if m.awaitingOAuth {
		b.WriteString(m.renderOAuthWaiting())
	} else if m.syncing {
		b.WriteString(m.renderSyncing())
	} else if m.errorMsg != "" {
		b.WriteString(m.renderError())
	}

	b.WriteString("\n")
	b.WriteString(m.renderHelp())

	return b.String()
}

func (m *BrokerModel) renderEmptyState() string {
	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#374151")).
		Padding(1, 2).
		Width(min(60, m.width-4))

	content := `No brokers connected.

Connect a brokerage to sync your portfolio:

  [1] Alpaca      - Stocks, ETFs, Crypto (Paper trading available)
  [2] Interactive Brokers - Full market access
  [3] Charles Schwab      - Stocks, ETFs, Options
  [4] Coinbase            - Cryptocurrency

Press a number key to connect, or 'a' for more options.`

	return cardStyle.Render(m.styles.Muted.Render(content))
}

func (m *BrokerModel) renderConnections() string {
	var rows strings.Builder

	for i, conn := range m.connections {
		cursor := "  "
		if i == m.selected {
			cursor = "> "
		}

		// Status indicator
		statusStyle := m.styles.Muted
		statusIcon := "○"
		switch conn.Status {
		case "connected":
			statusStyle = m.styles.Success
			statusIcon = "●"
		case "connecting":
			statusStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#F59E0B"))
			statusIcon = "◐"
		case "error", "token_expired":
			statusStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#EF4444"))
			statusIcon = "✗"
		}

		// Broker type icon
		brokerIcon := m.getBrokerIcon(conn.BrokerType)

		// Paper trading badge
		paperBadge := ""
		if conn.PaperTrading {
			paperBadge = lipgloss.NewStyle().
				Foreground(lipgloss.Color("#F59E0B")).
				Render(" [PAPER]")
		}

		row := fmt.Sprintf("%s%s %s %s%s",
			cursor,
			statusStyle.Render(statusIcon),
			brokerIcon,
			conn.DisplayName,
			paperBadge,
		)

		if conn.AccountID != "" {
			row += m.styles.Muted.Render(fmt.Sprintf(" (%s)", conn.AccountID))
		}

		if conn.ErrorMessage != "" && conn.Status == "error" {
			row += "\n    " + lipgloss.NewStyle().
				Foreground(lipgloss.Color("#EF4444")).
				Render(conn.ErrorMessage)
		}

		rows.WriteString(row)
		rows.WriteString("\n")
	}

	return rows.String()
}

func (m *BrokerModel) getBrokerIcon(brokerType string) string {
	switch brokerType {
	case "alpaca":
		return "🦙"
	case "ibkr":
		return "📊"
	case "schwab":
		return "💼"
	case "coinbase":
		return "₿"
	default:
		return "🏦"
	}
}

func (m *BrokerModel) renderOAuthWaiting() string {
	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#7C3AED")).
		Padding(1, 2).
		Width(min(60, m.width-4))

	content := fmt.Sprintf(`%s Waiting for authorization...

A browser window should open for you to log in.
If it doesn't open automatically, visit:

%s

Press 'Esc' to cancel`,
		m.spinner.View(),
		m.styles.Muted.Render(m.oauthURL),
	)

	return cardStyle.Render(content)
}

func (m *BrokerModel) renderSyncing() string {
	return fmt.Sprintf("%s Syncing positions from broker...", m.spinner.View())
}

func (m *BrokerModel) renderError() string {
	return lipgloss.NewStyle().
		Foreground(lipgloss.Color("#EF4444")).
		Render("Error: " + m.errorMsg)
}

func (m *BrokerModel) renderHelp() string {
	if len(m.connections) == 0 {
		return m.styles.Muted.Render("[1-4] Quick connect  [a] Add broker  [?] Help")
	}
	return m.styles.Muted.Render("[s] Sync holdings  [a] Add  [d] Disconnect  [r] Refresh  [↑↓] Navigate")
}

// AddConnection adds a broker connection to the list
func (m *BrokerModel) AddConnection(conn BrokerConnection) {
	m.connections = append(m.connections, conn)
}

// UpdateConnection updates an existing connection
func (m *BrokerModel) UpdateConnection(id string, status string, errorMsg string) {
	for i := range m.connections {
		if m.connections[i].ID == id {
			m.connections[i].Status = status
			m.connections[i].ErrorMessage = errorMsg
			break
		}
	}
}

// SetOAuthWaiting sets the OAuth waiting state
func (m *BrokerModel) SetOAuthWaiting(waiting bool, url string) {
	m.awaitingOAuth = waiting
	m.oauthURL = url
}

// SetSyncing sets the syncing state
func (m *BrokerModel) SetSyncing(syncing bool) {
	m.syncing = syncing
}

// SetError sets an error message
func (m *BrokerModel) SetError(msg string) {
	m.errorMsg = msg
}

// ClearError clears the error message
func (m *BrokerModel) ClearError() {
	m.errorMsg = ""
}

// GetSelectedConnection returns the currently selected connection
func (m *BrokerModel) GetSelectedConnection() *BrokerConnection {
	if len(m.connections) == 0 || m.selected >= len(m.connections) {
		return nil
	}
	return &m.connections[m.selected]
}

// SetMockData sets mock data for testing/demo
func (m *BrokerModel) SetMockData() {
	m.connections = []BrokerConnection{
		{
			ID:           "alpaca-1",
			BrokerType:   "alpaca",
			DisplayName:  "Alpaca Markets",
			AccountID:    "PA1234567",
			Status:       "connected",
			PaperTrading: true,
		},
		{
			ID:           "coinbase-1",
			BrokerType:   "coinbase",
			DisplayName:  "Coinbase",
			AccountID:    "user@example.com",
			Status:       "connected",
			PaperTrading: false,
		},
	}
}
