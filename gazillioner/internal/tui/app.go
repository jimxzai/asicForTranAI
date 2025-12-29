// Package tui implements the terminal user interface using Bubbletea
package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// View represents different screens in the TUI
type View int

const (
	ViewDashboard View = iota
	ViewPortfolio
	ViewWatchlist
	ViewChat
	ViewMarket
	ViewWallet
	ViewRiskMetrics
	ViewBroker
	ViewSettings
	ViewHelp
)

// KeyMap defines keyboard shortcuts
type KeyMap struct {
	Quit       key.Binding
	Help       key.Binding
	Tab        key.Binding
	ShiftTab   key.Binding
	Enter      key.Binding
	Escape     key.Binding
	Up         key.Binding
	Down       key.Binding
	Left       key.Binding
	Right      key.Binding
	Home       key.Binding
	End        key.Binding
	PageUp     key.Binding
	PageDown   key.Binding
	Delete     key.Binding
	Refresh    key.Binding
	Search     key.Binding
	New        key.Binding
	Dashboard   key.Binding
	Portfolio   key.Binding
	Watchlist   key.Binding
	Chat        key.Binding
	Market      key.Binding
	Wallet      key.Binding
	RiskMetrics key.Binding
	Broker      key.Binding
	Settings    key.Binding
}

// DefaultKeyMap returns the default key bindings
func DefaultKeyMap() KeyMap {
	return KeyMap{
		Quit:       key.NewBinding(key.WithKeys("q", "ctrl+c"), key.WithHelp("q", "quit")),
		Help:       key.NewBinding(key.WithKeys("?"), key.WithHelp("?", "help")),
		Tab:        key.NewBinding(key.WithKeys("tab"), key.WithHelp("tab", "next")),
		ShiftTab:   key.NewBinding(key.WithKeys("shift+tab"), key.WithHelp("shift+tab", "prev")),
		Enter:      key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "select")),
		Escape:     key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "back")),
		Up:         key.NewBinding(key.WithKeys("up", "k"), key.WithHelp("↑/k", "up")),
		Down:       key.NewBinding(key.WithKeys("down", "j"), key.WithHelp("↓/j", "down")),
		Left:       key.NewBinding(key.WithKeys("left", "h"), key.WithHelp("←/h", "left")),
		Right:      key.NewBinding(key.WithKeys("right", "l"), key.WithHelp("→/l", "right")),
		Home:       key.NewBinding(key.WithKeys("home", "g"), key.WithHelp("home", "top")),
		End:        key.NewBinding(key.WithKeys("end", "G"), key.WithHelp("end", "bottom")),
		PageUp:     key.NewBinding(key.WithKeys("pgup", "ctrl+u"), key.WithHelp("pgup", "page up")),
		PageDown:   key.NewBinding(key.WithKeys("pgdown", "ctrl+d"), key.WithHelp("pgdn", "page down")),
		Delete:     key.NewBinding(key.WithKeys("d", "delete"), key.WithHelp("d", "delete")),
		Refresh:    key.NewBinding(key.WithKeys("r"), key.WithHelp("r", "refresh")),
		Search:     key.NewBinding(key.WithKeys("/"), key.WithHelp("/", "search")),
		New:        key.NewBinding(key.WithKeys("n"), key.WithHelp("n", "new")),
		Dashboard:  key.NewBinding(key.WithKeys("1"), key.WithHelp("1", "dashboard")),
		Portfolio:  key.NewBinding(key.WithKeys("2"), key.WithHelp("2", "portfolio")),
		Watchlist:  key.NewBinding(key.WithKeys("3"), key.WithHelp("3", "watchlist")),
		Chat:       key.NewBinding(key.WithKeys("4"), key.WithHelp("4", "chat")),
		Market:      key.NewBinding(key.WithKeys("5"), key.WithHelp("5", "market")),
		Wallet:      key.NewBinding(key.WithKeys("6"), key.WithHelp("6", "wallet")),
		RiskMetrics: key.NewBinding(key.WithKeys("7"), key.WithHelp("7", "risk")),
		Broker:      key.NewBinding(key.WithKeys("8"), key.WithHelp("8", "broker")),
		Settings:    key.NewBinding(key.WithKeys("0"), key.WithHelp("0", "settings")),
	}
}

// Model is the main TUI model
type Model struct {
	// Core state
	currentView View
	width       int
	height      int
	ready       bool
	err         error

	// Submodels
	dashboard   *DashboardModel
	portfolio   *PortfolioModel
	chat        *ChatModel
	riskMetrics *RiskMetricsModel
	broker      *BrokerModel

	// UI components
	spinner spinner.Model
	keys    KeyMap

	// Styles
	styles Styles
}

// Styles holds all lipgloss styles
type Styles struct {
	App           lipgloss.Style
	Header        lipgloss.Style
	HeaderTitle   lipgloss.Style
	HeaderNav     lipgloss.Style
	NavItem       lipgloss.Style
	NavItemActive lipgloss.Style
	Content       lipgloss.Style
	Footer        lipgloss.Style
	StatusBar     lipgloss.Style
	Error         lipgloss.Style
	Success       lipgloss.Style
	Warning       lipgloss.Style
	Muted         lipgloss.Style
	Bold          lipgloss.Style
}

// DefaultStyles returns the default style configuration
func DefaultStyles() Styles {
	primary := lipgloss.Color("#7C3AED")    // Purple
	success := lipgloss.Color("#10B981")    // Green
	warning := lipgloss.Color("#F59E0B")    // Amber
	danger := lipgloss.Color("#EF4444")     // Red
	muted := lipgloss.Color("#6B7280")      // Gray
	bgDark := lipgloss.Color("#1F2937")     // Dark background
	bgLight := lipgloss.Color("#374151")    // Lighter background

	return Styles{
		App: lipgloss.NewStyle().
			Background(bgDark),

		Header: lipgloss.NewStyle().
			Background(bgLight).
			Padding(0, 1),

		HeaderTitle: lipgloss.NewStyle().
			Foreground(primary).
			Bold(true).
			Padding(0, 1),

		HeaderNav: lipgloss.NewStyle().
			Padding(0, 1),

		NavItem: lipgloss.NewStyle().
			Foreground(muted).
			Padding(0, 1),

		NavItemActive: lipgloss.NewStyle().
			Foreground(primary).
			Bold(true).
			Padding(0, 1).
			Underline(true),

		Content: lipgloss.NewStyle().
			Padding(1, 2),

		Footer: lipgloss.NewStyle().
			Background(bgLight).
			Foreground(muted).
			Padding(0, 1),

		StatusBar: lipgloss.NewStyle().
			Background(bgLight).
			Padding(0, 1),

		Error: lipgloss.NewStyle().
			Foreground(danger),

		Success: lipgloss.NewStyle().
			Foreground(success),

		Warning: lipgloss.NewStyle().
			Foreground(warning),

		Muted: lipgloss.NewStyle().
			Foreground(muted),

		Bold: lipgloss.NewStyle().
			Bold(true),
	}
}

// New creates a new TUI model
func New() Model {
	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("#7C3AED"))

	rm := NewRiskMetricsModel()
	rm.SetMockData() // Load demo data

	bm := NewBrokerModel()
	bm.SetMockData() // Load demo data

	return Model{
		currentView: ViewDashboard,
		spinner:     s,
		keys:        DefaultKeyMap(),
		styles:      DefaultStyles(),
		dashboard:   NewDashboardModel(),
		portfolio:   NewPortfolioModel(),
		chat:        NewChatModel(),
		riskMetrics: rm,
		broker:      bm,
	}
}

// Init implements tea.Model
func (m Model) Init() tea.Cmd {
	return tea.Batch(
		m.spinner.Tick,
		tea.EnterAltScreen,
	)
}

// Update implements tea.Model
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		// Global key handlers
		switch {
		case key.Matches(msg, m.keys.Quit):
			return m, tea.Quit

		case key.Matches(msg, m.keys.Help):
			m.currentView = ViewHelp
			return m, nil

		case key.Matches(msg, m.keys.Dashboard):
			m.currentView = ViewDashboard
			return m, nil

		case key.Matches(msg, m.keys.Portfolio):
			m.currentView = ViewPortfolio
			return m, nil

		case key.Matches(msg, m.keys.Watchlist):
			m.currentView = ViewWatchlist
			return m, nil

		case key.Matches(msg, m.keys.Chat):
			m.currentView = ViewChat
			return m, nil

		case key.Matches(msg, m.keys.Market):
			m.currentView = ViewMarket
			return m, nil

		case key.Matches(msg, m.keys.Wallet):
			m.currentView = ViewWallet
			return m, nil

		case key.Matches(msg, m.keys.RiskMetrics):
			m.currentView = ViewRiskMetrics
			return m, nil

		case key.Matches(msg, m.keys.Broker):
			m.currentView = ViewBroker
			return m, nil

		case key.Matches(msg, m.keys.Settings):
			m.currentView = ViewSettings
			return m, nil

		case key.Matches(msg, m.keys.Escape):
			if m.currentView != ViewDashboard {
				m.currentView = ViewDashboard
				return m, nil
			}
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.ready = true

		// Update submodels with new size
		if m.dashboard != nil {
			m.dashboard.SetSize(msg.Width, msg.Height-4) // Account for header/footer
		}
		if m.portfolio != nil {
			m.portfolio.SetSize(msg.Width, msg.Height-4)
		}
		if m.chat != nil {
			m.chat.SetSize(msg.Width, msg.Height-4)
		}
		if m.riskMetrics != nil {
			m.riskMetrics.SetSize(msg.Width, msg.Height-4)
		}
		if m.broker != nil {
			m.broker.SetSize(msg.Width, msg.Height-4)
		}

	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		cmds = append(cmds, cmd)

	case error:
		m.err = msg
	}

	// Update current view
	switch m.currentView {
	case ViewDashboard:
		if m.dashboard != nil {
			newModel, cmd := m.dashboard.Update(msg)
			m.dashboard = newModel.(*DashboardModel)
			cmds = append(cmds, cmd)
		}
	case ViewPortfolio:
		if m.portfolio != nil {
			newModel, cmd := m.portfolio.Update(msg)
			m.portfolio = newModel.(*PortfolioModel)
			cmds = append(cmds, cmd)
		}
	case ViewChat:
		if m.chat != nil {
			newModel, cmd := m.chat.Update(msg)
			m.chat = newModel.(*ChatModel)
			cmds = append(cmds, cmd)
		}
	case ViewRiskMetrics:
		if m.riskMetrics != nil {
			newModel, cmd := m.riskMetrics.Update(msg)
			m.riskMetrics = newModel.(*RiskMetricsModel)
			cmds = append(cmds, cmd)
		}
	case ViewBroker:
		if m.broker != nil {
			newModel, cmd := m.broker.Update(msg)
			m.broker = newModel.(*BrokerModel)
			cmds = append(cmds, cmd)
		}
	}

	return m, tea.Batch(cmds...)
}

// View implements tea.Model
func (m Model) View() string {
	if !m.ready {
		return m.spinner.View() + " Loading..."
	}

	var b strings.Builder

	// Header
	b.WriteString(m.renderHeader())
	b.WriteString("\n")

	// Content
	content := m.renderContent()
	b.WriteString(content)

	// Footer
	b.WriteString("\n")
	b.WriteString(m.renderFooter())

	return b.String()
}

func (m Model) renderHeader() string {
	title := m.styles.HeaderTitle.Render("GAZILLIONER")

	// Navigation items
	navItems := []struct {
		key  string
		name string
		view View
	}{
		{"1", "Dashboard", ViewDashboard},
		{"2", "Portfolio", ViewPortfolio},
		{"3", "Watchlist", ViewWatchlist},
		{"4", "Chat", ViewChat},
		{"5", "Market", ViewMarket},
		{"6", "Wallet", ViewWallet},
		{"7", "Risk", ViewRiskMetrics},
		{"8", "Broker", ViewBroker},
		{"0", "Settings", ViewSettings},
	}

	var nav strings.Builder
	for _, item := range navItems {
		style := m.styles.NavItem
		if item.view == m.currentView {
			style = m.styles.NavItemActive
		}
		nav.WriteString(style.Render(fmt.Sprintf("[%s] %s", item.key, item.name)))
		nav.WriteString(" ")
	}

	header := lipgloss.JoinHorizontal(
		lipgloss.Left,
		title,
		"  ",
		nav.String(),
	)

	return m.styles.Header.Width(m.width).Render(header)
}

func (m Model) renderContent() string {
	var content string

	switch m.currentView {
	case ViewDashboard:
		if m.dashboard != nil {
			content = m.dashboard.View()
		} else {
			content = "Dashboard loading..."
		}
	case ViewPortfolio:
		if m.portfolio != nil {
			content = m.portfolio.View()
		} else {
			content = "Portfolio loading..."
		}
	case ViewWatchlist:
		content = m.renderWatchlistView()
	case ViewChat:
		if m.chat != nil {
			content = m.chat.View()
		} else {
			content = "Chat loading..."
		}
	case ViewMarket:
		content = m.renderMarketView()
	case ViewWallet:
		content = m.renderWalletView()
	case ViewRiskMetrics:
		if m.riskMetrics != nil {
			content = m.riskMetrics.View()
		} else {
			content = "Risk Metrics loading..."
		}
	case ViewBroker:
		if m.broker != nil {
			content = m.broker.View()
		} else {
			content = "Broker loading..."
		}
	case ViewSettings:
		content = m.renderSettingsView()
	case ViewHelp:
		content = m.renderHelpView()
	default:
		content = "Unknown view"
	}

	// Apply content style and ensure it fills available space
	contentHeight := m.height - 4 // Account for header and footer
	return m.styles.Content.
		Width(m.width).
		Height(contentHeight).
		Render(content)
}

func (m Model) renderFooter() string {
	help := m.styles.Muted.Render("[?] Help  [q] Quit  [1-8,0] Navigate")

	status := m.styles.Success.Render("● Connected")
	if m.err != nil {
		status = m.styles.Error.Render("● " + m.err.Error())
	}

	footer := lipgloss.JoinHorizontal(
		lipgloss.Left,
		help,
		strings.Repeat(" ", max(0, m.width-lipgloss.Width(help)-lipgloss.Width(status)-4)),
		status,
	)

	return m.styles.Footer.Width(m.width).Render(footer)
}

// Placeholder views - to be expanded
func (m Model) renderWatchlistView() string {
	return `Watchlist

TICKER    PRICE      CHANGE     52W HIGH   52W LOW
------    -----      ------     --------   -------
[Empty - Press 'n' to add a ticker]

Press 'n' to add, 'd' to delete, 'r' to refresh`
}

func (m Model) renderMarketView() string {
	return `Market Data

Index           Last        Change      % Change
-----           ----        ------      --------
S&P 500         --          --          --
NASDAQ          --          --          --
DOW             --          --          --
BTC/USD         --          --          --

[Market data not connected. Press 'r' to refresh]`
}

func (m Model) renderWalletView() string {
	return `Bitcoin Wallet (Cold Storage)

Status: Not Initialized

To initialize your wallet:
1. Run 'gazillioner wallet init' from command line
2. Securely store your 24-word recovery phrase
3. Set up your PIN

Security Notice:
- Private keys never leave this device
- All signing happens locally
- Air-gapped signing available in V1.1`
}

func (m Model) renderSettingsView() string {
	return `Settings

General
-------
  Timezone:      UTC
  Currency:      USD
  Theme:         Dark

AI Model
--------
  Model:         Llama 3 13B (3.5-bit)
  Max Tokens:    2048
  Temperature:   0.7

Data
----
  Refresh Rate:  5 minutes
  Data Source:   Yahoo Finance (15-min delay)

Press 'Enter' on a setting to modify`
}

func (m Model) renderHelpView() string {
	return `Help - Keyboard Shortcuts

Navigation
----------
  1-6, 0        Switch views
  Tab           Next field/item
  Shift+Tab     Previous field/item
  ↑/k, ↓/j      Move up/down
  Enter         Select/Confirm
  Esc           Back/Cancel

Actions
-------
  n             New item
  d/Delete      Delete item
  r             Refresh data
  /             Search
  ?             Show this help

Chat
----
  Type your question and press Enter
  Ctrl+C        Cancel current generation

Press Esc to return`
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Run starts the TUI application
func Run() error {
	p := tea.NewProgram(New(), tea.WithAltScreen())
	_, err := p.Run()
	return err
}
