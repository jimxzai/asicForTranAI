package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// DashboardModel represents the dashboard view
type DashboardModel struct {
	width  int
	height int
	styles Styles

	// Portfolio data (mock for now)
	totalValue       float64
	dayChange        float64
	dayChangePercent float64
	holdingsCount    int
}

// NewDashboardModel creates a new dashboard model
func NewDashboardModel() *DashboardModel {
	return &DashboardModel{
		styles:           DefaultStyles(),
		totalValue:       0,
		dayChange:        0,
		dayChangePercent: 0,
		holdingsCount:    0,
	}
}

// SetSize updates the dashboard dimensions
func (m *DashboardModel) SetSize(width, height int) {
	m.width = width
	m.height = height
}

// Init implements tea.Model
func (m *DashboardModel) Init() tea.Cmd {
	return nil
}

// Update implements tea.Model
func (m *DashboardModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "r":
			// Refresh data
			return m, nil
		}
	}
	return m, nil
}

// View implements tea.Model
func (m *DashboardModel) View() string {
	var b strings.Builder

	// Title
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#FFFFFF")).
		MarginBottom(1).
		Render("Dashboard")
	b.WriteString(title)
	b.WriteString("\n\n")

	// Portfolio Summary Card
	b.WriteString(m.renderPortfolioSummary())
	b.WriteString("\n\n")

	// Quick Stats Row
	b.WriteString(m.renderQuickStats())
	b.WriteString("\n\n")

	// Recent Activity
	b.WriteString(m.renderRecentActivity())
	b.WriteString("\n\n")

	// AI Insights
	b.WriteString(m.renderAIInsights())

	return b.String()
}

func (m *DashboardModel) renderPortfolioSummary() string {
	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#7C3AED")).
		Padding(1, 2).
		Width(min(60, m.width-4))

	// Format values
	valueStr := "$0.00"
	changeStr := "$0.00 (0.00%)"
	changeStyle := m.styles.Muted

	if m.totalValue > 0 {
		valueStr = fmt.Sprintf("$%.2f", m.totalValue)
	}

	if m.dayChange != 0 {
		sign := "+"
		changeStyle = m.styles.Success
		if m.dayChange < 0 {
			sign = ""
			changeStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#EF4444"))
		}
		changeStr = fmt.Sprintf("%s$%.2f (%s%.2f%%)", sign, m.dayChange, sign, m.dayChangePercent)
	}

	content := fmt.Sprintf(`Portfolio Value

%s
%s

Holdings: %d`,
		lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#FFFFFF")).Render(valueStr),
		changeStyle.Render(changeStr),
		m.holdingsCount,
	)

	return cardStyle.Render(content)
}

func (m *DashboardModel) renderQuickStats() string {
	statStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#374151")).
		Padding(0, 2).
		Width(18)

	stats := []struct {
		label string
		value string
	}{
		{"Watchlist", "0 items"},
		{"AI Queries", "0 today"},
		{"BTC Balance", "0.00"},
	}

	var statCards []string
	for _, stat := range stats {
		card := statStyle.Render(fmt.Sprintf("%s\n%s",
			m.styles.Muted.Render(stat.label),
			lipgloss.NewStyle().Bold(true).Render(stat.value),
		))
		statCards = append(statCards, card)
	}

	return lipgloss.JoinHorizontal(lipgloss.Top, statCards...)
}

func (m *DashboardModel) renderRecentActivity() string {
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#FFFFFF")).
		Render("Recent Activity")

	content := m.styles.Muted.Render(`
  No recent activity

  Get started:
  • Press '2' to manage your portfolio
  • Press '4' to chat with the AI
  • Press 'n' to add your first holding`)

	return title + content
}

func (m *DashboardModel) renderAIInsights() string {
	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#7C3AED")).
		Padding(1, 2).
		Width(min(60, m.width-4))

	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#7C3AED")).
		Render("AI Insights")

	content := m.styles.Muted.Render(`
No insights yet. Add holdings to your portfolio
and the AI will provide personalized analysis.

Press '4' to start a conversation with the AI.`)

	return cardStyle.Render(title + content)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
