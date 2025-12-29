package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/table"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// PortfolioModel represents the portfolio view
type PortfolioModel struct {
	width    int
	height   int
	styles   Styles
	table    table.Model
	holdings []HoldingRow
	cursor   int
}

// HoldingRow represents a row in the portfolio table
type HoldingRow struct {
	ID              string
	Ticker          string
	Quantity        float64
	CostBasis       float64
	CurrentPrice    float64
	CurrentValue    float64
	GainLoss        float64
	GainLossPercent float64
	DayChange       float64
	DayChangePercent float64
}

// NewPortfolioModel creates a new portfolio model
func NewPortfolioModel() *PortfolioModel {
	columns := []table.Column{
		{Title: "Ticker", Width: 8},
		{Title: "Qty", Width: 10},
		{Title: "Cost", Width: 10},
		{Title: "Price", Width: 10},
		{Title: "Value", Width: 12},
		{Title: "Gain/Loss", Width: 12},
		{Title: "Day Chg", Width: 10},
	}

	t := table.New(
		table.WithColumns(columns),
		table.WithFocused(true),
		table.WithHeight(10),
	)

	s := table.DefaultStyles()
	s.Header = s.Header.
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color("#7C3AED")).
		BorderBottom(true).
		Bold(true)
	s.Selected = s.Selected.
		Foreground(lipgloss.Color("#FFFFFF")).
		Background(lipgloss.Color("#7C3AED")).
		Bold(false)
	t.SetStyles(s)

	return &PortfolioModel{
		styles:   DefaultStyles(),
		table:    t,
		holdings: []HoldingRow{},
		cursor:   0,
	}
}

// SetSize updates the portfolio dimensions
func (m *PortfolioModel) SetSize(width, height int) {
	m.width = width
	m.height = height
	m.table.SetHeight(max(5, height-10))
}

// Init implements tea.Model
func (m *PortfolioModel) Init() tea.Cmd {
	return nil
}

// Update implements tea.Model
func (m *PortfolioModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "n":
			// TODO: Open add holding modal
			return m, nil
		case "d", "delete":
			// TODO: Delete selected holding
			return m, nil
		case "e":
			// TODO: Edit selected holding
			return m, nil
		case "r":
			// TODO: Refresh prices
			return m, nil
		}
	}

	m.table, cmd = m.table.Update(msg)
	return m, cmd
}

// View implements tea.Model
func (m *PortfolioModel) View() string {
	var b strings.Builder

	// Title and summary
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#FFFFFF")).
		Render("Portfolio")
	b.WriteString(title)
	b.WriteString("\n\n")

	// Summary stats
	b.WriteString(m.renderSummary())
	b.WriteString("\n\n")

	// Holdings table
	if len(m.holdings) == 0 {
		emptyMsg := m.styles.Muted.Render(`
No holdings yet.

Press 'n' to add your first holding, or import from CSV:
  gazillioner portfolio import portfolio.csv

Supported formats:
  • Manual entry (ticker, quantity, cost basis)
  • CSV import with automatic column mapping
`)
		b.WriteString(emptyMsg)
	} else {
		b.WriteString(m.table.View())
	}

	b.WriteString("\n\n")
	b.WriteString(m.renderHelp())

	return b.String()
}

func (m *PortfolioModel) renderSummary() string {
	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#374151")).
		Padding(0, 2)

	// Calculate totals
	var totalValue, totalCost, dayChange float64
	for _, h := range m.holdings {
		totalValue += h.CurrentValue
		totalCost += h.Quantity * h.CostBasis
		dayChange += h.DayChange * h.Quantity
	}

	gainLoss := totalValue - totalCost
	gainLossPercent := 0.0
	if totalCost > 0 {
		gainLossPercent = (gainLoss / totalCost) * 100
	}
	dayChangePercent := 0.0
	if totalValue > 0 {
		dayChangePercent = (dayChange / (totalValue - dayChange)) * 100
	}

	// Format values
	valueStr := fmt.Sprintf("$%.2f", totalValue)
	if totalValue == 0 {
		valueStr = "$0.00"
	}

	gainStyle := m.styles.Success
	gainStr := fmt.Sprintf("+$%.2f (+%.2f%%)", gainLoss, gainLossPercent)
	if gainLoss < 0 {
		gainStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#EF4444"))
		gainStr = fmt.Sprintf("-$%.2f (%.2f%%)", -gainLoss, gainLossPercent)
	}
	if gainLoss == 0 {
		gainStyle = m.styles.Muted
		gainStr = "$0.00 (0.00%)"
	}

	dayStyle := m.styles.Success
	dayStr := fmt.Sprintf("+$%.2f (+%.2f%%)", dayChange, dayChangePercent)
	if dayChange < 0 {
		dayStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#EF4444"))
		dayStr = fmt.Sprintf("-$%.2f (%.2f%%)", -dayChange, dayChangePercent)
	}
	if dayChange == 0 {
		dayStyle = m.styles.Muted
		dayStr = "$0.00 (0.00%)"
	}

	stats := []string{
		cardStyle.Render(fmt.Sprintf("%s\n%s",
			m.styles.Muted.Render("Total Value"),
			lipgloss.NewStyle().Bold(true).Render(valueStr),
		)),
		cardStyle.Render(fmt.Sprintf("%s\n%s",
			m.styles.Muted.Render("Total Gain/Loss"),
			gainStyle.Render(gainStr),
		)),
		cardStyle.Render(fmt.Sprintf("%s\n%s",
			m.styles.Muted.Render("Day Change"),
			dayStyle.Render(dayStr),
		)),
		cardStyle.Render(fmt.Sprintf("%s\n%s",
			m.styles.Muted.Render("Holdings"),
			lipgloss.NewStyle().Bold(true).Render(fmt.Sprintf("%d", len(m.holdings))),
		)),
	}

	return lipgloss.JoinHorizontal(lipgloss.Top, stats...)
}

func (m *PortfolioModel) renderHelp() string {
	help := m.styles.Muted.Render("[n] Add  [e] Edit  [d] Delete  [r] Refresh  [/] Search")
	return help
}

// RefreshHoldings updates the holdings data
func (m *PortfolioModel) RefreshHoldings(holdings []HoldingRow) {
	m.holdings = holdings

	// Update table rows
	rows := make([]table.Row, len(holdings))
	for i, h := range holdings {
		gainStr := fmt.Sprintf("+$%.2f", h.GainLoss)
		if h.GainLoss < 0 {
			gainStr = fmt.Sprintf("-$%.2f", -h.GainLoss)
		}

		dayStr := fmt.Sprintf("+%.2f%%", h.DayChangePercent)
		if h.DayChangePercent < 0 {
			dayStr = fmt.Sprintf("%.2f%%", h.DayChangePercent)
		}

		rows[i] = table.Row{
			h.Ticker,
			fmt.Sprintf("%.4f", h.Quantity),
			fmt.Sprintf("$%.2f", h.CostBasis),
			fmt.Sprintf("$%.2f", h.CurrentPrice),
			fmt.Sprintf("$%.2f", h.CurrentValue),
			gainStr,
			dayStr,
		}
	}

	m.table.SetRows(rows)
}
