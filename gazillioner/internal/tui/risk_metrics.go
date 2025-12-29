package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// RiskMetricsModel represents the risk metrics dashboard view
type RiskMetricsModel struct {
	width  int
	height int
	styles Styles

	// FQ Score data
	fqScore      int
	fqCategory   string
	fqPercentile int

	// Dimensional scores (0-100 each)
	dimensions map[string]int

	// Analysis results
	strengths    []string
	improvements []string

	// Portfolio risk metrics
	portfolioVolatility float64
	sharpeRatio         float64
	maxDrawdown         float64
	betaToMarket        float64
	valueAtRisk         float64
}

// DimensionInfo holds display info for each dimension
type DimensionInfo struct {
	Name   string
	Weight float64
}

// Dimension definitions matching the Python backend
var dimensionDefinitions = []DimensionInfo{
	{Name: "preparedness", Weight: 1.0},
	{Name: "savings", Weight: 1.1},
	{Name: "awareness", Weight: 0.9},
	{Name: "debt_management", Weight: 1.2},
	{Name: "planning", Weight: 1.0},
	{Name: "emotional_control", Weight: 1.3},
	{Name: "retirement", Weight: 1.1},
	{Name: "tax_efficiency", Weight: 0.8},
	{Name: "goal_setting", Weight: 0.9},
	{Name: "discipline", Weight: 1.2},
}

// NewRiskMetricsModel creates a new risk metrics model
func NewRiskMetricsModel() *RiskMetricsModel {
	return &RiskMetricsModel{
		styles:       DefaultStyles(),
		fqScore:      0,
		fqCategory:   "Not Assessed",
		fqPercentile: 0,
		dimensions:   make(map[string]int),
		strengths:    []string{},
		improvements: []string{},
	}
}

// SetSize updates the model dimensions
func (m *RiskMetricsModel) SetSize(width, height int) {
	m.width = width
	m.height = height
}

// Init implements tea.Model
func (m *RiskMetricsModel) Init() tea.Cmd {
	return nil
}

// Update implements tea.Model
func (m *RiskMetricsModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "r":
			// TODO: Refresh risk metrics from API
			return m, nil
		case "a":
			// TODO: Take FQ assessment
			return m, nil
		}
	}
	return m, nil
}

// View implements tea.Model
func (m *RiskMetricsModel) View() string {
	var b strings.Builder

	// Title
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#FFFFFF")).
		MarginBottom(1).
		Render("Risk Metrics Dashboard")
	b.WriteString(title)
	b.WriteString("\n\n")

	// FQ Score Card
	b.WriteString(m.renderFQScoreCard())
	b.WriteString("\n\n")

	// Dimensional Scores
	b.WriteString(m.renderDimensionalScores())
	b.WriteString("\n\n")

	// Portfolio Risk Metrics
	b.WriteString(m.renderPortfolioRisk())
	b.WriteString("\n\n")

	// Strengths & Improvements
	b.WriteString(m.renderAnalysis())
	b.WriteString("\n\n")

	// Help
	b.WriteString(m.renderHelp())

	return b.String()
}

func (m *RiskMetricsModel) renderFQScoreCard() string {
	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#7C3AED")).
		Padding(1, 2).
		Width(min(60, m.width-4))

	// Score color based on category
	scoreColor := "#6B7280" // Gray for not assessed
	switch m.fqCategory {
	case "Master":
		scoreColor = "#10B981" // Green
	case "Strong":
		scoreColor = "#3B82F6" // Blue
	case "Developing":
		scoreColor = "#F59E0B" // Amber
	case "Beginner":
		scoreColor = "#EF4444" // Red
	}

	scoreStyle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color(scoreColor))

	// Progress bar for score (0-1000)
	progressWidth := 40
	filled := (m.fqScore * progressWidth) / 1000
	if filled > progressWidth {
		filled = progressWidth
	}
	progressBar := lipgloss.NewStyle().Foreground(lipgloss.Color(scoreColor)).
		Render(strings.Repeat("█", filled)) +
		lipgloss.NewStyle().Foreground(lipgloss.Color("#374151")).
			Render(strings.Repeat("░", progressWidth-filled))

	content := fmt.Sprintf(`Financial IQ Score

%s  %s

%s

Percentile: Top %d%%`,
		scoreStyle.Render(fmt.Sprintf("%d", m.fqScore)),
		m.styles.Muted.Render(fmt.Sprintf("/ 1000 (%s)", m.fqCategory)),
		progressBar,
		100-m.fqPercentile,
	)

	return cardStyle.Render(content)
}

func (m *RiskMetricsModel) renderDimensionalScores() string {
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#FFFFFF")).
		Render("Dimensional Scores")

	if len(m.dimensions) == 0 {
		return title + "\n" + m.styles.Muted.Render("  Take the FQ assessment to see your dimensional scores (press 'a')")
	}

	var rows strings.Builder
	rows.WriteString("\n")

	barWidth := 20
	for _, dim := range dimensionDefinitions {
		score := m.dimensions[dim.Name]
		filled := (score * barWidth) / 100

		// Color based on score
		barColor := "#EF4444" // Red
		if score >= 80 {
			barColor = "#10B981" // Green
		} else if score >= 60 {
			barColor = "#3B82F6" // Blue
		} else if score >= 40 {
			barColor = "#F59E0B" // Amber
		}

		bar := lipgloss.NewStyle().Foreground(lipgloss.Color(barColor)).
			Render(strings.Repeat("█", filled)) +
			lipgloss.NewStyle().Foreground(lipgloss.Color("#374151")).
				Render(strings.Repeat("░", barWidth-filled))

		// Format dimension name nicely
		displayName := strings.ReplaceAll(dim.Name, "_", " ")
		displayName = strings.Title(displayName)

		weightIndicator := ""
		if dim.Weight >= 1.2 {
			weightIndicator = " ★"
		}

		rows.WriteString(fmt.Sprintf("  %-18s %s %3d%%%s\n",
			displayName,
			bar,
			score,
			m.styles.Muted.Render(weightIndicator),
		))
	}

	return title + rows.String()
}

func (m *RiskMetricsModel) renderPortfolioRisk() string {
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#FFFFFF")).
		Render("Portfolio Risk Metrics")

	cardStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#374151")).
		Padding(0, 2)

	// Format risk metrics
	volatilityStr := fmt.Sprintf("%.2f%%", m.portfolioVolatility*100)
	sharpeStr := fmt.Sprintf("%.2f", m.sharpeRatio)
	drawdownStr := fmt.Sprintf("%.2f%%", m.maxDrawdown*100)
	betaStr := fmt.Sprintf("%.2f", m.betaToMarket)
	varStr := fmt.Sprintf("$%.2f", m.valueAtRisk)

	// Color code based on risk levels
	volatilityStyle := m.getRiskStyle(m.portfolioVolatility, 0.15, 0.25)
	sharpeStyle := m.getRewardStyle(m.sharpeRatio, 1.0, 2.0)
	drawdownStyle := m.getRiskStyle(m.maxDrawdown, 0.10, 0.20)
	betaStyle := m.styles.Muted
	if m.betaToMarket > 1.2 {
		betaStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#EF4444"))
	} else if m.betaToMarket < 0.8 {
		betaStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#10B981"))
	}

	stats := []string{
		cardStyle.Render(fmt.Sprintf("%s\n%s",
			m.styles.Muted.Render("Volatility"),
			volatilityStyle.Render(volatilityStr),
		)),
		cardStyle.Render(fmt.Sprintf("%s\n%s",
			m.styles.Muted.Render("Sharpe Ratio"),
			sharpeStyle.Render(sharpeStr),
		)),
		cardStyle.Render(fmt.Sprintf("%s\n%s",
			m.styles.Muted.Render("Max Drawdown"),
			drawdownStyle.Render(drawdownStr),
		)),
		cardStyle.Render(fmt.Sprintf("%s\n%s",
			m.styles.Muted.Render("Beta"),
			betaStyle.Render(betaStr),
		)),
		cardStyle.Render(fmt.Sprintf("%s\n%s",
			m.styles.Muted.Render("VaR (95%)"),
			m.styles.Warning.Render(varStr),
		)),
	}

	return title + "\n\n" + lipgloss.JoinHorizontal(lipgloss.Top, stats...)
}

func (m *RiskMetricsModel) getRiskStyle(value, low, high float64) lipgloss.Style {
	if value <= low {
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#10B981")) // Green - low risk
	} else if value <= high {
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#F59E0B")) // Amber - medium risk
	}
	return lipgloss.NewStyle().Foreground(lipgloss.Color("#EF4444")) // Red - high risk
}

func (m *RiskMetricsModel) getRewardStyle(value, low, high float64) lipgloss.Style {
	if value >= high {
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#10B981")) // Green - good
	} else if value >= low {
		return lipgloss.NewStyle().Foreground(lipgloss.Color("#F59E0B")) // Amber - okay
	}
	return lipgloss.NewStyle().Foreground(lipgloss.Color("#EF4444")) // Red - poor
}

func (m *RiskMetricsModel) renderAnalysis() string {
	strengthsCard := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#10B981")).
		Padding(1, 2).
		Width(min(35, (m.width-8)/2))

	improvementsCard := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#F59E0B")).
		Padding(1, 2).
		Width(min(35, (m.width-8)/2))

	// Strengths content
	strengthsTitle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#10B981")).
		Render("Strengths")

	strengthsList := ""
	if len(m.strengths) == 0 {
		strengthsList = m.styles.Muted.Render("\nComplete assessment\nto see strengths")
	} else {
		for _, s := range m.strengths {
			strengthsList += fmt.Sprintf("\n• %s", s)
		}
	}

	// Improvements content
	improvementsTitle := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#F59E0B")).
		Render("Areas to Improve")

	improvementsList := ""
	if len(m.improvements) == 0 {
		improvementsList = m.styles.Muted.Render("\nComplete assessment\nto see areas to improve")
	} else {
		for _, s := range m.improvements {
			improvementsList += fmt.Sprintf("\n• %s", s)
		}
	}

	left := strengthsCard.Render(strengthsTitle + strengthsList)
	right := improvementsCard.Render(improvementsTitle + improvementsList)

	return lipgloss.JoinHorizontal(lipgloss.Top, left, "  ", right)
}

func (m *RiskMetricsModel) renderHelp() string {
	return m.styles.Muted.Render("[a] Take Assessment  [r] Refresh  [esc] Back to Dashboard")
}

// UpdateFQScore updates the FQ score data
func (m *RiskMetricsModel) UpdateFQScore(score int, category string, percentile int) {
	m.fqScore = score
	m.fqCategory = category
	m.fqPercentile = percentile
}

// UpdateDimensions updates the dimensional scores
func (m *RiskMetricsModel) UpdateDimensions(dimensions map[string]int) {
	m.dimensions = dimensions
}

// UpdateAnalysis updates the strengths and improvements
func (m *RiskMetricsModel) UpdateAnalysis(strengths, improvements []string) {
	m.strengths = strengths
	m.improvements = improvements
}

// UpdatePortfolioRisk updates the portfolio risk metrics
func (m *RiskMetricsModel) UpdatePortfolioRisk(volatility, sharpe, drawdown, beta, var95 float64) {
	m.portfolioVolatility = volatility
	m.sharpeRatio = sharpe
	m.maxDrawdown = drawdown
	m.betaToMarket = beta
	m.valueAtRisk = var95
}

// SetMockData sets mock data for testing/demo
func (m *RiskMetricsModel) SetMockData() {
	m.fqScore = 687
	m.fqCategory = "Strong"
	m.fqPercentile = 72

	m.dimensions = map[string]int{
		"preparedness":      75,
		"savings":           82,
		"awareness":         68,
		"debt_management":   90,
		"planning":          65,
		"emotional_control": 55,
		"retirement":        78,
		"tax_efficiency":    60,
		"goal_setting":      72,
		"discipline":        85,
	}

	m.strengths = []string{"Debt Management", "Discipline", "Savings"}
	m.improvements = []string{"Emotional Control", "Tax Efficiency", "Planning"}

	m.portfolioVolatility = 0.18
	m.sharpeRatio = 1.45
	m.maxDrawdown = 0.12
	m.betaToMarket = 1.05
	m.valueAtRisk = 2450.00
}
