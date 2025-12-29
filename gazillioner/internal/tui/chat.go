package tui

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// ChatMessage represents a message in the chat
type ChatMessage struct {
	Role      string    // "user", "assistant", or "system"
	Content   string
	Timestamp time.Time
}

// ChatModel represents the chat view
type ChatModel struct {
	width    int
	height   int
	styles   Styles
	messages []ChatMessage
	viewport viewport.Model
	textarea textarea.Model
	spinner  spinner.Model
	loading  bool
	err      error
}

// NewChatModel creates a new chat model
func NewChatModel() *ChatModel {
	ta := textarea.New()
	ta.Placeholder = "Ask about your portfolio, markets, or financial concepts..."
	ta.Focus()
	ta.CharLimit = 4096
	ta.SetHeight(3)
	ta.ShowLineNumbers = false

	vp := viewport.New(80, 10)
	vp.SetContent("")

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("#7C3AED"))

	return &ChatModel{
		styles:   DefaultStyles(),
		messages: []ChatMessage{},
		viewport: vp,
		textarea: ta,
		spinner:  s,
		loading:  false,
	}
}

// SetSize updates the chat dimensions
func (m *ChatModel) SetSize(width, height int) {
	m.width = width
	m.height = height

	// Viewport takes most of the height, minus textarea and padding
	vpHeight := max(5, height-10)
	m.viewport.Width = width - 4
	m.viewport.Height = vpHeight

	m.textarea.SetWidth(width - 4)
}

// Init implements tea.Model
func (m *ChatModel) Init() tea.Cmd {
	return textarea.Blink
}

// Update implements tea.Model
func (m *ChatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyEnter:
			if !msg.Alt && !m.loading {
				// Send message
				content := strings.TrimSpace(m.textarea.Value())
				if content != "" {
					m.messages = append(m.messages, ChatMessage{
						Role:      "user",
						Content:   content,
						Timestamp: time.Now(),
					})
					m.textarea.Reset()
					m.loading = true
					m.updateViewport()

					// TODO: Send to inference service
					cmds = append(cmds, m.simulateResponse())
				}
			}

		case tea.KeyCtrlC:
			if m.loading {
				m.loading = false
				return m, nil
			}
		}

	case ResponseMsg:
		m.loading = false
		m.messages = append(m.messages, ChatMessage{
			Role:      "assistant",
			Content:   msg.Content,
			Timestamp: time.Now(),
		})
		m.updateViewport()

	case spinner.TickMsg:
		if m.loading {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			cmds = append(cmds, cmd)
		}
	}

	// Update textarea
	if !m.loading {
		var cmd tea.Cmd
		m.textarea, cmd = m.textarea.Update(msg)
		cmds = append(cmds, cmd)
	}

	// Update viewport
	var cmd tea.Cmd
	m.viewport, cmd = m.viewport.Update(msg)
	cmds = append(cmds, cmd)

	return m, tea.Batch(cmds...)
}

// ResponseMsg is sent when AI response is received
type ResponseMsg struct {
	Content string
	Error   error
}

// simulateResponse simulates an AI response (for demo)
func (m *ChatModel) simulateResponse() tea.Cmd {
	return func() tea.Msg {
		time.Sleep(1 * time.Second) // Simulate latency

		return ResponseMsg{
			Content: `I'm your local AI financial assistant. I can help you with:

• Portfolio analysis and risk assessment
• Market research and comparisons
• Financial concepts and education
• Investment strategy discussions

**Note:** This is a demo response. The inference engine is not yet connected.

---
*Disclaimer: This is not financial advice. Always consult a qualified financial advisor before making investment decisions.*`,
		}
	}
}

// View implements tea.Model
func (m *ChatModel) View() string {
	var b strings.Builder

	// Title
	title := lipgloss.NewStyle().
		Bold(true).
		Foreground(lipgloss.Color("#FFFFFF")).
		Render("AI Chat")
	b.WriteString(title)
	b.WriteString("\n")

	// Model info
	modelInfo := m.styles.Muted.Render("Model: Llama 3 13B (3.5-bit) • Local • Verified")
	b.WriteString(modelInfo)
	b.WriteString("\n\n")

	// Chat viewport
	vpStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#374151")).
		Padding(1)

	b.WriteString(vpStyle.Render(m.viewport.View()))
	b.WriteString("\n\n")

	// Loading indicator
	if m.loading {
		loadingStyle := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#7C3AED"))
		b.WriteString(loadingStyle.Render(m.spinner.View() + " Thinking..."))
		b.WriteString("\n")
	}

	// Input area
	inputStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("#7C3AED")).
		Padding(0, 1)

	b.WriteString(inputStyle.Render(m.textarea.View()))
	b.WriteString("\n")

	// Help
	help := m.styles.Muted.Render("Enter to send • Ctrl+C to cancel")
	b.WriteString(help)

	return b.String()
}

func (m *ChatModel) updateViewport() {
	var content strings.Builder

	if len(m.messages) == 0 {
		welcomeMsg := `Welcome to Gazillioner AI Chat!

I'm your local, privacy-first AI financial assistant. All processing
happens on this device - your data never leaves.

Try asking:
• "What's my portfolio exposure to tech?"
• "Compare AAPL vs MSFT"
• "Explain dollar cost averaging"
• "What are the risks in my portfolio?"

Press Enter to send a message.`

		content.WriteString(m.styles.Muted.Render(welcomeMsg))
	} else {
		for _, msg := range m.messages {
			// Style based on role
			var roleStyle, contentStyle lipgloss.Style
			var prefix string

			switch msg.Role {
			case "user":
				roleStyle = lipgloss.NewStyle().
					Bold(true).
					Foreground(lipgloss.Color("#10B981"))
				prefix = "You"
				contentStyle = lipgloss.NewStyle()

			case "assistant":
				roleStyle = lipgloss.NewStyle().
					Bold(true).
					Foreground(lipgloss.Color("#7C3AED"))
				prefix = "AI"
				contentStyle = lipgloss.NewStyle()

			case "system":
				roleStyle = m.styles.Muted
				prefix = "System"
				contentStyle = m.styles.Muted
			}

			// Timestamp
			timeStr := msg.Timestamp.Format("15:04")

			// Render message
			content.WriteString(roleStyle.Render(fmt.Sprintf("%s [%s]:", prefix, timeStr)))
			content.WriteString("\n")
			content.WriteString(contentStyle.Render(msg.Content))
			content.WriteString("\n\n")
		}
	}

	m.viewport.SetContent(content.String())
	m.viewport.GotoBottom()
}
