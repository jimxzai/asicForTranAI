package broker

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"
)

const (
	// OAuthCallbackPort is the default port for OAuth callbacks
	OAuthCallbackPort = 47391
	// OAuthTimeout is the default timeout for OAuth flow
	OAuthTimeout = 5 * time.Minute
)

// OAuthCallbackServer handles OAuth redirects from brokers
type OAuthCallbackServer struct {
	port     int
	server   *http.Server
	listener net.Listener

	mu       sync.Mutex
	codeChan chan string
	errChan  chan error
	state    string
	running  bool
}

// OAuthResult contains the result of an OAuth flow
type OAuthResult struct {
	Code  string
	State string
	Error string
}

// NewOAuthCallbackServer creates a new OAuth callback server
func NewOAuthCallbackServer(port int) *OAuthCallbackServer {
	if port == 0 {
		port = OAuthCallbackPort
	}
	return &OAuthCallbackServer{
		port:     port,
		codeChan: make(chan string, 1),
		errChan:  make(chan error, 1),
	}
}

// GenerateState creates a cryptographically secure state parameter
func GenerateState() (string, error) {
	bytes := make([]byte, 16)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return hex.EncodeToString(bytes), nil
}

// Start begins listening for OAuth callbacks
func (s *OAuthCallbackServer) Start(state string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		return fmt.Errorf("server already running")
	}

	s.state = state

	// Create listener
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", s.port))
	if err != nil {
		return fmt.Errorf("failed to start OAuth server: %w", err)
	}
	s.listener = listener

	// Create HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/callback", s.handleCallback)
	mux.HandleFunc("/", s.handleRoot)

	s.server = &http.Server{
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	// Start serving
	go func() {
		if err := s.server.Serve(listener); err != nil && err != http.ErrServerClosed {
			select {
			case s.errChan <- err:
			default:
			}
		}
	}()

	s.running = true
	return nil
}

// handleCallback processes the OAuth redirect
func (s *OAuthCallbackServer) handleCallback(w http.ResponseWriter, r *http.Request) {
	// Check for errors from the OAuth provider
	if errParam := r.URL.Query().Get("error"); errParam != "" {
		errDesc := r.URL.Query().Get("error_description")
		s.sendError(fmt.Errorf("OAuth error: %s - %s", errParam, errDesc))
		s.renderErrorPage(w, errParam, errDesc)
		return
	}

	// Get authorization code
	code := r.URL.Query().Get("code")
	if code == "" {
		s.sendError(fmt.Errorf("no authorization code received"))
		s.renderErrorPage(w, "missing_code", "No authorization code was received")
		return
	}

	// Verify state parameter to prevent CSRF
	state := r.URL.Query().Get("state")
	if state != s.state {
		s.sendError(fmt.Errorf("state mismatch: expected %s, got %s", s.state, state))
		s.renderErrorPage(w, "state_mismatch", "Security validation failed")
		return
	}

	// Send code to channel
	select {
	case s.codeChan <- code:
	default:
		// Channel full, ignore duplicate callbacks
	}

	// Render success page
	s.renderSuccessPage(w)
}

// handleRoot serves a simple status page
func (s *OAuthCallbackServer) handleRoot(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	fmt.Fprint(w, `<!DOCTYPE html>
<html>
<head>
    <title>Gazillioner - OAuth</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               display: flex; justify-content: center; align-items: center;
               height: 100vh; margin: 0; background: #1F2937; color: #F9FAFB; }
        .container { text-align: center; padding: 2rem; }
        h1 { color: #7C3AED; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gazillioner</h1>
        <p>Waiting for broker authorization...</p>
        <p>Please complete the login in your browser.</p>
    </div>
</body>
</html>`)
}

// renderSuccessPage shows a success message and closes the window
func (s *OAuthCallbackServer) renderSuccessPage(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	fmt.Fprint(w, `<!DOCTYPE html>
<html>
<head>
    <title>Authorization Successful</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               display: flex; justify-content: center; align-items: center;
               height: 100vh; margin: 0; background: #1F2937; color: #F9FAFB; }
        .container { text-align: center; padding: 2rem; }
        h1 { color: #10B981; }
        .success-icon { font-size: 4rem; margin-bottom: 1rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="success-icon">✓</div>
        <h1>Authorization Successful!</h1>
        <p>You can close this window and return to Gazillioner.</p>
        <script>
            // Try to close the window after a short delay
            setTimeout(function() { window.close(); }, 2000);
        </script>
    </div>
</body>
</html>`)
}

// renderErrorPage shows an error message
func (s *OAuthCallbackServer) renderErrorPage(w http.ResponseWriter, errCode, errDesc string) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusBadRequest)
	fmt.Fprintf(w, `<!DOCTYPE html>
<html>
<head>
    <title>Authorization Failed</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               display: flex; justify-content: center; align-items: center;
               height: 100vh; margin: 0; background: #1F2937; color: #F9FAFB; }
        .container { text-align: center; padding: 2rem; }
        h1 { color: #EF4444; }
        .error-icon { font-size: 4rem; margin-bottom: 1rem; }
        .error-code { color: #6B7280; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <div class="error-icon">✗</div>
        <h1>Authorization Failed</h1>
        <p>%s</p>
        <p class="error-code">Error: %s</p>
        <p>Please close this window and try again in Gazillioner.</p>
    </div>
</body>
</html>`, errDesc, errCode)
}

// sendError sends an error to the error channel
func (s *OAuthCallbackServer) sendError(err error) {
	select {
	case s.errChan <- err:
	default:
	}
}

// WaitForCode waits for the OAuth code with timeout
func (s *OAuthCallbackServer) WaitForCode(timeout time.Duration) (string, error) {
	select {
	case code := <-s.codeChan:
		return code, nil
	case err := <-s.errChan:
		return "", err
	case <-time.After(timeout):
		return "", ErrOAuthTimeout
	}
}

// WaitForCodeWithContext waits for the OAuth code with context
func (s *OAuthCallbackServer) WaitForCodeWithContext(ctx context.Context) (string, error) {
	select {
	case code := <-s.codeChan:
		return code, nil
	case err := <-s.errChan:
		return "", err
	case <-ctx.Done():
		if ctx.Err() == context.Canceled {
			return "", ErrOAuthCanceled
		}
		return "", ErrOAuthTimeout
	}
}

// Stop shuts down the OAuth server
func (s *OAuthCallbackServer) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := s.server.Shutdown(ctx); err != nil {
		// Force close if graceful shutdown fails
		s.listener.Close()
	}

	s.running = false
	return nil
}

// GetCallbackURL returns the OAuth callback URL
func (s *OAuthCallbackServer) GetCallbackURL() string {
	return fmt.Sprintf("http://127.0.0.1:%d/callback", s.port)
}

// IsRunning returns true if the server is running
func (s *OAuthCallbackServer) IsRunning() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.running
}
