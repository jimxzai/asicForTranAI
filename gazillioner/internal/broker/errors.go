package broker

import (
	"errors"
	"fmt"
)

// Common broker errors
var (
	ErrNotConnected       = errors.New("broker not connected")
	ErrAlreadyConnected   = errors.New("broker already connected")
	ErrInvalidCredentials = errors.New("invalid credentials")
	ErrTokenExpired       = errors.New("OAuth token expired")
	ErrTokenRefreshFailed = errors.New("failed to refresh OAuth token")
	ErrRateLimited        = errors.New("rate limited by broker API")
	ErrAccountNotFound    = errors.New("account not found")
	ErrPositionNotFound   = errors.New("position not found")
	ErrOrderNotFound      = errors.New("order not found")
	ErrOrderRejected      = errors.New("order rejected")
	ErrInsufficientFunds  = errors.New("insufficient funds")
	ErrMarketClosed       = errors.New("market is closed")
	ErrUnsupportedAsset   = errors.New("unsupported asset class")
	ErrOAuthTimeout       = errors.New("OAuth authorization timeout")
	ErrOAuthCanceled      = errors.New("OAuth authorization canceled")
)

// BrokerError wraps broker-specific errors with context
type BrokerError struct {
	Broker  BrokerType
	Op      string // Operation that failed
	Err     error
	Code    string // Broker-specific error code
	Message string // Human-readable message
}

func (e *BrokerError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s: %s failed: [%s] %s", e.Broker, e.Op, e.Code, e.Message)
	}
	if e.Message != "" {
		return fmt.Sprintf("%s: %s failed: %s", e.Broker, e.Op, e.Message)
	}
	return fmt.Sprintf("%s: %s failed: %v", e.Broker, e.Op, e.Err)
}

func (e *BrokerError) Unwrap() error {
	return e.Err
}

// NewBrokerError creates a new broker error
func NewBrokerError(broker BrokerType, op string, err error) *BrokerError {
	return &BrokerError{
		Broker: broker,
		Op:     op,
		Err:    err,
	}
}

// WithCode adds an error code to the broker error
func (e *BrokerError) WithCode(code string) *BrokerError {
	e.Code = code
	return e
}

// WithMessage adds a human-readable message
func (e *BrokerError) WithMessage(msg string) *BrokerError {
	e.Message = msg
	return e
}

// APIError represents an error response from a broker API
type APIError struct {
	StatusCode int
	Code       string
	Message    string
	Details    map[string]interface{}
}

func (e *APIError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("API error %d [%s]: %s", e.StatusCode, e.Code, e.Message)
	}
	return fmt.Sprintf("API error %d: %s", e.StatusCode, e.Message)
}

// IsRetryable returns true if the error is transient and can be retried
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}

	// Rate limiting is retryable
	if errors.Is(err, ErrRateLimited) {
		return true
	}

	// Check for API errors
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		// 5xx errors are typically retryable
		if apiErr.StatusCode >= 500 && apiErr.StatusCode < 600 {
			return true
		}
		// 429 Too Many Requests
		if apiErr.StatusCode == 429 {
			return true
		}
	}

	return false
}

// IsAuthError returns true if the error is authentication-related
func IsAuthError(err error) bool {
	if err == nil {
		return false
	}

	if errors.Is(err, ErrInvalidCredentials) ||
		errors.Is(err, ErrTokenExpired) ||
		errors.Is(err, ErrTokenRefreshFailed) {
		return true
	}

	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == 401 || apiErr.StatusCode == 403
	}

	return false
}
