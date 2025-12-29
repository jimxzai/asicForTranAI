package alpaca

import (
	"fmt"
	"net/url"
)

// OAuth configuration for Alpaca
const (
	// Production OAuth endpoints
	OAuthAuthorizeURL = "https://app.alpaca.markets/oauth/authorize"
	OAuthTokenURL     = "https://api.alpaca.markets/oauth/token"

	// Paper trading uses same OAuth
	OAuthRedirectURI  = "http://127.0.0.1:47391/callback"

	// Scopes
	ScopeAccount  = "account:write"
	ScopeTrading  = "trading"
	ScopeData     = "data"
)

// OAuthConfig holds OAuth configuration
type OAuthConfig struct {
	ClientID     string
	ClientSecret string
	RedirectURI  string
	Scopes       []string
}

// DefaultOAuthConfig returns the default OAuth configuration
func DefaultOAuthConfig(clientID, clientSecret string) OAuthConfig {
	return OAuthConfig{
		ClientID:     clientID,
		ClientSecret: clientSecret,
		RedirectURI:  OAuthRedirectURI,
		Scopes:       []string{ScopeAccount, ScopeTrading, ScopeData},
	}
}

// BuildAuthURL constructs the OAuth authorization URL
func BuildAuthURL(cfg OAuthConfig, state string) string {
	params := url.Values{}
	params.Set("response_type", "code")
	params.Set("client_id", cfg.ClientID)
	params.Set("redirect_uri", cfg.RedirectURI)
	params.Set("state", state)

	// Join scopes with space
	scope := ""
	for i, s := range cfg.Scopes {
		if i > 0 {
			scope += " "
		}
		scope += s
	}
	params.Set("scope", scope)

	return fmt.Sprintf("%s?%s", OAuthAuthorizeURL, params.Encode())
}

// TokenRequest represents the OAuth token exchange request
type TokenRequest struct {
	GrantType    string `json:"grant_type"`
	Code         string `json:"code"`
	ClientID     string `json:"client_id"`
	ClientSecret string `json:"client_secret"`
	RedirectURI  string `json:"redirect_uri"`
}

// NewTokenRequest creates a new token exchange request
func NewTokenRequest(cfg OAuthConfig, code string) TokenRequest {
	return TokenRequest{
		GrantType:    "authorization_code",
		Code:         code,
		ClientID:     cfg.ClientID,
		ClientSecret: cfg.ClientSecret,
		RedirectURI:  cfg.RedirectURI,
	}
}

// RefreshTokenRequest represents the OAuth token refresh request
type RefreshTokenRequest struct {
	GrantType    string `json:"grant_type"`
	RefreshToken string `json:"refresh_token"`
	ClientID     string `json:"client_id"`
	ClientSecret string `json:"client_secret"`
}

// NewRefreshTokenRequest creates a new token refresh request
func NewRefreshTokenRequest(cfg OAuthConfig, refreshToken string) RefreshTokenRequest {
	return RefreshTokenRequest{
		GrantType:    "refresh_token",
		RefreshToken: refreshToken,
		ClientID:     cfg.ClientID,
		ClientSecret: cfg.ClientSecret,
	}
}
