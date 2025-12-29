// Package ffi provides Go bindings to the Rust database layer via CGO
package ffi

/*
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -lgazillioner_db -ldl -lm -lpthread
#cgo darwin LDFLAGS: -framework Security -framework CoreFoundation

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

// Result codes from Rust
typedef enum {
    RESULT_OK = 0,
    RESULT_ERROR = 1,
    RESULT_NOT_INITIALIZED = 2,
    RESULT_INVALID_INPUT = 3,
    RESULT_NOT_FOUND = 4,
    RESULT_AUTH_FAILED = 5,
} ResultCode;

// String result from Rust
typedef struct {
    ResultCode code;
    char* data;
    char* error;
} StringResult;

// Function declarations (implemented in Rust)
extern ResultCode gazillioner_db_init(
    const char* db_path,
    const uint8_t* device_key,
    size_t device_key_len,
    const uint8_t* salt,
    size_t salt_len,
    const char* pin
);
extern ResultCode gazillioner_db_close();
extern bool gazillioner_db_is_initialized();

extern StringResult gazillioner_holding_create(
    const char* ticker,
    double quantity,
    double cost_basis,
    const char* notes
);
extern StringResult gazillioner_holding_list();
extern StringResult gazillioner_holding_get(const char* id);
extern ResultCode gazillioner_holding_delete(const char* id);

extern StringResult gazillioner_watchlist_add(const char* ticker, const char* notes);
extern StringResult gazillioner_watchlist_list();
extern ResultCode gazillioner_watchlist_remove(const char* ticker);

extern bool gazillioner_validate_pin(const char* pin);
extern void gazillioner_free_string(char* ptr);
extern void gazillioner_free_result(StringResult result);
extern const char* gazillioner_version();
*/
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"
	"unsafe"
)

// Common errors
var (
	ErrNotInitialized = errors.New("database not initialized")
	ErrInvalidInput   = errors.New("invalid input")
	ErrNotFound       = errors.New("not found")
	ErrAuthFailed     = errors.New("authentication failed")
)

// Holding represents a portfolio position
type Holding struct {
	ID              string    `json:"id"`
	Ticker          string    `json:"ticker"`
	Name            *string   `json:"name,omitempty"`
	Quantity        float64   `json:"quantity"`
	CostBasis       float64   `json:"cost_basis"`
	AcquisitionDate time.Time `json:"acquisition_date"`
	Notes           *string   `json:"notes,omitempty"`
	AssetClass      string    `json:"asset_class"`
	Sector          *string   `json:"sector,omitempty"`
	Exchange        *string   `json:"exchange,omitempty"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

// WatchlistItem represents a watched security
type WatchlistItem struct {
	ID      string    `json:"id"`
	Ticker  string    `json:"ticker"`
	Name    *string   `json:"name,omitempty"`
	Notes   *string   `json:"notes,omitempty"`
	AddedAt time.Time `json:"added_at"`
}

// resultCodeToError converts a C result code to a Go error
func resultCodeToError(code C.ResultCode) error {
	switch code {
	case C.RESULT_OK:
		return nil
	case C.RESULT_NOT_INITIALIZED:
		return ErrNotInitialized
	case C.RESULT_INVALID_INPUT:
		return ErrInvalidInput
	case C.RESULT_NOT_FOUND:
		return ErrNotFound
	case C.RESULT_AUTH_FAILED:
		return ErrAuthFailed
	default:
		return fmt.Errorf("unknown error code: %d", code)
	}
}

// parseStringResult handles a StringResult from Rust
func parseStringResult(result C.StringResult) (string, error) {
	defer C.gazillioner_free_result(result)

	if result.code != C.RESULT_OK {
		var errMsg string
		if result.error != nil {
			errMsg = C.GoString(result.error)
		}
		err := resultCodeToError(result.code)
		if errMsg != "" {
			return "", fmt.Errorf("%w: %s", err, errMsg)
		}
		return "", err
	}

	if result.data == nil {
		return "", nil
	}

	return C.GoString(result.data), nil
}

// InitDB initializes the encrypted database
func InitDB(dbPath string, deviceKey, salt [32]byte, pin string) error {
	cPath := C.CString(dbPath)
	defer C.free(unsafe.Pointer(cPath))

	cPin := C.CString(pin)
	defer C.free(unsafe.Pointer(cPin))

	code := C.gazillioner_db_init(
		cPath,
		(*C.uint8_t)(unsafe.Pointer(&deviceKey[0])),
		32,
		(*C.uint8_t)(unsafe.Pointer(&salt[0])),
		32,
		cPin,
	)

	return resultCodeToError(code)
}

// CloseDB closes the database connection
func CloseDB() error {
	code := C.gazillioner_db_close()
	return resultCodeToError(code)
}

// IsDBInitialized checks if the database is initialized
func IsDBInitialized() bool {
	return bool(C.gazillioner_db_is_initialized())
}

// ValidatePIN checks if a PIN has valid format (6 digits)
func ValidatePIN(pin string) bool {
	cPin := C.CString(pin)
	defer C.free(unsafe.Pointer(cPin))
	return bool(C.gazillioner_validate_pin(cPin))
}

// Version returns the library version
func Version() string {
	return C.GoString(C.gazillioner_version())
}

// CreateHolding creates a new portfolio holding
func CreateHolding(ticker string, quantity, costBasis float64, notes *string) (*Holding, error) {
	cTicker := C.CString(ticker)
	defer C.free(unsafe.Pointer(cTicker))

	var cNotes *C.char
	if notes != nil {
		cNotes = C.CString(*notes)
		defer C.free(unsafe.Pointer(cNotes))
	}

	result := C.gazillioner_holding_create(cTicker, C.double(quantity), C.double(costBasis), cNotes)
	jsonStr, err := parseStringResult(result)
	if err != nil {
		return nil, err
	}

	var holding Holding
	if err := json.Unmarshal([]byte(jsonStr), &holding); err != nil {
		return nil, fmt.Errorf("failed to parse holding: %w", err)
	}

	return &holding, nil
}

// ListHoldings returns all portfolio holdings
func ListHoldings() ([]Holding, error) {
	result := C.gazillioner_holding_list()
	jsonStr, err := parseStringResult(result)
	if err != nil {
		return nil, err
	}

	var holdings []Holding
	if err := json.Unmarshal([]byte(jsonStr), &holdings); err != nil {
		return nil, fmt.Errorf("failed to parse holdings: %w", err)
	}

	return holdings, nil
}

// GetHolding retrieves a holding by ID
func GetHolding(id string) (*Holding, error) {
	cID := C.CString(id)
	defer C.free(unsafe.Pointer(cID))

	result := C.gazillioner_holding_get(cID)
	jsonStr, err := parseStringResult(result)
	if err != nil {
		return nil, err
	}

	var holding Holding
	if err := json.Unmarshal([]byte(jsonStr), &holding); err != nil {
		return nil, fmt.Errorf("failed to parse holding: %w", err)
	}

	return &holding, nil
}

// DeleteHolding removes a holding by ID
func DeleteHolding(id string) error {
	cID := C.CString(id)
	defer C.free(unsafe.Pointer(cID))

	code := C.gazillioner_holding_delete(cID)
	return resultCodeToError(code)
}

// AddToWatchlist adds a ticker to the watchlist
func AddToWatchlist(ticker string, notes *string) (*WatchlistItem, error) {
	cTicker := C.CString(ticker)
	defer C.free(unsafe.Pointer(cTicker))

	var cNotes *C.char
	if notes != nil {
		cNotes = C.CString(*notes)
		defer C.free(unsafe.Pointer(cNotes))
	}

	result := C.gazillioner_watchlist_add(cTicker, cNotes)
	jsonStr, err := parseStringResult(result)
	if err != nil {
		return nil, err
	}

	var item WatchlistItem
	if err := json.Unmarshal([]byte(jsonStr), &item); err != nil {
		return nil, fmt.Errorf("failed to parse watchlist item: %w", err)
	}

	return &item, nil
}

// ListWatchlist returns all watchlist items
func ListWatchlist() ([]WatchlistItem, error) {
	result := C.gazillioner_watchlist_list()
	jsonStr, err := parseStringResult(result)
	if err != nil {
		return nil, err
	}

	var items []WatchlistItem
	if err := json.Unmarshal([]byte(jsonStr), &items); err != nil {
		return nil, fmt.Errorf("failed to parse watchlist: %w", err)
	}

	return items, nil
}

// RemoveFromWatchlist removes a ticker from the watchlist
func RemoveFromWatchlist(ticker string) error {
	cTicker := C.CString(ticker)
	defer C.free(unsafe.Pointer(cTicker))

	code := C.gazillioner_watchlist_remove(cTicker)
	return resultCodeToError(code)
}
