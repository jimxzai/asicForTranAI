package cmd

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Manage configuration settings",
	Long:  `View and modify Gazillioner configuration settings.`,
}

var configShowCmd = &cobra.Command{
	Use:   "show",
	Short: "Show current configuration",
	Run:   runConfigShow,
}

var configSetCmd = &cobra.Command{
	Use:   "set [key] [value]",
	Short: "Set a configuration value",
	Long: `Set a configuration value.

Available settings:
  timezone        Your timezone (e.g., America/New_York)
  currency        Display currency (USD, EUR, GBP, etc.)
  theme           TUI theme (dark, light)
  data_refresh    Market data refresh interval in seconds
  model           AI model to use (llama3-13b)
  max_tokens      Maximum tokens for AI responses

Examples:
  gazillioner config set timezone America/New_York
  gazillioner config set currency EUR
  gazillioner config set theme dark`,
	Args: cobra.ExactArgs(2),
	Run:  runConfigSet,
}

var configGetCmd = &cobra.Command{
	Use:   "get [key]",
	Short: "Get a configuration value",
	Args:  cobra.ExactArgs(1),
	Run:   runConfigGet,
}

var configResetCmd = &cobra.Command{
	Use:   "reset",
	Short: "Reset configuration to defaults",
	Run:   runConfigReset,
}

func init() {
	rootCmd.AddCommand(configCmd)
	configCmd.AddCommand(configShowCmd)
	configCmd.AddCommand(configSetCmd)
	configCmd.AddCommand(configGetCmd)
	configCmd.AddCommand(configResetCmd)

	// Set defaults
	viper.SetDefault("timezone", "UTC")
	viper.SetDefault("currency", "USD")
	viper.SetDefault("theme", "dark")
	viper.SetDefault("data_refresh", 300)
	viper.SetDefault("model", "llama3-13b")
	viper.SetDefault("max_tokens", 2048)
}

func runConfigShow(cmd *cobra.Command, args []string) {
	fmt.Println("Gazillioner Configuration")
	fmt.Println("=========================")
	fmt.Println()

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "KEY\tVALUE\tDESCRIPTION")
	fmt.Fprintln(w, "---\t-----\t-----------")
	fmt.Fprintf(w, "timezone\t%s\tYour timezone\n", viper.GetString("timezone"))
	fmt.Fprintf(w, "currency\t%s\tDisplay currency\n", viper.GetString("currency"))
	fmt.Fprintf(w, "theme\t%s\tTUI color theme\n", viper.GetString("theme"))
	fmt.Fprintf(w, "data_refresh\t%d\tMarket data refresh (seconds)\n", viper.GetInt("data_refresh"))
	fmt.Fprintf(w, "model\t%s\tAI model\n", viper.GetString("model"))
	fmt.Fprintf(w, "max_tokens\t%d\tMax AI response tokens\n", viper.GetInt("max_tokens"))
	w.Flush()

	fmt.Println()
	configPath := viper.ConfigFileUsed()
	if configPath != "" {
		fmt.Printf("Config file: %s\n", configPath)
	} else {
		fmt.Println("Config file: (using defaults)")
	}
}

func runConfigSet(cmd *cobra.Command, args []string) {
	key := strings.ToLower(args[0])
	value := args[1]

	// Validate key
	validKeys := map[string]bool{
		"timezone":     true,
		"currency":     true,
		"theme":        true,
		"data_refresh": true,
		"model":        true,
		"max_tokens":   true,
	}

	if !validKeys[key] {
		fmt.Fprintf(os.Stderr, "Error: unknown configuration key '%s'\n", key)
		fmt.Fprintln(os.Stderr, "Valid keys: timezone, currency, theme, data_refresh, model, max_tokens")
		os.Exit(1)
	}

	// Validate values
	switch key {
	case "theme":
		if value != "dark" && value != "light" {
			fmt.Fprintln(os.Stderr, "Error: theme must be 'dark' or 'light'")
			os.Exit(1)
		}
	case "currency":
		value = strings.ToUpper(value)
		validCurrencies := []string{"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"}
		valid := false
		for _, c := range validCurrencies {
			if value == c {
				valid = true
				break
			}
		}
		if !valid {
			fmt.Fprintf(os.Stderr, "Error: unsupported currency '%s'\n", value)
			os.Exit(1)
		}
	}

	viper.Set(key, value)

	// TODO: Persist to config file
	fmt.Printf("Set %s = %s\n", key, value)
	fmt.Println("[Config persistence not yet implemented.]")
}

func runConfigGet(cmd *cobra.Command, args []string) {
	key := strings.ToLower(args[0])
	value := viper.Get(key)

	if value == nil {
		fmt.Fprintf(os.Stderr, "Error: unknown configuration key '%s'\n", key)
		os.Exit(1)
	}

	fmt.Printf("%s = %v\n", key, value)
}

func runConfigReset(cmd *cobra.Command, args []string) {
	fmt.Println("Resetting configuration to defaults...")

	viper.Set("timezone", "UTC")
	viper.Set("currency", "USD")
	viper.Set("theme", "dark")
	viper.Set("data_refresh", 300)
	viper.Set("model", "llama3-13b")
	viper.Set("max_tokens", 2048)

	fmt.Println("Configuration reset to defaults.")
	fmt.Println("[Config persistence not yet implemented.]")
}
