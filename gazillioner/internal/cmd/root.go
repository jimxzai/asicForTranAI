// Package cmd implements the CLI commands for Gazillioner
package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	cfgFile     string
	versionInfo struct {
		Version string
		Commit  string
		Date    string
	}
)

// rootCmd represents the base command
var rootCmd = &cobra.Command{
	Use:   "gazillioner",
	Short: "Self-Sovereign AI Financial Intelligence",
	Long: `Gazillioner is a privacy-first AI financial assistant that runs entirely on your device.

All data stays local. Zero cloud dependencies. Your portfolio, your rules.

Run without arguments to launch interactive TUI mode, or use subcommands for CLI access.`,
	Run: func(cmd *cobra.Command, args []string) {
		// Launch TUI if no subcommand provided
		launchTUI()
	},
}

// Execute runs the root command
func Execute() error {
	return rootCmd.Execute()
}

// SetVersionInfo sets version information from build flags
func SetVersionInfo(version, commit, date string) {
	versionInfo.Version = version
	versionInfo.Commit = commit
	versionInfo.Date = date
}

func init() {
	cobra.OnInitialize(initConfig)

	// Global flags
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.gazillioner/config.yaml)")
	rootCmd.PersistentFlags().Bool("tui", false, "Force TUI mode")
	rootCmd.PersistentFlags().Bool("json", false, "Output in JSON format")
	rootCmd.PersistentFlags().BoolP("verbose", "v", false, "Verbose output")

	// Bind flags to viper
	viper.BindPFlag("tui", rootCmd.PersistentFlags().Lookup("tui"))
	viper.BindPFlag("json", rootCmd.PersistentFlags().Lookup("json"))
	viper.BindPFlag("verbose", rootCmd.PersistentFlags().Lookup("verbose"))
}

func initConfig() {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		home, err := os.UserHomeDir()
		cobra.CheckErr(err)

		// Look for config in home directory
		viper.AddConfigPath(home + "/.gazillioner")
		viper.AddConfigPath(".")
		viper.SetConfigType("yaml")
		viper.SetConfigName("config")
	}

	viper.AutomaticEnv()
	viper.SetEnvPrefix("GAZILLIONER")

	if err := viper.ReadInConfig(); err == nil {
		if viper.GetBool("verbose") {
			fmt.Fprintln(os.Stderr, "Using config file:", viper.ConfigFileUsed())
		}
	}
}

func launchTUI() {
	fmt.Println("Launching Gazillioner TUI...")
	// Import cycle prevention - the actual TUI is launched from main
	// This is a fallback message
	fmt.Println("Use 'gazillioner --tui' or run without arguments to launch TUI.")
	fmt.Println("For CLI mode, use subcommands:")
	fmt.Println("  gazillioner query \"your question\"")
	fmt.Println("  gazillioner portfolio list")
	fmt.Println("  gazillioner --help")
}
