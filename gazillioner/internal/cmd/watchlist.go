package cmd

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"
)

var watchlistCmd = &cobra.Command{
	Use:   "watchlist",
	Short: "Manage your watchlist",
	Long:  `Manage a list of securities you want to monitor without owning.`,
}

var watchlistListCmd = &cobra.Command{
	Use:   "list",
	Short: "Show watchlist",
	Run:   runWatchlistList,
}

var watchlistAddCmd = &cobra.Command{
	Use:   "add [ticker...]",
	Short: "Add tickers to watchlist",
	Long: `Add one or more tickers to your watchlist.

Examples:
  gazillioner watchlist add NVDA
  gazillioner watchlist add GOOGL AMZN META`,
	Args: cobra.MinimumNArgs(1),
	Run:  runWatchlistAdd,
}

var watchlistRemoveCmd = &cobra.Command{
	Use:   "remove [ticker...]",
	Short: "Remove tickers from watchlist",
	Args:  cobra.MinimumNArgs(1),
	Run:   runWatchlistRemove,
}

func init() {
	rootCmd.AddCommand(watchlistCmd)
	watchlistCmd.AddCommand(watchlistListCmd)
	watchlistCmd.AddCommand(watchlistAddCmd)
	watchlistCmd.AddCommand(watchlistRemoveCmd)
}

func runWatchlistList(cmd *cobra.Command, args []string) {
	fmt.Println("Watchlist")
	fmt.Println("=========")
	fmt.Println()

	// TODO: Fetch from gRPC service
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "TICKER\tPRICE\tCHANGE\t% CHG\t52W HIGH\t52W LOW")
	fmt.Fprintln(w, "------\t-----\t------\t-----\t--------\t-------")
	fmt.Fprintln(w, "[Watchlist empty. Use 'gazillioner watchlist add TICKER' to add items.]")
	w.Flush()
}

func runWatchlistAdd(cmd *cobra.Command, args []string) {
	tickers := make([]string, len(args))
	for i, t := range args {
		tickers[i] = strings.ToUpper(strings.TrimSpace(t))
	}

	// TODO: Validate tickers via market service
	// TODO: Add via gRPC service

	fmt.Printf("Adding to watchlist: %s\n", strings.Join(tickers, ", "))
	fmt.Println("[Database not yet connected. Watchlist not updated.]")
}

func runWatchlistRemove(cmd *cobra.Command, args []string) {
	tickers := make([]string, len(args))
	for i, t := range args {
		tickers[i] = strings.ToUpper(strings.TrimSpace(t))
	}

	// TODO: Remove via gRPC service

	fmt.Printf("Removing from watchlist: %s\n", strings.Join(tickers, ", "))
	fmt.Println("[Database not yet connected. Watchlist not updated.]")
}
