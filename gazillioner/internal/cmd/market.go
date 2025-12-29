package cmd

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"
)

var marketCmd = &cobra.Command{
	Use:   "market",
	Short: "Get market data and quotes",
	Long:  `Retrieve stock quotes, historical prices, and market data.`,
}

var marketQuoteCmd = &cobra.Command{
	Use:   "quote [ticker...]",
	Short: "Get current quotes",
	Long: `Get current price quotes for one or more tickers.

Examples:
  gazillioner market quote AAPL
  gazillioner market quote AAPL MSFT GOOGL
  gazillioner market quote BTC-USD ETH-USD`,
	Args: cobra.MinimumNArgs(1),
	Run:  runMarketQuote,
}

var marketHistoryCmd = &cobra.Command{
	Use:   "history [ticker]",
	Short: "Get historical prices",
	Long: `Get historical price data for a ticker.

Examples:
  gazillioner market history AAPL --days 30
  gazillioner market history BTC-USD --days 365`,
	Args: cobra.ExactArgs(1),
	Run:  runMarketHistory,
}

var marketChartCmd = &cobra.Command{
	Use:   "chart [ticker]",
	Short: "Display ASCII price chart",
	Long: `Display an ASCII chart of price history.

Examples:
  gazillioner market chart AAPL --period 1M
  gazillioner market chart SPY --period 1Y`,
	Args: cobra.ExactArgs(1),
	Run:  runMarketChart,
}

func init() {
	rootCmd.AddCommand(marketCmd)
	marketCmd.AddCommand(marketQuoteCmd)
	marketCmd.AddCommand(marketHistoryCmd)
	marketCmd.AddCommand(marketChartCmd)

	// History flags
	marketHistoryCmd.Flags().Int("days", 30, "Number of days of history")
	marketHistoryCmd.Flags().String("interval", "1d", "Data interval (1d, 1wk, 1mo)")

	// Chart flags
	marketChartCmd.Flags().String("period", "1M", "Chart period (1D, 1W, 1M, 3M, 1Y, 5Y)")
	marketChartCmd.Flags().Int("width", 80, "Chart width in characters")
	marketChartCmd.Flags().Int("height", 20, "Chart height in lines")
}

func runMarketQuote(cmd *cobra.Command, args []string) {
	tickers := make([]string, len(args))
	for i, t := range args {
		tickers[i] = strings.ToUpper(strings.TrimSpace(t))
	}

	fmt.Println("Market Quotes")
	fmt.Println("=============")
	fmt.Println()

	// TODO: Fetch from market gateway via gRPC
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "TICKER\tPRICE\tCHANGE\t% CHG\tVOLUME\tMKT CAP")
	fmt.Fprintln(w, "------\t-----\t------\t-----\t------\t-------")

	for _, ticker := range tickers {
		fmt.Fprintf(w, "%s\t--\t--\t--\t--\t--\n", ticker)
	}
	w.Flush()

	fmt.Println()
	fmt.Println("[Market gateway not yet connected. Data delayed 15 minutes.]")
}

func runMarketHistory(cmd *cobra.Command, args []string) {
	ticker := strings.ToUpper(args[0])
	days, _ := cmd.Flags().GetInt("days")
	interval, _ := cmd.Flags().GetString("interval")

	fmt.Printf("Historical Prices: %s\n", ticker)
	fmt.Printf("Period: %d days, Interval: %s\n", days, interval)
	fmt.Println("================================")
	fmt.Println()

	// TODO: Fetch from market gateway via gRPC
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "DATE\tOPEN\tHIGH\tLOW\tCLOSE\tVOLUME")
	fmt.Fprintln(w, "----\t----\t----\t---\t-----\t------")
	fmt.Fprintln(w, "[No data available. Market gateway not connected.]")
	w.Flush()
}

func runMarketChart(cmd *cobra.Command, args []string) {
	ticker := strings.ToUpper(args[0])
	period, _ := cmd.Flags().GetString("period")
	width, _ := cmd.Flags().GetInt("width")
	height, _ := cmd.Flags().GetInt("height")

	fmt.Printf("%s - %s\n", ticker, period)
	fmt.Println(strings.Repeat("=", width))
	fmt.Println()

	// TODO: Generate ASCII chart from historical data
	_ = height

	// Placeholder chart
	fmt.Println("    ^")
	fmt.Println("    |")
	fmt.Println("    |     [Chart data unavailable]")
	fmt.Println("    |")
	fmt.Println("    |")
	fmt.Println("    +---------------------------------->")
	fmt.Println()
	fmt.Println("[Market gateway not connected. Cannot generate chart.]")
}
