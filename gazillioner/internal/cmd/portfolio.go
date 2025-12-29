package cmd

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"
)

var portfolioCmd = &cobra.Command{
	Use:   "portfolio",
	Short: "Manage your portfolio holdings",
	Long: `Manage your portfolio of stocks, ETFs, and other assets.
All data is stored locally with encryption.`,
}

var portfolioListCmd = &cobra.Command{
	Use:   "list",
	Short: "List all holdings",
	Run:   runPortfolioList,
}

var portfolioAddCmd = &cobra.Command{
	Use:   "add [ticker] [quantity] [cost_basis]",
	Short: "Add a new holding",
	Long: `Add a new holding to your portfolio.

Examples:
  gazillioner portfolio add AAPL 100 150.00
  gazillioner portfolio add BTC 0.5 45000.00
  gazillioner portfolio add VTSAX 50 200.00 --date 2024-01-15`,
	Args: cobra.ExactArgs(3),
	Run:  runPortfolioAdd,
}

var portfolioRemoveCmd = &cobra.Command{
	Use:   "remove [ticker]",
	Short: "Remove a holding",
	Args:  cobra.ExactArgs(1),
	Run:   runPortfolioRemove,
}

var portfolioImportCmd = &cobra.Command{
	Use:   "import [file.csv]",
	Short: "Import holdings from CSV file",
	Long: `Import portfolio holdings from a CSV file.

Expected columns: ticker, quantity, cost_basis, date (optional)

Example CSV:
  ticker,quantity,cost_basis,date
  AAPL,100,150.00,2024-01-15
  MSFT,50,380.00,2024-02-01`,
	Args: cobra.ExactArgs(1),
	Run:  runPortfolioImport,
}

var portfolioExportCmd = &cobra.Command{
	Use:   "export",
	Short: "Export holdings to CSV",
	Run:   runPortfolioExport,
}

func init() {
	rootCmd.AddCommand(portfolioCmd)
	portfolioCmd.AddCommand(portfolioListCmd)
	portfolioCmd.AddCommand(portfolioAddCmd)
	portfolioCmd.AddCommand(portfolioRemoveCmd)
	portfolioCmd.AddCommand(portfolioImportCmd)
	portfolioCmd.AddCommand(portfolioExportCmd)

	// Add flags
	portfolioAddCmd.Flags().String("date", "", "Acquisition date (YYYY-MM-DD)")
	portfolioAddCmd.Flags().String("notes", "", "Optional notes")

	portfolioExportCmd.Flags().StringP("output", "o", "portfolio.csv", "Output file path")
	portfolioExportCmd.Flags().String("format", "csv", "Export format (csv, json)")
}

func runPortfolioList(cmd *cobra.Command, args []string) {
	// TODO: Fetch from gRPC service
	fmt.Println("Portfolio Holdings")
	fmt.Println("==================")
	fmt.Println()

	// Mock data for demonstration
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "TICKER\tQTY\tCOST BASIS\tCURRENT\tGAIN/LOSS\tDAY CHG")
	fmt.Fprintln(w, "------\t---\t----------\t-------\t---------\t-------")
	fmt.Fprintln(w, "[No holdings yet. Use 'gazillioner portfolio add' to add holdings.]")
	w.Flush()

	fmt.Println()
	fmt.Println("Total Value: $0.00")
	fmt.Println("Day Change:  $0.00 (0.00%)")
}

func runPortfolioAdd(cmd *cobra.Command, args []string) {
	ticker := strings.ToUpper(args[0])
	quantity, err := strconv.ParseFloat(args[1], 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: invalid quantity '%s'\n", args[1])
		os.Exit(1)
	}
	costBasis, err := strconv.ParseFloat(args[2], 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: invalid cost basis '%s'\n", args[2])
		os.Exit(1)
	}

	dateStr, _ := cmd.Flags().GetString("date")
	var acquisitionDate time.Time
	if dateStr != "" {
		acquisitionDate, err = time.Parse("2006-01-02", dateStr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: invalid date format '%s' (use YYYY-MM-DD)\n", dateStr)
			os.Exit(1)
		}
	} else {
		acquisitionDate = time.Now()
	}

	notes, _ := cmd.Flags().GetString("notes")

	// TODO: Validate ticker via market service
	// TODO: Add via gRPC service

	fmt.Printf("Adding holding:\n")
	fmt.Printf("  Ticker:      %s\n", ticker)
	fmt.Printf("  Quantity:    %.4f\n", quantity)
	fmt.Printf("  Cost Basis:  $%.2f\n", costBasis)
	fmt.Printf("  Total Cost:  $%.2f\n", quantity*costBasis)
	fmt.Printf("  Date:        %s\n", acquisitionDate.Format("2006-01-02"))
	if notes != "" {
		fmt.Printf("  Notes:       %s\n", notes)
	}
	fmt.Println()
	fmt.Println("[Database not yet connected. Holding not saved.]")
}

func runPortfolioRemove(cmd *cobra.Command, args []string) {
	ticker := strings.ToUpper(args[0])

	// TODO: Remove via gRPC service
	fmt.Printf("Removing holding: %s\n", ticker)
	fmt.Println("[Database not yet connected. Holding not removed.]")
}

func runPortfolioImport(cmd *cobra.Command, args []string) {
	filePath := args[0]

	file, err := os.Open(filePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening file: %v\n", err)
		os.Exit(1)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading CSV: %v\n", err)
		os.Exit(1)
	}

	if len(records) < 2 {
		fmt.Fprintln(os.Stderr, "Error: CSV file must have header and at least one data row")
		os.Exit(1)
	}

	// Parse header
	header := records[0]
	colMap := make(map[string]int)
	for i, col := range header {
		colMap[strings.ToLower(strings.TrimSpace(col))] = i
	}

	// Validate required columns
	required := []string{"ticker", "quantity", "cost_basis"}
	for _, col := range required {
		if _, ok := colMap[col]; !ok {
			fmt.Fprintf(os.Stderr, "Error: missing required column '%s'\n", col)
			os.Exit(1)
		}
	}

	fmt.Printf("Importing %d holdings from %s\n", len(records)-1, filePath)
	fmt.Println()

	for i, row := range records[1:] {
		ticker := strings.ToUpper(strings.TrimSpace(row[colMap["ticker"]]))
		quantity := strings.TrimSpace(row[colMap["quantity"]])
		costBasis := strings.TrimSpace(row[colMap["cost_basis"]])

		fmt.Printf("  %d. %s: %s @ $%s\n", i+1, ticker, quantity, costBasis)
	}

	fmt.Println()
	fmt.Println("[Database not yet connected. Holdings not imported.]")
}

func runPortfolioExport(cmd *cobra.Command, args []string) {
	output, _ := cmd.Flags().GetString("output")
	format, _ := cmd.Flags().GetString("format")

	fmt.Printf("Exporting portfolio to %s (format: %s)\n", output, format)
	fmt.Println("[Database not yet connected. Nothing to export.]")
}
