package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

var queryCmd = &cobra.Command{
	Use:   "query [question]",
	Short: "Ask the AI about your portfolio or markets",
	Long: `Query the local AI with questions about your portfolio, market conditions,
or financial concepts. All processing happens on-device.

Examples:
  gazillioner query "Analyze my portfolio risk"
  gazillioner query "Compare AAPL vs MSFT"
  gazillioner query "What's my tech sector exposure?"
  gazillioner query "Explain dollar cost averaging"`,
	Args: cobra.MinimumNArgs(1),
	Run:  runQuery,
}

func init() {
	rootCmd.AddCommand(queryCmd)

	queryCmd.Flags().Bool("stream", true, "Stream response tokens")
	queryCmd.Flags().Bool("no-context", false, "Query without portfolio context")
	queryCmd.Flags().Int("max-tokens", 2048, "Maximum tokens in response")
	queryCmd.Flags().Float64("temperature", 0.7, "Sampling temperature (0.0-1.0)")
}

func runQuery(cmd *cobra.Command, args []string) {
	question := strings.Join(args, " ")
	stream, _ := cmd.Flags().GetBool("stream")
	noContext, _ := cmd.Flags().GetBool("no-context")
	maxTokens, _ := cmd.Flags().GetInt("max-tokens")
	temperature, _ := cmd.Flags().GetFloat64("temperature")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	fmt.Printf("Question: %s\n\n", question)

	// TODO: Connect to inference service via gRPC
	_ = ctx
	_ = stream
	_ = noContext
	_ = maxTokens
	_ = temperature

	// Placeholder response
	fmt.Println("Connecting to local inference engine...")
	fmt.Println()
	fmt.Println("---")
	fmt.Println("DISCLAIMER: This is not financial advice. AI responses are for")
	fmt.Println("informational purposes only. Always consult a qualified financial")
	fmt.Println("advisor before making investment decisions.")
	fmt.Println("---")
	fmt.Println()
	fmt.Println("[Inference engine not yet connected. Run 'gazillioner status' for details.]")

	os.Exit(0)
}
