package cmd

import (
	"fmt"
	"os"
	"runtime"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"
)

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show system status",
	Long:  `Display the current status of all Gazillioner services and components.`,
	Run:   runStatus,
}

func init() {
	rootCmd.AddCommand(statusCmd)
}

func runStatus(cmd *cobra.Command, args []string) {
	fmt.Println("Gazillioner System Status")
	fmt.Println("=========================")
	fmt.Printf("Version: %s (%s)\n", versionInfo.Version, versionInfo.Commit[:min(7, len(versionInfo.Commit))])
	fmt.Printf("Built:   %s\n", versionInfo.Date)
	fmt.Printf("Time:    %s\n", time.Now().Format(time.RFC3339))
	fmt.Println()

	// System info
	fmt.Println("System")
	fmt.Println("------")
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "  Platform:\t%s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Fprintf(w, "  Go Version:\t%s\n", runtime.Version())
	fmt.Fprintf(w, "  CPUs:\t%d\n", runtime.NumCPU())
	w.Flush()
	fmt.Println()

	// Service status
	fmt.Println("Services")
	fmt.Println("--------")
	w = tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "  gRPC Server:\t%s\t(localhost:50051)\n", statusBadge(false))
	fmt.Fprintf(w, "  Inference Engine:\t%s\t(Llama3-13B)\n", statusBadge(false))
	fmt.Fprintf(w, "  Market Gateway:\t%s\t(Yahoo Finance)\n", statusBadge(false))
	fmt.Fprintf(w, "  Database:\t%s\t(SQLCipher)\n", statusBadge(false))
	w.Flush()
	fmt.Println()

	// Model info
	fmt.Println("AI Model")
	fmt.Println("--------")
	w = tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "  Model:\tLlama 3 13B (3.5-bit quantized)\n")
	fmt.Fprintf(w, "  Size:\t5.7 GB\n")
	fmt.Fprintf(w, "  Context:\t8,192 tokens\n")
	fmt.Fprintf(w, "  Status:\t%s\n", statusBadge(false))
	w.Flush()
	fmt.Println()

	// Portfolio summary
	fmt.Println("Portfolio")
	fmt.Println("---------")
	w = tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "  Holdings:\t0\n")
	fmt.Fprintf(w, "  Watchlist:\t0 items\n")
	fmt.Fprintf(w, "  Total Value:\t$0.00\n")
	w.Flush()
	fmt.Println()

	// Security status
	fmt.Println("Security")
	fmt.Println("--------")
	w = tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "  Encryption:\tAES-256 (SQLCipher)\n")
	fmt.Fprintf(w, "  PIN Protected:\tNo\n")
	fmt.Fprintf(w, "  Air-Gap Mode:\tDisabled\n")
	fmt.Fprintf(w, "  Last Backup:\tNever\n")
	w.Flush()
	fmt.Println()

	fmt.Println("---")
	fmt.Println("Run 'gazillioner --help' for available commands.")
}

func statusBadge(running bool) string {
	if running {
		return "[RUNNING]"
	}
	return "[STOPPED]"
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
