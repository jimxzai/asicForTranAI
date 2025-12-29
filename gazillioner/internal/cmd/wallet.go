package cmd

import (
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"
)

var walletCmd = &cobra.Command{
	Use:   "wallet",
	Short: "Manage crypto cold storage wallet",
	Long: `Manage your Bitcoin cold storage wallet.
Generate receive addresses and check balances.

All private keys are generated and stored locally with encryption.
Signing happens in an air-gapped environment (V1.1 feature).`,
}

var walletBalanceCmd = &cobra.Command{
	Use:   "balance",
	Short: "Show wallet balances",
	Run:   runWalletBalance,
}

var walletReceiveCmd = &cobra.Command{
	Use:   "receive",
	Short: "Generate a receive address",
	Long: `Generate a new receive address for deposits.

Examples:
  gazillioner wallet receive --coin btc
  gazillioner wallet receive --coin btc --label "Exchange withdrawal"`,
	Run: runWalletReceive,
}

var walletAddressesCmd = &cobra.Command{
	Use:   "addresses",
	Short: "List all addresses",
	Run:   runWalletAddresses,
}

func init() {
	rootCmd.AddCommand(walletCmd)
	walletCmd.AddCommand(walletBalanceCmd)
	walletCmd.AddCommand(walletReceiveCmd)
	walletCmd.AddCommand(walletAddressesCmd)

	// Receive flags
	walletReceiveCmd.Flags().String("coin", "btc", "Coin type (btc)")
	walletReceiveCmd.Flags().String("label", "", "Optional label for address")
}

func runWalletBalance(cmd *cobra.Command, args []string) {
	fmt.Println("Wallet Balances")
	fmt.Println("===============")
	fmt.Println()

	// TODO: Fetch balances via blockchain API (through market gateway)
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "COIN\tBALANCE\tUSD VALUE\tADDRESSES")
	fmt.Fprintln(w, "----\t-------\t---------\t---------")
	fmt.Fprintln(w, "BTC\t0.00000000\t$0.00\t0")
	w.Flush()

	fmt.Println()
	fmt.Println("Total Value: $0.00")
	fmt.Println()
	fmt.Println("[Wallet not initialized. Run 'gazillioner wallet receive' to create first address.]")
}

func runWalletReceive(cmd *cobra.Command, args []string) {
	coin, _ := cmd.Flags().GetString("coin")
	label, _ := cmd.Flags().GetString("label")

	coin = strings.ToUpper(coin)
	if coin != "BTC" {
		fmt.Fprintf(os.Stderr, "Error: only BTC is supported in MVP\n")
		os.Exit(1)
	}

	fmt.Println("Generating new receive address...")
	fmt.Println()

	// TODO: Generate HD wallet address
	// TODO: Store in encrypted database

	fmt.Println("Coin:    BTC")
	if label != "" {
		fmt.Printf("Label:   %s\n", label)
	}
	fmt.Println("Address: [Wallet not yet initialized]")
	fmt.Println()
	fmt.Println("SECURITY NOTICE:")
	fmt.Println("  - Private keys never leave this device")
	fmt.Println("  - Backup your recovery phrase securely")
	fmt.Println("  - Verify addresses on a second device before large transfers")
}

func runWalletAddresses(cmd *cobra.Command, args []string) {
	fmt.Println("Wallet Addresses")
	fmt.Println("================")
	fmt.Println()

	// TODO: Fetch from encrypted database
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "COIN\tADDRESS\tLABEL\tBALANCE\tTXS")
	fmt.Fprintln(w, "----\t-------\t-----\t-------\t---")
	fmt.Fprintln(w, "[No addresses generated yet.]")
	w.Flush()
}
