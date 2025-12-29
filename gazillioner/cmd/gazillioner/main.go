// Gazillioner - Self-Sovereign AI Financial Intelligence
// Copyright (c) 2025 Gazillioner Inc. All rights reserved.

package main

import (
	"os"

	"github.com/gazillioner/gazillioner/internal/cmd"
)

// Version information (set via ldflags)
var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

func main() {
	cmd.SetVersionInfo(version, commit, date)
	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
