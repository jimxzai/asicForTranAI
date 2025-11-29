#!/bin/bash
# GNAT/SPARK Installation and Verification Script
# Supports macOS, Linux, and Docker installation methods

set -e  # Exit on error

echo "=== GNAT/SPARK Installation Script ==="
echo "Detecting system..."

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Darwin*)    OS_TYPE="macOS";;
    Linux*)     OS_TYPE="Linux";;
    *)          OS_TYPE="UNKNOWN";;
esac

echo "OS detected: ${OS_TYPE}"

# Function: Docker installation
install_via_docker() {
    echo ""
    echo "=== Installing GNAT via Docker ==="

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo "ERROR: Docker is not running. Please start Docker Desktop and try again."
        echo "Download Docker Desktop from: https://www.docker.com/products/docker-desktop"
        return 1
    fi

    echo "Pulling AdaCore GNAT Community Edition Docker image..."
    docker pull adacore/gnat-ce:latest

    echo "Creating workspace alias..."
    echo "" >> ~/.bashrc
    echo "# GNAT/SPARK via Docker" >> ~/.bashrc
    echo "alias gnat-docker='docker run -it -v \$(pwd):/workspace adacore/gnat-ce bash'" >> ~/.bashrc

    echo "Installation complete! Usage:"
    echo "  gnat-docker                  # Start GNAT Docker container"
    echo "  cd /workspace                # Your current directory is mounted here"
    echo "  gnatprove -P transformer.gpr # Run verification"
}

# Function: Manual download installation
install_via_manual() {
    echo ""
    echo "=== Manual Installation Instructions ==="
    echo "1. Visit AdaCore download page:"
    echo "   https://www.adacore.com/download"
    echo ""
    echo "2. Select 'GNAT Community Edition 2024'"
    echo "3. Choose your platform: ${OS_TYPE}"
    echo "4. Download the installer (approx. 500 MB)"
    echo ""
    echo "5. Run the installer:"
    if [ "${OS_TYPE}" = "macOS" ]; then
        echo "   Open the .dmg file and follow installation wizard"
        echo "   Default path: /usr/local/gnat"
    elif [ "${OS_TYPE}" = "Linux" ]; then
        echo "   chmod +x gnat-*.bin"
        echo "   ./gnat-*.bin"
        echo "   Default path: ~/GNAT/2024"
    fi
    echo ""
    echo "6. Add to PATH:"
    if [ "${OS_TYPE}" = "macOS" ]; then
        echo "   export PATH=/usr/local/gnat/bin:\$PATH"
    else
        echo "   export PATH=~/GNAT/2024/bin:\$PATH"
    fi
    echo ""
    echo "7. Verify installation:"
    echo "   gnatprove --version"
}

# Function: Homebrew installation (macOS only)
install_via_homebrew() {
    echo ""
    echo "=== Installing GNAT via Homebrew (macOS) ==="

    if ! command -v brew &> /dev/null; then
        echo "ERROR: Homebrew not found. Install from https://brew.sh"
        return 1
    fi

    echo "Installing GNAT..."
    brew install gnat

    echo "Installation complete! Verifying..."
    gnatprove --version || echo "WARNING: GNATprove not in PATH"
}

# Function: Verify installation
verify_installation() {
    echo ""
    echo "=== Verifying GNAT Installation ==="

    if command -v gnatprove &> /dev/null; then
        echo "✓ GNATprove found!"
        gnatprove --version

        echo ""
        echo "✓ Testing on transformer project..."
        cd "$(dirname "$0")"

        if [ -f "transformer.gpr" ]; then
            echo "Running quick verification (level 2)..."
            gnatprove -P transformer.gpr --level=2 --timeout=10 || echo "NOTE: Some checks may fail - this is normal for first run"
            echo "✓ Verification test complete!"
        else
            echo "WARNING: transformer.gpr not found. Skipping project test."
        fi
    else
        echo "✗ GNATprove not found in PATH"
        echo "Please add GNAT bin directory to your PATH"
        return 1
    fi
}

# Main installation menu
main() {
    echo ""
    echo "Choose installation method:"
    echo "  1) Docker (recommended - isolated environment)"
    echo "  2) Manual download (official AdaCore installer)"
    if [ "${OS_TYPE}" = "macOS" ]; then
        echo "  3) Homebrew (macOS package manager)"
    fi
    echo "  4) Verify existing installation"
    echo "  5) Exit"
    echo ""
    read -p "Enter choice [1-5]: " choice

    case $choice in
        1)
            install_via_docker
            verify_installation
            ;;
        2)
            install_via_manual
            echo ""
            echo "After manual installation, run:"
            echo "  $0  # To verify installation"
            ;;
        3)
            if [ "${OS_TYPE}" = "macOS" ]; then
                install_via_homebrew
                verify_installation
            else
                echo "ERROR: Homebrew is only available on macOS"
                exit 1
            fi
            ;;
        4)
            verify_installation
            ;;
        5)
            echo "Exiting."
            exit 0
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
}

# Run main menu
main

echo ""
echo "=== Installation Complete ==="
echo "Next steps:"
echo "  1. Navigate to: $(dirname "$0")"
echo "  2. Run: gnatprove -P transformer.gpr --level=2"
echo "  3. See README_GNAT.md for detailed usage"
