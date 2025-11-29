#!/usr/bin/env python3
"""
Download LLaMA 70B AWQ weights from Hugging Face
Requires: pip install huggingface_hub
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def download_llama_weights(model_id="TheBloke/Llama-2-70B-AWQ", output_dir="./llama-70b-awq"):
    """
    Download LLaMA 70B AWQ weights from Hugging Face

    Args:
        model_id: Hugging Face model ID
        output_dir: Local directory to save weights
    """
    print("=" * 60)
    print("LLaMA 70B AWQ Weight Downloader")
    print("=" * 60)
    print(f"\nModel: {model_id}")
    print(f"Output: {output_dir}")
    print("\nNote: This will download ~140GB of data!")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nDownloading model files...")
    print("(This may take several hours depending on your connection)")
    print("")

    try:
        # Download all files
        snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,  # Copy files instead of symlinking
            resume_download=True,  # Resume if interrupted
        )

        print("\n" + "=" * 60)
        print("✓ Download complete!")
        print("=" * 60)
        print(f"\nFiles saved to: {output_path.absolute()}")

        # List downloaded files
        print("\nDownloaded files:")
        for file in sorted(output_path.rglob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  {file.name:50s} {size_mb:>8.1f} MB")

        print("\n" + "=" * 60)
        print("Next steps:")
        print("  1. Run: python scripts/convert_weights_to_fortran.py")
        print("  2. This will convert safetensors → Fortran binary")
        print("=" * 60)

        return True

    except KeyboardInterrupt:
        print("\n\n✗ Download interrupted by user")
        print("You can resume by running this script again.")
        return False

    except Exception as e:
        print(f"\n✗ Error downloading weights: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Verify you have enough disk space (~140GB)")
        print("  3. Try: pip install --upgrade huggingface_hub")
        print("  4. Check https://huggingface.co/TheBloke/Llama-2-70B-AWQ")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download LLaMA 70B AWQ weights from Hugging Face"
    )
    parser.add_argument(
        "--model",
        default="TheBloke/Llama-2-70B-AWQ",
        help="Hugging Face model ID (default: TheBloke/Llama-2-70B-AWQ)"
    )
    parser.add_argument(
        "--output",
        default="./llama-70b-awq",
        help="Output directory (default: ./llama-70b-awq)"
    )

    args = parser.parse_args()

    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
        print(f"✓ Using huggingface_hub version {huggingface_hub.__version__}")
    except ImportError:
        print("✗ Error: huggingface_hub not installed")
        print("\nInstall with:")
        print("  pip install huggingface_hub")
        exit(1)

    success = download_llama_weights(args.model, args.output)
    exit(0 if success else 1)
