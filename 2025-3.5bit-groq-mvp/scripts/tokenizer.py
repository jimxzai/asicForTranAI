#!/usr/bin/env python3
"""
SentencePiece Tokenizer for LLaMA 70B
Provides text ↔ token ID conversion

Requires: pip install sentencepiece
"""

import argparse
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:
    print("✗ Error: sentencepiece not installed")
    print("\nInstall with:")
    print("  pip install sentencepiece")
    exit(1)


class LLaMATokenizer:
    """
    LLaMA tokenizer wrapper for SentencePiece
    """

    def __init__(self, model_path="tokenizer.model"):
        """
        Initialize tokenizer

        Args:
            model_path: Path to tokenizer.model file
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Tokenizer model not found: {model_path}\n"
                "Download from: https://huggingface.co/meta-llama/Llama-2-70b-hf"
            )

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))

        # LLaMA special tokens
        self.bos_id = self.sp.bos_id()  # Beginning of sequence
        self.eos_id = self.sp.eos_id()  # End of sequence
        self.pad_id = self.sp.pad_id()  # Padding
        self.unk_id = self.sp.unk_id()  # Unknown

        print(f"✓ Loaded tokenizer from {model_path}")
        print(f"  Vocab size: {self.vocab_size()}")
        print(f"  BOS ID: {self.bos_id}")
        print(f"  EOS ID: {self.eos_id}")

    def vocab_size(self):
        """Get vocabulary size"""
        return self.sp.vocab_size()

    def encode(self, text, add_bos=True, add_eos=False):
        """
        Encode text to token IDs

        Args:
            text: Input text string
            add_bos: Add BOS token at start
            add_eos: Add EOS token at end

        Returns:
            List of token IDs
        """
        token_ids = self.sp.encode(text)

        if add_bos:
            token_ids = [self.bos_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_id]

        return token_ids

    def decode(self, token_ids):
        """
        Decode token IDs to text

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        return self.sp.decode(token_ids)

    def encode_to_file(self, text, output_file, add_bos=True):
        """
        Encode text and save token IDs to binary file for Fortran

        Args:
            text: Input text
            output_file: Output file path
            add_bos: Add BOS token

        Returns:
            Number of tokens written
        """
        token_ids = self.encode(text, add_bos=add_bos)

        # Write as INT32 binary (Fortran compatible)
        import struct

        with open(output_file, 'wb') as f:
            # Write number of tokens first
            f.write(struct.pack('i', len(token_ids)))

            # Write token IDs
            for token_id in token_ids:
                f.write(struct.pack('i', token_id))

        print(f"✓ Encoded {len(token_ids)} tokens to {output_file}")
        return len(token_ids)

    def decode_from_file(self, input_file):
        """
        Read token IDs from binary file and decode to text

        Args:
            input_file: Input file path

        Returns:
            Decoded text
        """
        import struct

        with open(input_file, 'rb') as f:
            # Read number of tokens
            num_tokens = struct.unpack('i', f.read(4))[0]

            # Read token IDs
            token_ids = []
            for _ in range(num_tokens):
                token_id = struct.unpack('i', f.read(4))[0]
                token_ids.append(token_id)

        text = self.decode(token_ids)
        print(f"✓ Decoded {num_tokens} tokens from {input_file}")
        return text


def main():
    parser = argparse.ArgumentParser(description="LLaMA Tokenizer")
    parser.add_argument(
        "--model",
        default="tokenizer.model",
        help="Path to tokenizer.model file"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode text to token IDs")
    encode_parser.add_argument("text", help="Text to encode")
    encode_parser.add_argument("--output", help="Output file (optional)")
    encode_parser.add_argument("--no-bos", action="store_true", help="Don't add BOS token")

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode token IDs to text")
    decode_parser.add_argument("input", help="Input file with token IDs")

    # Interactive mode
    subparsers.add_parser("interactive", help="Interactive tokenizer")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize tokenizer
    try:
        tokenizer = LLaMATokenizer(args.model)
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        print("\nTo get the tokenizer:")
        print("  1. Visit: https://huggingface.co/meta-llama/Llama-2-70b-hf")
        print("  2. Download tokenizer.model")
        print("  3. Place it in the current directory")
        return

    # Execute command
    if args.command == "encode":
        token_ids = tokenizer.encode(args.text, add_bos=not args.no_bos)
        print(f"\nText: {args.text}")
        print(f"Tokens: {len(token_ids)}")
        print(f"Token IDs: {token_ids}")

        if args.output:
            tokenizer.encode_to_file(args.text, args.output, add_bos=not args.no_bos)

    elif args.command == "decode":
        text = tokenizer.decode_from_file(args.input)
        print(f"\nDecoded text:\n{text}")

    elif args.command == "interactive":
        print("\n" + "=" * 60)
        print("Interactive Tokenizer (type 'quit' to exit)")
        print("=" * 60)

        while True:
            try:
                text = input("\nEnter text: ")
                if text.lower() in ['quit', 'exit', 'q']:
                    break

                token_ids = tokenizer.encode(text)
                decoded = tokenizer.decode(token_ids)

                print(f"  Tokens: {len(token_ids)}")
                print(f"  Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
                print(f"  Decoded: {decoded}")

            except KeyboardInterrupt:
                break

        print("\nGoodbye!")


if __name__ == "__main__":
    main()
