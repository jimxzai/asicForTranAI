#!/usr/bin/env python3
"""
Convert LLaMA AWQ safetensors weights to Fortran binary format
Requires: pip install safetensors numpy
"""

import os
import argparse
import struct
import json
from pathlib import Path
import numpy as np

try:
    from safetensors import safe_open
except ImportError:
    print("✗ Error: safetensors not installed")
    print("\nInstall with:")
    print("  pip install safetensors")
    exit(1)


def convert_layer_weights(safetensors_path, layer_idx, output_dir):
    """
    Convert one layer's weights from safetensors to Fortran binary

    Args:
        safetensors_path: Path to safetensors file
        layer_idx: Layer number (0-79)
        output_dir: Output directory for binary files
    """
    print(f"\nProcessing layer {layer_idx}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open safetensors file
    with safe_open(safetensors_path, framework="numpy") as f:
        # Layer weight names in LLaMA format
        layer_prefix = f"model.layers.{layer_idx}"

        # Attention weights
        wq_key = f"{layer_prefix}.self_attn.q_proj.qweight"
        wk_key = f"{layer_prefix}.self_attn.k_proj.qweight"
        wv_key = f"{layer_prefix}.self_attn.v_proj.qweight"
        wo_key = f"{layer_prefix}.self_attn.o_proj.qweight"

        wq_scales_key = f"{layer_prefix}.self_attn.q_proj.scales"
        wk_scales_key = f"{layer_prefix}.self_attn.k_proj.scales"
        wv_scales_key = f"{layer_prefix}.self_attn.v_proj.scales"
        wo_scales_key = f"{layer_prefix}.self_attn.o_proj.scales"

        # FFN weights
        w_gate_key = f"{layer_prefix}.mlp.gate_proj.qweight"
        w_up_key = f"{layer_prefix}.mlp.up_proj.qweight"
        w_down_key = f"{layer_prefix}.mlp.down_proj.qweight"

        w_gate_scales_key = f"{layer_prefix}.mlp.gate_proj.scales"
        w_up_scales_key = f"{layer_prefix}.mlp.up_proj.scales"
        w_down_scales_key = f"{layer_prefix}.mlp.down_proj.scales"

        # Normalization weights
        attn_norm_key = f"{layer_prefix}.input_layernorm.weight"
        ffn_norm_key = f"{layer_prefix}.post_attention_layernorm.weight"

        # Load tensors
        tensors = {}
        try:
            tensors["wq"] = f.get_tensor(wq_key)
            tensors["wq_scales"] = f.get_tensor(wq_scales_key)
            tensors["wk"] = f.get_tensor(wk_key)
            tensors["wk_scales"] = f.get_tensor(wk_scales_key)
            tensors["wv"] = f.get_tensor(wv_key)
            tensors["wv_scales"] = f.get_tensor(wv_scales_key)
            tensors["wo"] = f.get_tensor(wo_key)
            tensors["wo_scales"] = f.get_tensor(wo_scales_key)

            tensors["w_gate"] = f.get_tensor(w_gate_key)
            tensors["w_gate_scales"] = f.get_tensor(w_gate_scales_key)
            tensors["w_up"] = f.get_tensor(w_up_key)
            tensors["w_up_scales"] = f.get_tensor(w_up_scales_key)
            tensors["w_down"] = f.get_tensor(w_down_key)
            tensors["w_down_scales"] = f.get_tensor(w_down_scales_key)

            tensors["attn_norm"] = f.get_tensor(attn_norm_key)
            tensors["ffn_norm"] = f.get_tensor(ffn_norm_key)

        except Exception as e:
            print(f"  ✗ Error loading tensors: {e}")
            print(f"  Available keys: {list(f.keys())[:10]}...")
            return False

    # Save to Fortran binary format
    output_file = output_dir / f"weights_layer{layer_idx}.bin"

    with open(output_file, "wb") as f:
        # Write in the same order as Fortran expects
        # INT8 weights (packed INT4)
        tensors["wq"].astype(np.int8).tofile(f)
        tensors["wq_scales"].astype(np.float32).tofile(f)

        tensors["wk"].astype(np.int8).tofile(f)
        tensors["wk_scales"].astype(np.float32).tofile(f)

        tensors["wv"].astype(np.int8).tofile(f)
        tensors["wv_scales"].astype(np.float32).tofile(f)

        tensors["wo"].astype(np.int8).tofile(f)
        tensors["wo_scales"].astype(np.float32).tofile(f)

        tensors["w_gate"].astype(np.int8).tofile(f)
        tensors["w_gate_scales"].astype(np.float32).tofile(f)

        tensors["w_up"].astype(np.int8).tofile(f)
        tensors["w_up_scales"].astype(np.float32).tofile(f)

        tensors["w_down"].astype(np.int8).tofile(f)
        tensors["w_down_scales"].astype(np.float32).tofile(f)

        tensors["attn_norm"].astype(np.float32).tofile(f)
        tensors["ffn_norm"].astype(np.float32).tofile(f)

    print(f"  ✓ Saved to {output_file}")

    # Save metadata
    metadata = {
        "layer": layer_idx,
        "shapes": {k: list(v.shape) for k, v in tensors.items()},
        "dtypes": {k: str(v.dtype) for k, v in tensors.items()}
    }

    meta_file = output_dir / f"weights_layer{layer_idx}_meta.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Metadata saved to {meta_file}")

    return True


def convert_all_layers(safetensors_dir, output_dir="./weights_fortran", num_layers=80):
    """
    Convert all 80 layers from safetensors to Fortran binary

    Args:
        safetensors_dir: Directory containing safetensors files
        output_dir: Output directory for Fortran binaries
        num_layers: Number of layers (default 80 for LLaMA 70B)
    """
    print("=" * 60)
    print("LLaMA AWQ → Fortran Binary Converter")
    print("=" * 60)
    print(f"\nInput:  {safetensors_dir}")
    print(f"Output: {output_dir}")
    print(f"Layers: {num_layers}")
    print("=" * 60)

    safetensors_dir = Path(safetensors_dir)

    # Find safetensors files
    safetensors_files = list(safetensors_dir.glob("*.safetensors"))

    if not safetensors_files:
        print(f"\n✗ No safetensors files found in {safetensors_dir}")
        print("\nMake sure you've run download_llama_weights.py first!")
        return False

    print(f"\nFound {len(safetensors_files)} safetensors file(s)")

    # Convert each layer
    success_count = 0
    for layer_idx in range(num_layers):
        # Determine which file contains this layer
        # (LLaMA weights are usually sharded across multiple files)
        success = False
        for st_file in safetensors_files:
            try:
                success = convert_layer_weights(st_file, layer_idx, output_dir)
                if success:
                    success_count += 1
                    break
            except Exception as e:
                continue  # Try next file

        if not success:
            print(f"  ⚠ Could not find weights for layer {layer_idx}")

    print("\n" + "=" * 60)
    print(f"✓ Conversion complete: {success_count}/{num_layers} layers")
    print("=" * 60)

    if success_count == num_layers:
        print("\nAll layers converted successfully!")
        print("\nNext steps:")
        print("  1. Implement load_weights() in llama_model.f90")
        print("  2. Test loading one layer")
        print("  3. Test full 80-layer loading")
        print("  4. Run inference!")
    else:
        print(f"\n⚠ Warning: Only {success_count}/{num_layers} layers converted")
        print("Check the safetensors file structure.")

    return success_count == num_layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LLaMA safetensors weights to Fortran binary"
    )
    parser.add_argument(
        "--input",
        default="./llama-70b-awq",
        help="Directory containing safetensors files"
    )
    parser.add_argument(
        "--output",
        default="./weights_fortran",
        help="Output directory for Fortran binaries"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=80,
        help="Number of layers to convert (default: 80)"
    )

    args = parser.parse_args()

    success = convert_all_layers(args.input, args.output, args.layers)
    exit(0 if success else 1)
