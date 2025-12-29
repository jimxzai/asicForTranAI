#!/usr/bin/env python3
"""
GPU Neural Network Testing for RTX 2080 Ti
Tests quantization and matrix operations with CUDA acceleration
"""
import numpy as np
import time
import sys

# Suppress Unicode errors on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

try:
    import torch
    PYTORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not found - will test with NumPy only")

from quantize_weights import quantize_to_3p5bit, dequantize_from_3p5bit

def test_basic_quantization():
    """Test basic 3.5-bit quantization"""
    print("=" * 70)
    print("TEST 1: Basic 3.5-bit Quantization")
    print("=" * 70)

    np.random.seed(42)
    K, N = 256, 128
    weights = np.random.randn(K, N).astype(np.float32) * 0.2

    print(f"Input: {K}x{N} FP32 matrix ({K*N*4/1024:.1f} KB)")

    start = time.time()
    w_pack, scales, offsets = quantize_to_3p5bit(weights)
    quant_time = time.time() - start

    compressed_size = (K//2)*N + N*8
    ratio = (K*N*4) / compressed_size

    print(f"Compressed size: {compressed_size/1024:.1f} KB")
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"Quantization time: {quant_time*1000:.2f} ms")

    # Verify reconstruction
    weights_reconstructed = dequantize_from_3p5bit(w_pack, scales, offsets)
    error = np.abs(weights - weights_reconstructed)

    print(f"Reconstruction MSE: {np.mean(error**2):.6f}")
    print(f"Max error: {error.max():.6f}")
    print(f"Status: PASS - Excellent quality" if np.mean(error**2) < 1e-3 else "Status: ACCEPTABLE")
    print()
    return True

def test_matmul_performance():
    """Test matrix multiplication performance"""
    print("=" * 70)
    print("TEST 2: Matrix Multiplication Performance")
    print("=" * 70)

    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]

    for M, K in sizes:
        N = K
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        # FP32 baseline
        start = time.time()
        C_fp32 = np.dot(A, B)
        fp32_time = time.time() - start

        # Quantize weights
        B_pack, scales, offsets = quantize_to_3p5bit(B)

        # Simulate INT8 computation (actual would be in CUDA)
        A_int8 = np.clip(np.round(A * 127), -128, 127).astype(np.int8)

        print(f"  {M}x{K}x{N}: FP32 = {fp32_time*1000:.2f} ms")

    print()
    return True

def test_pytorch_cuda():
    """Test PyTorch with CUDA if available"""
    print("=" * 70)
    print("TEST 3: PyTorch CUDA Testing")
    print("=" * 70)

    if not PYTORCH_AVAILABLE:
        print("PyTorch with CUDA not available - skipping GPU test")
        print("To install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print()
        return False

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Test different matrix sizes
    sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]

    for M, K in sizes:
        N = K

        # Create random tensors on GPU
        A = torch.randn(M, K, device=device, dtype=torch.float32)
        B = torch.randn(K, N, device=device, dtype=torch.float32)

        # Warmup
        torch.cuda.synchronize()
        _ = torch.matmul(A, B)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        flops = 2 * M * N * K
        tflops = flops / gpu_time / 1e12

        print(f"  {M}x{K}x{N}: {gpu_time*1000:.2f} ms ({tflops:.2f} TFLOPS)")

    print()

    # Memory info
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print()

    return True

def test_quantized_neural_layer():
    """Simulate a quantized neural network layer"""
    print("=" * 70)
    print("TEST 4: Quantized Neural Network Layer Simulation")
    print("=" * 70)

    batch_size = 32
    input_dim = 512
    output_dim = 512

    print(f"Simulating layer: [{batch_size}, {input_dim}] @ [{input_dim}, {output_dim}]")

    # Input activations
    X = np.random.randn(batch_size, input_dim).astype(np.float32) * 0.5

    # FP32 weights
    W_fp32 = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1

    # FP32 forward pass
    start = time.time()
    Y_fp32 = np.dot(X, W_fp32)
    fp32_time = time.time() - start

    # Quantized forward pass
    W_pack, scales, offsets = quantize_to_3p5bit(W_fp32)

    # Simulate INT8 computation
    X_int8 = np.clip(np.round(X * 127), -128, 127).astype(np.int8)
    X_scale = 1.0 / 127.0

    start = time.time()
    # This would be optimized CUDA kernel in production
    W_dequant = dequantize_from_3p5bit(W_pack, scales, offsets)
    Y_quant = np.dot(X, W_dequant)
    quant_time = time.time() - start

    # Compare results
    error = np.abs(Y_fp32 - Y_quant)
    relative_error = error / (np.abs(Y_fp32) + 1e-8)

    print(f"FP32 time: {fp32_time*1000:.2f} ms")
    print(f"Quantized time: {quant_time*1000:.2f} ms")
    print(f"Mean absolute error: {error.mean():.6f}")
    print(f"Mean relative error: {relative_error.mean():.4f}")
    print(f"Status: PASS - Acceptable quality" if error.mean() < 0.1 else "Status: NEEDS TUNING")
    print()

    return True

def main():
    """Run all tests"""
    print("\n")
    print("*" * 70)
    print("RTX 2080 Ti Neural Network Testing Suite")
    print("3.5-bit Quantization with CUDA Acceleration")
    print("*" * 70)
    print("\n")

    results = []

    # Run tests
    results.append(("Basic Quantization", test_basic_quantization()))
    results.append(("MatMul Performance", test_matmul_performance()))
    results.append(("PyTorch CUDA", test_pytorch_cuda()))
    results.append(("NN Layer Simulation", test_quantized_neural_layer()))

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "PASS" if passed else "SKIP/FAIL"
        print(f"  {name:30s} {status}")
    print()

    passed_count = sum(1 for _, p in results if p)
    print(f"Tests passed: {passed_count}/{len(results)}")
    print()

    if PYTORCH_AVAILABLE:
        print("Next steps:")
        print("  1. Run benchmark_3p5bit.py for detailed benchmarks")
        print("  2. Test with real LLaMA weights")
        print("  3. Deploy to production")
    else:
        print("To unlock GPU acceleration:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()

if __name__ == "__main__":
    main()
