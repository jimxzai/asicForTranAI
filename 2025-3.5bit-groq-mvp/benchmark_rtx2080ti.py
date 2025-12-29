#!/usr/bin/env python3
"""
Comprehensive RTX 2080 Ti Benchmark Suite
Tests neural network operations with 3.5-bit quantization
"""
import numpy as np
import time
import sys
import json

# Suppress Unicode errors on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

try:
    import torch
    PYTORCH_AVAILABLE = torch.cuda.is_available()
    if PYTORCH_AVAILABLE:
        DEVICE = torch.device("cuda:0")
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1e9
except ImportError:
    PYTORCH_AVAILABLE = False

from quantize_weights import quantize_to_3p5bit, dequantize_from_3p5bit

class Benchmark:
    def __init__(self):
        self.results = {}

    def run_numpy_matmul_benchmark(self):
        """Benchmark NumPy matrix multiplication (CPU baseline)"""
        print("\n" + "="*70)
        print("BENCHMARK 1: NumPy MatMul (CPU Baseline)")
        print("="*70)

        sizes = [
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
        ]

        results = []
        for M, K, N in sizes:
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)

            # Warmup
            _ = np.dot(A, B)

            # Benchmark
            times = []
            for _ in range(5):
                start = time.time()
                C = np.dot(A, B)
                times.append(time.time() - start)

            avg_time = np.mean(times)
            flops = 2 * M * N * K
            gflops = flops / avg_time / 1e9

            results.append({
                'size': f"{M}x{K}x{N}",
                'time_ms': avg_time * 1000,
                'gflops': gflops
            })

            print(f"  {M:4d}x{K:4d}x{N:4d}: {avg_time*1000:7.2f} ms  ({gflops:6.2f} GFLOPS)")

        self.results['numpy_matmul'] = results
        return results

    def run_pytorch_gpu_benchmark(self):
        """Benchmark PyTorch GPU operations"""
        if not PYTORCH_AVAILABLE:
            print("\n" + "="*70)
            print("BENCHMARK 2: PyTorch GPU - SKIPPED (PyTorch not available)")
            print("="*70)
            return None

        print("\n" + "="*70)
        print(f"BENCHMARK 2: PyTorch GPU ({GPU_NAME})")
        print("="*70)
        print(f"GPU Memory: {GPU_MEMORY:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print()

        sizes = [
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]

        results = []
        for M, K, N in sizes:
            try:
                A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
                B = torch.randn(K, N, device=DEVICE, dtype=torch.float32)

                # Warmup
                torch.cuda.synchronize()
                _ = torch.matmul(A, B)
                torch.cuda.synchronize()

                # Benchmark
                times = []
                for _ in range(10):
                    start = time.time()
                    C = torch.matmul(A, B)
                    torch.cuda.synchronize()
                    times.append(time.time() - start)

                avg_time = np.mean(times)
                flops = 2 * M * N * K
                tflops = flops / avg_time / 1e12

                results.append({
                    'size': f"{M}x{K}x{N}",
                    'time_ms': avg_time * 1000,
                    'tflops': tflops
                })

                print(f"  {M:4d}x{K:4d}x{N:4d}: {avg_time*1000:7.2f} ms  ({tflops:6.2f} TFLOPS)")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  {M:4d}x{K:4d}x{N:4d}: OUT OF MEMORY")
                    break
                else:
                    raise

        # Memory usage
        print()
        print(f"Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

        self.results['pytorch_gpu'] = results
        return results

    def run_quantization_benchmark(self):
        """Benchmark 3.5-bit quantization"""
        print("\n" + "="*70)
        print("BENCHMARK 3: 3.5-bit Quantization Performance")
        print("="*70)

        sizes = [
            (128, 128),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
        ]

        results = []
        for K, N in sizes:
            weights = np.random.randn(K, N).astype(np.float32) * 0.1

            # Benchmark quantization
            times = []
            for _ in range(5):
                start = time.time()
                w_pack, scales, offsets = quantize_to_3p5bit(weights)
                times.append(time.time() - start)

            quant_time = np.mean(times)

            # Benchmark dequantization
            times = []
            for _ in range(5):
                start = time.time()
                w_recon = dequantize_from_3p5bit(w_pack, scales, offsets)
                times.append(time.time() - start)

            dequant_time = np.mean(times)

            # Compression stats
            original_bytes = K * N * 4
            compressed_bytes = (K // 2) * N + N * 8
            ratio = original_bytes / compressed_bytes

            # Accuracy
            error = np.abs(weights - w_recon)
            mse = np.mean(error ** 2)

            results.append({
                'size': f"{K}x{N}",
                'quant_ms': quant_time * 1000,
                'dequant_ms': dequant_time * 1000,
                'ratio': ratio,
                'mse': mse
            })

            print(f"  {K:4d}x{N:4d}: Quant={quant_time*1000:6.2f}ms, "
                  f"Dequant={dequant_time*1000:6.2f}ms, "
                  f"Ratio={ratio:.2f}x, MSE={mse:.6f}")

        self.results['quantization'] = results
        return results

    def run_end_to_end_inference(self):
        """Simulate end-to-end neural network inference"""
        print("\n" + "="*70)
        print("BENCHMARK 4: End-to-End LLM Inference Simulation")
        print("="*70)

        # Simulate LLaMA-13B layer
        batch_size = 1
        seq_len = 512
        hidden_dim = 5120
        intermediate_dim = 13824

        print(f"Simulating LLaMA-13B layer:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Intermediate dim: {intermediate_dim}")
        print()

        # Input activations
        X = np.random.randn(batch_size * seq_len, hidden_dim).astype(np.float32) * 0.5

        # Weight matrices (3 projections: Q, K, V)
        Wq = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02
        Wk = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02
        Wv = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02

        print("FP32 Baseline:")
        start = time.time()
        Q = np.dot(X, Wq)
        K = np.dot(X, Wk)
        V = np.dot(X, Wv)
        fp32_time = time.time() - start
        print(f"  Time: {fp32_time*1000:.2f} ms")

        print("\n3.5-bit Quantized:")
        # Quantize weights
        Wq_pack, Wq_scales, Wq_offsets = quantize_to_3p5bit(Wq)
        Wk_pack, Wk_scales, Wk_offsets = quantize_to_3p5bit(Wk)
        Wv_pack, Wv_scales, Wv_offsets = quantize_to_3p5bit(Wv)

        start = time.time()
        Wq_dequant = dequantize_from_3p5bit(Wq_pack, Wq_scales, Wq_offsets)
        Wk_dequant = dequantize_from_3p5bit(Wk_pack, Wk_scales, Wk_offsets)
        Wv_dequant = dequantize_from_3p5bit(Wv_pack, Wv_scales, Wv_offsets)

        Q_quant = np.dot(X, Wq_dequant)
        K_quant = np.dot(X, Wk_dequant)
        V_quant = np.dot(X, Wv_dequant)
        quant_time = time.time() - start

        print(f"  Time: {quant_time*1000:.2f} ms")

        # Compare accuracy
        error_q = np.abs(Q - Q_quant).mean()
        error_k = np.abs(K - K_quant).mean()
        error_v = np.abs(V - V_quant).mean()

        print(f"\nAccuracy:")
        print(f"  Q error: {error_q:.6f}")
        print(f"  K error: {error_k:.6f}")
        print(f"  V error: {error_v:.6f}")

        # Memory savings
        fp32_memory = (hidden_dim * hidden_dim * 3 * 4) / 1e6
        quant_memory = (hidden_dim * hidden_dim // 2 * 3 + hidden_dim * 3 * 8) / 1e6

        print(f"\nMemory:")
        print(f"  FP32: {fp32_memory:.2f} MB")
        print(f"  3.5-bit: {quant_memory:.2f} MB")
        print(f"  Savings: {(1 - quant_memory/fp32_memory)*100:.1f}%")

        self.results['inference'] = {
            'fp32_time_ms': fp32_time * 1000,
            'quant_time_ms': quant_time * 1000,
            'fp32_memory_mb': fp32_memory,
            'quant_memory_mb': quant_memory,
            'error': (error_q + error_k + error_v) / 3
        }

    def save_results(self, filename="benchmark_results_rtx2080ti.json"):
        """Save benchmark results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {filename}")

    def run_all(self):
        """Run all benchmarks"""
        print("\n" + "*"*70)
        print("RTX 2080 Ti Comprehensive Benchmark Suite")
        print("3.5-bit Quantization Performance Analysis")
        print("*"*70)

        self.run_numpy_matmul_benchmark()
        self.run_pytorch_gpu_benchmark()
        self.run_quantization_benchmark()
        self.run_end_to_end_inference()

        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70)

        self.save_results()

        print("\nNext steps:")
        print("  1. Review benchmark_results_rtx2080ti.json")
        print("  2. Run with actual LLaMA weights")
        print("  3. Deploy to production")

if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run_all()
