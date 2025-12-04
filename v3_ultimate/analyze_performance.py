#!/usr/bin/env python3
"""
性能分析工具 - 解析测试结果并生成报告

用法:
    python3 analyze_performance.py test_results_*.txt
"""

import sys
import re
from pathlib import Path

def parse_results(filename):
    """解析测试结果文件"""
    with open(filename, 'r') as f:
        content = f.read()

    results = {
        'cpu': None,
        'compiler': None,
        'v3_compiled': False,
        'performances': []
    }

    # 提取CPU信息
    cpu_match = re.search(r'CPU: (.+)', content)
    if cpu_match:
        results['cpu'] = cpu_match.group(1).strip()

    # 提取编译器信息
    if 'Intel Fortran' in content:
        results['compiler'] = 'Intel ifort'
    elif 'GNU Fortran' in content:
        results['compiler'] = 'GNU gfortran'

    # 提取性能数据
    perf_pattern = r'Time:\s+([\d.]+)\s+ms.*?QPS:\s+([\d.]+).*?TFLOPS:\s+([\d.]+)'
    for match in re.finditer(perf_pattern, content, re.DOTALL):
        results['performances'].append({
            'time_ms': float(match.group(1)),
            'qps': float(match.group(2)),
            'tflops': float(match.group(3))
        })

    return results

def analyze_performance(results):
    """分析性能并给出建议"""
    print("=" * 70)
    print("性能分析报告")
    print("=" * 70)
    print()

    print(f"CPU: {results['cpu']}")
    print(f"编译器: {results['compiler']}")
    print()

    if not results['performances']:
        print("❌ 未检测到性能数据")
        print()
        print("可能原因:")
        print("  1. 编译失败")
        print("  2. 运行崩溃")
        print("  3. 输出格式不匹配")
        print()
        print("建议: 查看 compile_*.log 文件中的错误信息")
        return

    # 分析最佳性能
    best = max(results['performances'], key=lambda x: x['tflops'])
    print(f"✅ 检测到 {len(results['performances'])} 组性能数据")
    print()
    print("最佳性能:")
    print(f"  延迟:     {best['time_ms']:.2f} ms")
    print(f"  吞吐量:   {best['qps']:.0f} QPS")
    print(f"  算力:     {best['tflops']:.2f} TFLOPS")
    print()

    # 性能评级
    print("性能评级:")
    if best['tflops'] > 1.0:
        print("  🏆 优秀 (> 1.0 TFLOPS)")
        print("  接近理论峰值，优化效果显著")
    elif best['tflops'] > 0.5:
        print("  ✅ 良好 (0.5-1.0 TFLOPS)")
        print("  性能良好，有进一步优化空间")
    elif best['tflops'] > 0.2:
        print("  ⚠️  一般 (0.2-0.5 TFLOPS)")
        print("  需要进一步优化向量化和内存访问")
    else:
        print("  ❌ 较差 (< 0.2 TFLOPS)")
        print("  可能存在严重性能瓶颈")
    print()

    # 瓶颈分析
    print("瓶颈分析:")

    # 理论峰值 (假设i9-13900K: 24核 × 2.5 GHz × 32 FLOPs/cycle = 1920 GFLOPS)
    theoretical_peak = 1920  # GFLOPS
    efficiency = (best['tflops'] * 1000) / theoretical_peak * 100

    print(f"  理论峰值:  {theoretical_peak} GFLOPS")
    print(f"  实际算力:  {best['tflops']*1000:.0f} GFLOPS")
    print(f"  效率:      {efficiency:.1f}%")
    print()

    if efficiency < 30:
        print("  主要瓶颈可能是:")
        print("    • 内存带宽不足 (量化数据解包开销)")
        print("    • 向量化不充分 (检查编译器报告)")
        print("    • 缓存命中率低 (优化数据局部性)")
    elif efficiency < 60:
        print("  主要瓶颈可能是:")
        print("    • 解包LUT访问开销")
        print("    • 寄存器溢出")
        print("    • 分支预测失败")
    else:
        print("  ✅ 性能接近最优")
    print()

    # 建议
    print("优化建议:")
    if efficiency < 50:
        print("  1. 检查编译器是否启用了 AVX-512")
        print("     ifort: -xCORE-AVX512")
        print("     gcc:   -march=native")
        print()
        print("  2. 增加分块大小（TILE_M/TILE_N）")
        print()
        print("  3. 使用预取指令 (!dir$ prefetch)")
        print()

    print("  4. 查看编译器优化报告:")
    print("     ifort -qopt-report=5")
    print()
    print("  5. 使用性能分析工具:")
    print("     perf stat -d ./gemv_v3")
    print("     vtune -collect hotspots ./gemv_v3")
    print()

    # 多线程扩展性分析
    if len(results['performances']) > 1:
        print("=" * 70)
        print("多线程扩展性分析")
        print("=" * 70)
        print()

        baseline = results['performances'][0]['tflops']
        for i, perf in enumerate(results['performances']):
            speedup = perf['tflops'] / baseline
            ideal_speedup = i + 1
            efficiency = speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0

            print(f"配置 {i+1}:")
            print(f"  TFLOPS:   {perf['tflops']:.2f}")
            print(f"  加速比:   {speedup:.2f}x (理想: {ideal_speedup:.0f}x)")
            print(f"  效率:     {efficiency:.1f}%")
            print()

def main():
    if len(sys.argv) < 2:
        print("用法: python3 analyze_performance.py test_results_*.txt")
        sys.exit(1)

    result_file = sys.argv[1]

    if not Path(result_file).exists():
        print(f"错误: 文件 {result_file} 不存在")
        sys.exit(1)

    results = parse_results(result_file)
    analyze_performance(results)

if __name__ == "__main__":
    main()
