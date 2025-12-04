#!/bin/bash
##############################################################################
# 3.5-bit Matmul v3 完整测试脚本
#
# 功能：
#   1. 检测编译器
#   2. 编译v3/MPI/ASIC三个版本
#   3. 运行性能基准测试
#   4. 生成详细报告
#
# 用法：
#   chmod +x test_all.sh
#   ./test_all.sh
#
##############################################################################

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() { echo -e "\n${BLUE}========================================${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}========================================${NC}\n"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

RESULTS_FILE="test_results_$(date +%Y%m%d_%H%M%S).txt"

# 开始测试
print_header "3.5-bit Matmul v3 测试套件"
echo "测试时间: $(date)" | tee $RESULTS_FILE
echo "主机: $(hostname)" | tee -a $RESULTS_FILE
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

##############################################################################
# 1. 编译器检测
##############################################################################
print_header "1. 编译器检测"

# 检测Intel编译器
if command -v ifort &> /dev/null; then
    IFORT_VERSION=$(ifort --version | head -n1)
    print_success "Intel Fortran: $IFORT_VERSION"
    echo "Intel Fortran: $IFORT_VERSION" >> $RESULTS_FILE
    HAS_IFORT=true
    FC_FAST="ifort"
    FFLAGS_FAST="-O3 -xCORE-AVX512 -qopenmp -fp-model fast=2 -no-prec-div"
else
    print_warning "Intel Fortran未找到"
    echo "Intel Fortran: 未安装" >> $RESULTS_FILE
    HAS_IFORT=false
fi

# 检测GCC
if command -v gfortran &> /dev/null; then
    GFORTRAN_VERSION=$(gfortran --version | head -n1)
    print_success "GNU Fortran: $GFORTRAN_VERSION"
    echo "GNU Fortran: $GFORTRAN_VERSION" >> $RESULTS_FILE
    HAS_GFORTRAN=true
    if [ "$HAS_IFORT" = false ]; then
        FC_FAST="gfortran"
        FFLAGS_FAST="-O3 -march=native -fopenmp -ffast-math"
    fi
else
    print_error "GCC Fortran未找到"
    echo "GNU Fortran: 未安装" >> $RESULTS_FILE
    HAS_GFORTRAN=false
fi

if [ "$HAS_IFORT" = false ] && [ "$HAS_GFORTRAN" = false ]; then
    print_error "没有找到Fortran编译器！请安装 ifort 或 gfortran"
    exit 1
fi

# 检测MPI
if command -v mpiifort &> /dev/null; then
    print_success "MPI: Intel MPI (mpiifort)"
    echo "MPI: Intel MPI" >> $RESULTS_FILE
    MPIFC="mpiifort"
    HAS_MPI=true
elif command -v mpifort &> /dev/null; then
    print_success "MPI: OpenMPI (mpifort)"
    echo "MPI: OpenMPI" >> $RESULTS_FILE
    MPIFC="mpifort"
    HAS_MPI=true
else
    print_warning "MPI编译器未找到（MPI测试将跳过）"
    echo "MPI: 未安装" >> $RESULTS_FILE
    HAS_MPI=false
fi

echo "" | tee -a $RESULTS_FILE

##############################################################################
# 2. 编译v3 (单机版)
##############################################################################
print_header "2. 编译 v3 (单机AVX-512版本)"

echo "编译命令: $FC_FAST $FFLAGS_FAST matmul_3p5bit_v3.f90" | tee -a $RESULTS_FILE

if $FC_FAST $FFLAGS_FAST -o gemv_v3 matmul_3p5bit_v3.f90 2>&1 | tee compile_v3.log; then
    print_success "v3编译成功"
    echo "v3编译: 成功" >> $RESULTS_FILE
    V3_COMPILED=true
else
    print_error "v3编译失败，查看 compile_v3.log"
    echo "v3编译: 失败" >> $RESULTS_FILE
    cat compile_v3.log >> $RESULTS_FILE
    V3_COMPILED=false
fi

echo "" | tee -a $RESULTS_FILE

##############################################################################
# 3. 编译MPI版本
##############################################################################
if [ "$HAS_MPI" = true ]; then
    print_header "3. 编译 MPI分布式版本"

    if $MPIFC -O3 -o gemv_mpi matmul_3p5bit_mpi.f90 2>&1 | tee compile_mpi.log; then
        print_success "MPI版本编译成功"
        echo "MPI编译: 成功" >> $RESULTS_FILE
        MPI_COMPILED=true
    else
        print_error "MPI版本编译失败，查看 compile_mpi.log"
        echo "MPI编译: 失败" >> $RESULTS_FILE
        cat compile_mpi.log >> $RESULTS_FILE
        MPI_COMPILED=false
    fi
else
    print_warning "跳过MPI编译（MPI未安装）"
    echo "MPI编译: 跳过" >> $RESULTS_FILE
    MPI_COMPILED=false
fi

echo "" | tee -a $RESULTS_FILE

##############################################################################
# 4. 编译ASIC仿真版本
##############################################################################
print_header "4. 编译 ASIC仿真版本"

if gfortran -O3 -o gemv_asic matmul_3p5bit_asic.f90 2>&1 | tee compile_asic.log; then
    print_success "ASIC仿真版本编译成功"
    echo "ASIC编译: 成功" >> $RESULTS_FILE
    ASIC_COMPILED=true
else
    print_error "ASIC仿真版本编译失败，查看 compile_asic.log"
    echo "ASIC编译: 失败" >> $RESULTS_FILE
    cat compile_asic.log >> $RESULTS_FILE
    ASIC_COMPILED=false
fi

echo "" | tee -a $RESULTS_FILE

##############################################################################
# 5. 运行v3性能测试
##############################################################################
if [ "$V3_COMPILED" = true ]; then
    print_header "5. v3性能测试"

    echo "========== v3性能测试 ==========" >> $RESULTS_FILE

    # 单线程测试
    print_success "测试 1: 单线程（OMP_NUM_THREADS=1）"
    export OMP_NUM_THREADS=1
    ./gemv_v3 2>&1 | tee -a $RESULTS_FILE

    # 多线程测试（如果有多核）
    NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
    if [ $NCORES -gt 1 ]; then
        print_success "测试 2: 多线程（OMP_NUM_THREADS=$NCORES）"
        export OMP_NUM_THREADS=$NCORES
        ./gemv_v3 2>&1 | tee -a $RESULTS_FILE

        # 测试不同线程数的扩展性
        print_success "测试 3: 线程扩展性"
        echo "线程扩展性测试:" >> $RESULTS_FILE
        for t in 1 2 4 8 16; do
            if [ $t -le $NCORES ]; then
                export OMP_NUM_THREADS=$t
                echo "  OMP_NUM_THREADS=$t:" >> $RESULTS_FILE
                ./gemv_v3 | grep "Time\|QPS\|TFLOPS" >> $RESULTS_FILE
            fi
        done
    fi

    echo "" | tee -a $RESULTS_FILE
fi

##############################################################################
# 6. 运行MPI测试
##############################################################################
if [ "$MPI_COMPILED" = true ]; then
    print_header "6. MPI分布式测试"

    echo "========== MPI性能测试 ==========" >> $RESULTS_FILE

    # 测试不同进程数
    for np in 1 2 4 8; do
        if [ $np -le $NCORES ]; then
            print_success "测试: $np 进程"
            echo "MPI进程数: $np" >> $RESULTS_FILE
            mpirun -np $np ./gemv_mpi 2>&1 | tee -a $RESULTS_FILE
        fi
    done

    echo "" | tee -a $RESULTS_FILE
fi

##############################################################################
# 7. 运行ASIC仿真
##############################################################################
if [ "$ASIC_COMPILED" = true ]; then
    print_header "7. ASIC仿真测试"

    echo "========== ASIC仿真 ==========" >> $RESULTS_FILE
    ./gemv_asic 2>&1 | tee -a $RESULTS_FILE

    echo "" | tee -a $RESULTS_FILE
fi

##############################################################################
# 8. Lean 4证明检查
##############################################################################
print_header "8. Lean 4证明检查"

if command -v lean &> /dev/null; then
    print_success "Lean 4已安装，运行证明检查..."
    echo "========== Lean 4证明 ==========" >> $RESULTS_FILE

    if lean Quant3p5bit.lean 2>&1 | tee -a $RESULTS_FILE; then
        print_success "Lean 4证明通过"
        echo "Lean 4证明: 通过" >> $RESULTS_FILE
    else
        print_error "Lean 4证明失败"
        echo "Lean 4证明: 失败" >> $RESULTS_FILE
    fi
else
    print_warning "Lean 4未安装（跳过证明检查）"
    echo "Lean 4证明: 跳过（未安装）" >> $RESULTS_FILE
fi

echo "" | tee -a $RESULTS_FILE

##############################################################################
# 9. 生成总结报告
##############################################################################
print_header "9. 测试总结"

echo "========== 测试总结 ==========" | tee -a $RESULTS_FILE
echo "编译结果:" | tee -a $RESULTS_FILE
echo "  v3 (AVX-512):     $( [ "$V3_COMPILED" = true ] && echo "✅" || echo "❌" )" | tee -a $RESULTS_FILE
echo "  MPI (分布式):     $( [ "$MPI_COMPILED" = true ] && echo "✅" || [ "$HAS_MPI" = false ] && echo "⏭️ (跳过)" || echo "❌" )" | tee -a $RESULTS_FILE
echo "  ASIC (仿真):      $( [ "$ASIC_COMPILED" = true ] && echo "✅" || echo "❌" )" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

if [ "$V3_COMPILED" = true ]; then
    echo "✅ 核心功能测试完成" | tee -a $RESULTS_FILE
else
    echo "❌ 核心功能测试失败" | tee -a $RESULTS_FILE
fi

echo "" | tee -a $RESULTS_FILE
echo "完整结果已保存到: $RESULTS_FILE"
echo ""
echo "下一步："
echo "  1. 查看详细结果: cat $RESULTS_FILE"
echo "  2. 如果有编译错误，查看 compile_*.log"
echo "  3. 把结果发给Claude，我会根据性能数据优化v4"
echo ""

print_success "测试脚本完成！"
