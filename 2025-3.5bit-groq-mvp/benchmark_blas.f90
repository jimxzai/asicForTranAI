! Comprehensive BLAS Performance Benchmark
! Compares: Baseline -> SIMD+OpenMP -> BLAS
! Target: Validate 50-100× speedup with OpenBLAS SGEMM

program benchmark_blas
    use iso_fortran_env, only: int8, int32, real32, int64
    use matmul_int4_groq, only: matmul_int4_awq, dequantize_output
    use matmul_simd_optimized, only: matmul_int4_simd, dequantize_output_simd
    use matmul_blas_optimized, only: matmul_int4_blas, matmul_int4_blas_cached
    implicit none

    ! Benchmark configuration
    integer(int32), parameter :: M = 1         ! Batch size (single token)
    integer(int32), parameter :: K_dim = 8192  ! Hidden dimension
    integer(int32), parameter :: N = 8192      ! Output dimension
    integer(int32), parameter :: NUM_ITERS = 100  ! Iterations for timing

    ! Test data
    integer(int8), allocatable :: A(:,:)           ! Activations [M, K]
    integer(int8), allocatable :: W_Q(:,:)         ! Weights [K/2, N]
    real(real32), allocatable :: W_scales(:)       ! Scales [N]
    real(real32), allocatable :: W_fp32(:,:)       ! Dequantized weights [K, N]

    ! Output buffers
    integer(int32), allocatable :: C_baseline(:,:)
    integer(int32), allocatable :: C_simd(:,:)
    real(real32), allocatable :: Out_baseline(:,:)
    real(real32), allocatable :: Out_simd(:,:)
    real(real32), allocatable :: Out_blas(:,:)
    real(real32), allocatable :: Out_blas_cached(:,:)

    ! Timing
    integer(int64) :: t_start, t_end, clock_rate
    real(real32) :: time_baseline, time_simd, time_blas, time_blas_cached
    real(real32) :: speedup_simd, speedup_blas, speedup_blas_cached

    integer(int32) :: iter, i, j, k
    real(real32) :: diff, max_diff

    print *, "=========================================="
    print *, "BLAS Performance Benchmark"
    print *, "=========================================="
    print *, "Matrix dimensions:"
    print *, "  M (batch) =", M
    print *, "  K (hidden) =", K_dim
    print *, "  N (output) =", N
    print *, "  Iterations =", NUM_ITERS
    print *, ""
    print *, "Comparing implementations:"
    print *, "  1. Baseline (matmul_int4_groq)"
    print *, "  2. SIMD+OpenMP (matmul_simd_optimized)"
    print *, "  3. BLAS (matmul_blas_optimized)"
    print *, "  4. BLAS+Cached Weights (best for inference)"
    print *, ""

    ! Allocate arrays
    allocate(A(M, K_dim))
    allocate(W_Q(K_dim/2, N))
    allocate(W_scales(N))
    allocate(W_fp32(K_dim, N))
    allocate(C_baseline(M, N))
    allocate(C_simd(M, N))
    allocate(Out_baseline(M, N))
    allocate(Out_simd(M, N))
    allocate(Out_blas(M, N))
    allocate(Out_blas_cached(M, N))

    ! Initialize random test data
    call random_seed()
    call random_number(W_scales)
    W_scales = W_scales * 0.01  ! Typical scale values

    do i = 1, M
        do k = 1, K_dim
            A(i, k) = int(mod(i * k, 127), int8)
        end do
    end do

    do j = 1, N
        do k = 1, K_dim/2
            W_Q(k, j) = int(mod(j * k, 127), int8)
        end do
    end do

    print *, "✓ Test data initialized"
    print *, ""

    ! =========================================================================
    ! Benchmark 1: Baseline (matmul_int4_groq)
    ! =========================================================================
    print *, "[1/4] Benchmarking BASELINE implementation..."
    call system_clock(count_rate=clock_rate)
    call system_clock(t_start)

    do iter = 1, NUM_ITERS
        call matmul_int4_awq(A, W_Q, W_scales, C_baseline, M, N, K_dim)
        call dequantize_output(C_baseline, W_scales, Out_baseline, M, N)
    end do

    call system_clock(t_end)
    time_baseline = real(t_end - t_start, real32) / real(clock_rate, real32)
    time_baseline = time_baseline / NUM_ITERS

    print *, "  Time per iteration:", time_baseline * 1000.0, "ms"
    print *, ""

    ! =========================================================================
    ! Benchmark 2: SIMD + OpenMP
    ! =========================================================================
    print *, "[2/4] Benchmarking SIMD+OpenMP implementation..."
    call system_clock(t_start)

    do iter = 1, NUM_ITERS
        call matmul_int4_simd(A, W_Q, W_scales, C_simd, M, N, K_dim)
        call dequantize_output_simd(C_simd, W_scales, Out_simd, M, N)
    end do

    call system_clock(t_end)
    time_simd = real(t_end - t_start, real32) / real(clock_rate, real32)
    time_simd = time_simd / NUM_ITERS

    speedup_simd = time_baseline / time_simd

    print *, "  Time per iteration:", time_simd * 1000.0, "ms"
    print *, "  Speedup vs baseline:", speedup_simd, "×"

    ! Validate correctness
    max_diff = 0.0
    do j = 1, N
        do i = 1, M
            diff = abs(Out_simd(i,j) - Out_baseline(i,j))
            if (diff > max_diff) max_diff = diff
        end do
    end do
    print *, "  Max difference vs baseline:", max_diff
    if (max_diff < 1.0) then
        print *, "  ✓ Correctness validated"
    else
        print *, "  ✗ WARNING: Large difference detected!"
    end if
    print *, ""

    ! =========================================================================
    ! Benchmark 3: BLAS (on-the-fly dequantization)
    ! =========================================================================
    print *, "[3/4] Benchmarking BLAS (with on-the-fly dequantization)..."
    call system_clock(t_start)

    do iter = 1, NUM_ITERS
        call matmul_int4_blas(A, W_Q, W_scales, Out_blas, M, N, K_dim)
    end do

    call system_clock(t_end)
    time_blas = real(t_end - t_start, real32) / real(clock_rate, real32)
    time_blas = time_blas / NUM_ITERS

    speedup_blas = time_baseline / time_blas

    print *, "  Time per iteration:", time_blas * 1000.0, "ms"
    print *, "  Speedup vs baseline:", speedup_blas, "×"

    ! Validate correctness
    max_diff = 0.0
    do j = 1, N
        do i = 1, M
            diff = abs(Out_blas(i,j) - Out_baseline(i,j))
            if (diff > max_diff) max_diff = diff
        end do
    end do
    print *, "  Max difference vs baseline:", max_diff
    if (max_diff < 1.0) then
        print *, "  ✓ Correctness validated"
    else
        print *, "  ✗ WARNING: Large difference detected!"
    end if
    print *, ""

    ! =========================================================================
    ! Benchmark 4: BLAS with cached weights (BEST for inference)
    ! =========================================================================
    print *, "[4/4] Benchmarking BLAS with CACHED WEIGHTS (inference mode)..."

    ! Dequantize weights once (this happens at model load time)
    print *, "  Pre-dequantizing weights..."
    call dequantize_weights_from_module(W_Q, W_scales, W_fp32, N, K_dim)

    call system_clock(t_start)

    do iter = 1, NUM_ITERS
        call matmul_int4_blas_cached(A, W_fp32, Out_blas_cached, M, N, K_dim)
    end do

    call system_clock(t_end)
    time_blas_cached = real(t_end - t_start, real32) / real(clock_rate, real32)
    time_blas_cached = time_blas_cached / NUM_ITERS

    speedup_blas_cached = time_baseline / time_blas_cached

    print *, "  Time per iteration:", time_blas_cached * 1000.0, "ms"
    print *, "  Speedup vs baseline:", speedup_blas_cached, "×"

    ! Validate correctness
    max_diff = 0.0
    do j = 1, N
        do i = 1, M
            diff = abs(Out_blas_cached(i,j) - Out_baseline(i,j))
            if (diff > max_diff) max_diff = diff
        end do
    end do
    print *, "  Max difference vs baseline:", max_diff
    if (max_diff < 1.0) then
        print *, "  ✓ Correctness validated"
    else
        print *, "  ✗ WARNING: Large difference detected!"
    end if
    print *, ""

    ! =========================================================================
    ! Summary
    ! =========================================================================
    print *, "=========================================="
    print *, "PERFORMANCE SUMMARY"
    print *, "=========================================="
    print *, ""
    print *, "Time per iteration (ms):"
    print *, "  Baseline:           ", time_baseline * 1000.0
    print *, "  SIMD+OpenMP:        ", time_simd * 1000.0
    print *, "  BLAS:               ", time_blas * 1000.0
    print *, "  BLAS (cached):      ", time_blas_cached * 1000.0
    print *, ""
    print *, "Speedup vs Baseline:"
    print *, "  SIMD+OpenMP:        ", speedup_simd, "×"
    print *, "  BLAS:               ", speedup_blas, "×"
    print *, "  BLAS (cached):      ", speedup_blas_cached, "×"
    print *, ""
    print *, "Throughput (tokens/second, for 70B model):"
    print *, "  Baseline:           ", 1.0 / (time_baseline * 80), "tok/s"
    print *, "  SIMD+OpenMP:        ", 1.0 / (time_simd * 80), "tok/s"
    print *, "  BLAS:               ", 1.0 / (time_blas * 80), "tok/s"
    print *, "  BLAS (cached):      ", 1.0 / (time_blas_cached * 80), "tok/s"
    print *, ""
    print *, "Recommendation for inference: Use BLAS with cached weights"
    print *, "Expected speedup on 8-core CPU: 50-100× vs baseline"
    print *, ""
    print *, "=========================================="

    ! Cleanup
    deallocate(A, W_Q, W_scales, W_fp32)
    deallocate(C_baseline, C_simd)
    deallocate(Out_baseline, Out_simd, Out_blas, Out_blas_cached)

contains

    ! Helper to call the dequantization routine
    subroutine dequantize_weights_from_module(W_Q, W_scales, W_fp32, N, K_dim)
        use matmul_blas_optimized, only: dequantize_weights_int4
        integer(int8), intent(in) :: W_Q(:,:)
        real(real32), intent(in) :: W_scales(:)
        real(real32), intent(out) :: W_fp32(:,:)
        integer(int32), intent(in) :: N, K_dim

        call dequantize_weights_int4(W_Q, W_scales, W_fp32, N, K_dim)
    end subroutine

end program benchmark_blas
