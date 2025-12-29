! GPU Performance Benchmark for RTX 2080 Ti
! Tests cuBLAS acceleration vs CPU implementations

program benchmark_gpu
    use iso_fortran_env, only: int8, int32, real32, int64
    use matmul_int4_groq, only: matmul_int4_awq, dequantize_output
    use matmul_simd_optimized, only: matmul_int4_simd, dequantize_output_simd
    use matmul_cublas
    implicit none

    ! Benchmark configuration
    integer(int32), parameter :: M = 1         ! Batch size (single token)
    integer(int32), parameter :: K_dim = 8192  ! Hidden dimension
    integer(int32), parameter :: N = 8192      ! Output dimension
    integer(int32), parameter :: NUM_ITERS = 10  ! Fewer iterations for GPU

    ! Test data
    integer(int8), allocatable :: A(:,:)
    integer(int8), allocatable :: W_Q(:,:)
    real(real32), allocatable :: W_scales(:)

    ! Output buffers
    integer(int32), allocatable :: C_baseline(:,:)
    integer(int32), allocatable :: C_simd(:,:)
    real(real32), allocatable :: Out_baseline(:,:)
    real(real32), allocatable :: Out_simd(:,:)
    real(real32), allocatable :: Out_cublas(:,:)
    real(real32), allocatable :: Out_cublas_cached(:,:)

    ! Timing
    integer(int64) :: t_start, t_end, clock_rate
    real(real32) :: time_baseline, time_simd, time_cublas, time_cublas_cached
    real(real32) :: speedup_simd, speedup_cublas, speedup_cublas_cached

    integer(int32) :: iter, i, j, k
    real(real32) :: diff, max_diff

    print *, "=========================================="
    print *, "GPU Performance Benchmark (RTX 2080 Ti)"
    print *, "=========================================="
    print *, "Matrix dimensions:"
    print *, "  M (batch) =", M
    print *, "  K (hidden) =", K_dim
    print *, "  N (output) =", N
    print *, "  Iterations =", NUM_ITERS
    print *, ""
    print *, "Comparing implementations:"
    print *, "  1. CPU Baseline (matmul_int4_groq)"
    print *, "  2. CPU SIMD+OpenMP (current best)"
    print *, "  3. GPU cuBLAS (on-the-fly dequant)"
    print *, "  4. GPU cuBLAS (cached weights)"
    print *, ""

    ! Allocate arrays
    allocate(A(M, K_dim))
    allocate(W_Q(K_dim/2, N))
    allocate(W_scales(N))
    allocate(C_baseline(M, N))
    allocate(C_simd(M, N))
    allocate(Out_baseline(M, N))
    allocate(Out_simd(M, N))
    allocate(Out_cublas(M, N))
    allocate(Out_cublas_cached(M, N))

    ! Initialize random test data
    call random_seed()
    call random_number(W_scales)
    W_scales = W_scales * 0.01

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

    ! Initialize cuBLAS
    call cublas_init()
    call allocate_gpu_memory(M, N, K_dim)
    print *, ""

    ! =========================================================================
    ! Benchmark 1: CPU Baseline
    ! =========================================================================
    print *, "[1/4] Benchmarking CPU Baseline..."
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
    ! Benchmark 2: CPU SIMD+OpenMP
    ! =========================================================================
    print *, "[2/4] Benchmarking CPU SIMD+OpenMP..."
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
    print *, ""

    ! =========================================================================
    ! Benchmark 3: GPU cuBLAS (on-the-fly dequantization)
    ! =========================================================================
    print *, "[3/4] Benchmarking GPU cuBLAS..."
    call system_clock(t_start)

    do iter = 1, NUM_ITERS
        call matmul_int4_cublas(A, W_Q, W_scales, Out_cublas, M, N, K_dim)
    end do

    call system_clock(t_end)
    time_cublas = real(t_end - t_start, real32) / real(clock_rate, real32)
    time_cublas = time_cublas / NUM_ITERS

    speedup_cublas = time_baseline / time_cublas

    print *, "  Time per iteration:", time_cublas * 1000.0, "ms"
    print *, "  Speedup vs baseline:", speedup_cublas, "×"

    ! Validate correctness
    max_diff = 0.0
    do j = 1, N
        do i = 1, M
            diff = abs(Out_cublas(i,j) - Out_baseline(i,j))
            if (diff > max_diff) max_diff = diff
        end do
    end do
    print *, "  Max difference vs baseline:", max_diff
    if (max_diff < 1.0) then
        print *, "  ✓ Correctness validated"
    else
        print *, "  ✗ WARNING: Large difference!"
    end if
    print *, ""

    ! =========================================================================
    ! Benchmark 4: GPU cuBLAS (cached weights)
    ! =========================================================================
    print *, "[4/4] Benchmarking GPU cuBLAS (cached)..."

    ! Pre-dequantize weights to GPU
    call dequantize_weights_gpu(W_Q, W_scales, N, K_dim)

    call system_clock(t_start)

    do iter = 1, NUM_ITERS
        call matmul_int4_cublas_cached(A, Out_cublas_cached, M, N, K_dim)
    end do

    call system_clock(t_end)
    time_cublas_cached = real(t_end - t_start, real32) / real(clock_rate, real32)
    time_cublas_cached = time_cublas_cached / NUM_ITERS

    speedup_cublas_cached = time_baseline / time_cublas_cached

    print *, "  Time per iteration:", time_cublas_cached * 1000.0, "ms"
    print *, "  Speedup vs baseline:", speedup_cublas_cached, "×"

    ! Validate
    max_diff = 0.0
    do j = 1, N
        do i = 1, M
            diff = abs(Out_cublas_cached(i,j) - Out_baseline(i,j))
            if (diff > max_diff) max_diff = diff
        end do
    end do
    print *, "  Max difference vs baseline:", max_diff
    if (max_diff < 1.0) then
        print *, "  ✓ Correctness validated"
    else
        print *, "  ✗ WARNING: Large difference!"
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
    print *, "  CPU Baseline:       ", time_baseline * 1000.0
    print *, "  CPU SIMD+OpenMP:    ", time_simd * 1000.0
    print *, "  GPU cuBLAS:         ", time_cublas * 1000.0
    print *, "  GPU cuBLAS (cached):", time_cublas_cached * 1000.0
    print *, ""
    print *, "Speedup vs CPU Baseline:"
    print *, "  CPU SIMD+OpenMP:    ", speedup_simd, "×"
    print *, "  GPU cuBLAS:         ", speedup_cublas, "×"
    print *, "  GPU cuBLAS (cached):", speedup_cublas_cached, "×"
    print *, ""
    print *, "Throughput (tokens/sec, 70B model):"
    print *, "  CPU Baseline:       ", 1.0 / (time_baseline * 80), "tok/s"
    print *, "  CPU SIMD+OpenMP:    ", 1.0 / (time_simd * 80), "tok/s"
    print *, "  GPU cuBLAS:         ", 1.0 / (time_cublas * 80), "tok/s"
    print *, "  GPU cuBLAS (cached):", 1.0 / (time_cublas_cached * 80), "tok/s"
    print *, ""

    if (speedup_cublas_cached > speedup_simd) then
        print *, "🚀 GPU cuBLAS wins! Use for production inference."
    else
        print *, "⚡ CPU SIMD wins for M=1. GPU better for batched (M>1)."
    end if
    print *, ""
    print *, "=========================================="

    ! Cleanup
    call cublas_shutdown()
    deallocate(A, W_Q, W_scales)
    deallocate(C_baseline, C_simd)
    deallocate(Out_baseline, Out_simd, Out_cublas, Out_cublas_cached)

end program benchmark_gpu
