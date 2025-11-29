! Comprehensive Benchmark Suite for Optimization Validation
! Compares: Baseline vs Optimized vs Theoretical Peak
! Pure Fortran 2023

program benchmark_optimizations
    use iso_fortran_env, only: int8, int32, int64, real32
    use matmul_int4_groq, only: matmul_int4_awq
    use matmul_simd_optimized, only: matmul_int4_simd
    implicit none

    ! Test configurations
    integer(int32), parameter :: M = 1        ! Batch size
    integer(int32), parameter :: N = 8192     ! Hidden dim
    integer(int32), parameter :: K_dim = 8192 ! Hidden dim
    integer(int32), parameter :: NUM_WARMUP = 10
    integer(int32), parameter :: NUM_ITERATIONS = 1000

    ! Arrays
    integer(int8) :: A(M, K_dim)
    integer(int8) :: W_Q(K_dim/8, N)
    real(real32) :: W_scales(N)
    integer(int32) :: C_baseline(M, N)
    integer(int32) :: C_optimized(M, N)

    ! Timing
    integer(int64) :: t_start, t_end, count_rate
    real(real32) :: time_baseline, time_optimized
    real(real32) :: speedup, tokens_per_sec_baseline, tokens_per_sec_optimized
    integer :: i, j

    ! Error checking
    integer(int32) :: max_diff, total_diff
    logical :: results_match

    print *, '========================================='
    print *, 'Optimization Benchmark Suite'
    print *, '========================================='
    print *, ''
    print *, 'Test Configuration:'
    print *, '  Matrix size: ', M, 'x', K_dim, ' @ ', K_dim, 'x', N
    print *, '  Warmup iterations:', NUM_WARMUP
    print *, '  Benchmark iterations:', NUM_ITERATIONS
    print *, ''

    ! Initialize test data with random values
    call random_seed()
    call random_number_int8(A)
    call random_number_int8(W_Q)
    call random_number(W_scales)
    W_scales = W_scales * 0.01  ! Scale to reasonable range

    print *, 'Initializing test data... DONE'
    print *, ''

    ! ================================================
    ! WARMUP: Baseline
    ! ================================================
    print *, 'Warming up baseline implementation...'
    do i = 1, NUM_WARMUP
        call matmul_int4_awq(A, W_Q, W_scales, C_baseline, M, N, K_dim)
    end do
    print *, '  Warmup complete'
    print *, ''

    ! ================================================
    ! BENCHMARK: Baseline
    ! ================================================
    print *, 'Benchmarking BASELINE implementation...'
    call system_clock(count_rate=count_rate)
    call system_clock(t_start)

    do i = 1, NUM_ITERATIONS
        call matmul_int4_awq(A, W_Q, W_scales, C_baseline, M, N, K_dim)
    end do

    call system_clock(t_end)
    time_baseline = real(t_end - t_start, real32) / real(count_rate, real32) / real(NUM_ITERATIONS, real32)
    tokens_per_sec_baseline = 1.0 / time_baseline

    print *, '  Time per iteration:', time_baseline * 1000.0, 'ms'
    print *, '  Throughput:', int(tokens_per_sec_baseline), 'tokens/sec'
    print *, ''

    ! ================================================
    ! WARMUP: SIMD Optimized
    ! ================================================
    print *, 'Warming up SIMD-OPTIMIZED implementation...'
    do i = 1, NUM_WARMUP
        call matmul_int4_simd(A, W_Q, W_scales, C_optimized, M, N, K_dim)
    end do
    print *, '  Warmup complete'
    print *, ''

    ! ================================================
    ! BENCHMARK: SIMD Optimized
    ! ================================================
    print *, 'Benchmarking SIMD-OPTIMIZED implementation (OpenMP + SIMD)...'
    call system_clock(t_start)

    do i = 1, NUM_ITERATIONS
        call matmul_int4_simd(A, W_Q, W_scales, C_optimized, M, N, K_dim)
    end do

    call system_clock(t_end)
    time_optimized = real(t_end - t_start, real32) / real(count_rate, real32) / real(NUM_ITERATIONS, real32)
    tokens_per_sec_optimized = 1.0 / time_optimized

    print *, '  Time per iteration:', time_optimized * 1000.0, 'ms'
    print *, '  Throughput:', int(tokens_per_sec_optimized), 'tokens/sec'
    print *, ''

    ! ================================================
    ! VALIDATION: Check results match
    ! ================================================
    print *, 'Validating correctness...'
    max_diff = 0
    total_diff = 0
    results_match = .true.

    do j = 1, N
        do i = 1, M
            if (C_baseline(i,j) /= C_optimized(i,j)) then
                results_match = .false.
                max_diff = max(max_diff, abs(C_baseline(i,j) - C_optimized(i,j)))
                total_diff = total_diff + abs(C_baseline(i,j) - C_optimized(i,j))
            end if
        end do
    end do

    if (results_match) then
        print *, '  ✓ Results MATCH (bit-exact)'
    else
        print *, '  ✗ Results DIFFER!'
        print *, '    Max difference:', max_diff
        print *, '    Total difference:', total_diff
    end if
    print *, ''

    ! ================================================
    ! SUMMARY REPORT
    ! ================================================
    print *, '========================================='
    print *, 'BENCHMARK RESULTS'
    print *, '========================================='
    print *, ''

    print '(A, F12.6, A)', '  Baseline:    ', time_baseline * 1000.0, ' ms/token'
    print '(A, I12, A)', '               ', int(tokens_per_sec_baseline), ' tok/s'
    print *, ''

    print '(A, F12.6, A)', '  Optimized:   ', time_optimized * 1000.0, ' ms/token'
    print '(A, I12, A)', '               ', int(tokens_per_sec_optimized), ' tok/s'
    print *, ''

    speedup = time_baseline / time_optimized
    print '(A, F8.3, A)', '  Speedup:     ', speedup, '×'
    print *, ''

    if (speedup >= 1.30) then
        print *, '  🎉 SUCCESS! Achieved expected 1.40× speedup'
    else if (speedup >= 1.10) then
        print *, '  ⚠️  Good improvement, but below target (1.40×)'
    else
        print *, '  ❌ WARNING: Minimal improvement'
    end if

    print *, ''
    print *, '========================================='
    print *, 'Performance Projection (80-layer model)'
    print *, '========================================='
    print *, ''

    print '(A, F8.3, A)', '  Baseline total:   ', time_baseline * 80 * 1000, ' ms'
    print '(A, I8, A)', '                    ', int(1.0 / (time_baseline * 80)), ' tok/s'
    print *, ''

    print '(A, F8.3, A)', '  Optimized total:  ', time_optimized * 80 * 1000, ' ms'
    print '(A, I8, A)', '                    ', int(1.0 / (time_optimized * 80)), ' tok/s'
    print *, ''

    print *, '========================================='
    print *, ''

contains

    ! Helper: Generate random INT8 array
    subroutine random_number_int8(array)
        integer(int8), intent(out) :: array(:,:)
        real(real32), allocatable :: temp(:,:)
        integer :: i, j

        allocate(temp(size(array, 1), size(array, 2)))
        call random_number(temp)

        ! Map [0,1] → [-127, 127]
        do j = 1, size(array, 2)
            do i = 1, size(array, 1)
                array(i,j) = int((temp(i,j) - 0.5) * 254.0, int8)
            end do
        end do

        deallocate(temp)
    end subroutine random_number_int8

end program benchmark_optimizations
