! CPU micro-benchmark: packed 3.5-bit vs packed INT4 matrix-vector (M=1, the decode case).
! Single thread, scalar reference kernels. This measures this machine, not an accelerator,
! and says nothing about accuracy. Prints one JSON object on stdout.
program bench_matmul
    use iso_fortran_env, only: int8, int32, int64, real64
    use pack3p5
    implicit none
    integer, parameter :: M = 1
    integer :: K, N, reps, r, j, i
    integer(int8), allocatable :: A(:,:), W(:), Wp3(:,:), Wp4(:,:)
    integer(int32), allocatable :: C3(:,:), C4(:,:)
    integer(int64) :: t0, t1, rate, chk
    real(real64) :: t3, t4, bytes3, bytes4
    real :: u
    character(len=32) :: arg

    K = 4096; N = 4096; reps = 20
    if (command_argument_count() >= 1) then; call get_command_argument(1, arg); read (arg, *) K; end if
    if (command_argument_count() >= 2) then; call get_command_argument(2, arg); read (arg, *) N; end if
    if (command_argument_count() >= 3) then; call get_command_argument(3, arg); read (arg, *) reps; end if

    allocate (A(M, K), W(K), Wp3((K/16)*7, N), Wp4(K/2, N), C3(M, N), C4(M, N))
    do j = 1, K
        call random_number(u); A(1, j) = int(floor(u * 256.0) - 128, int8)
    end do
    do j = 1, N
        do i = 1, K
            call random_number(u)
            if (mod(i, 2) == 1) then; W(i) = int(floor(u * 16.0) - 8, int8)
            else;                     W(i) = int(floor(u * 8.0) - 4, int8); end if
        end do
        call pack_column(W, Wp3(:, j), K)
        call pack4_column(W, Wp4(:, j), K)     ! same values, valid 4-bit range
    end do

    call matmul_packed3p5(A, Wp3, C3, M, N, K)  ! warm-up
    call matmul_packed4(A, Wp4, C4, M, N, K)
    if (any(C3 /= C4)) error stop "3.5-bit and INT4 kernels disagree on identical weights"

    call system_clock(count_rate=rate)
    call system_clock(t0)
    do r = 1, reps; call matmul_packed3p5(A, Wp3, C3, M, N, K); end do
    call system_clock(t1); t3 = real(t1 - t0, real64) / real(rate, real64) / reps
    call system_clock(t0)
    do r = 1, reps; call matmul_packed4(A, Wp4, C4, M, N, K); end do
    call system_clock(t1); t4 = real(t1 - t0, real64) / real(rate, real64) / reps
    chk = sum(int(C3, int64)) + sum(int(C4, int64))

    bytes3 = real(size(Wp3), real64); bytes4 = real(size(Wp4), real64)
    print '(A)', '{'
    print '(A,I0,A,I0,A,I0,A)', '  "K": ', K, ', "N": ', N, ', "reps": ', reps, ','
    print '(A,ES12.4,A,ES12.4,A)', '  "seconds_per_matvec_3p5bit": ', t3, ', "seconds_per_matvec_int4": ', t4, ','
    print '(A,F8.3,A)', '  "ratio_3p5bit_over_int4_time": ', t3 / t4, ','
    print '(A,F10.2,A,F10.2,A)', '  "weight_MB_3p5bit": ', bytes3/1.0e6_real64, ', "weight_MB_int4": ', bytes4/1.0e6_real64, ','
    print '(A,F8.4,A)', '  "size_ratio_3p5bit_over_int4": ', bytes3 / bytes4, ','
    print '(A,I0)', '  "checksum": ', chk
    print '(A)', '}'
end program bench_matmul
