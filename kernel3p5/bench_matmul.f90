! CPU micro-benchmark: packed 3.5-bit vs packed INT4 vs MXFP4 matrix-vector (M=1, the decode case).
! Single thread, scalar reference kernels. Measures this machine only (not an accelerator) and says
! nothing about accuracy. Each kernel is timed per repetition; median and minimum are reported
! because single runs vary by tens of percent. Prints one JSON object on stdout.
program bench_matmul
    use iso_fortran_env, only: int8, int32, int64, real64
    use pack3p5
    use mxfp4
    implicit none
    integer, parameter :: M = 1
    integer :: K, N, reps, j, i
    integer(int8), allocatable :: A(:,:), W(:), Wp3(:,:), Wp4(:,:), Wmx(:,:)
    integer(int32), allocatable :: C3(:,:), C4(:,:)
    real(real64), allocatable :: Cmx(:,:), wr(:)
    real(real64) :: s3(2), s4(2), sm(2), u8
    real :: u
    character(len=32) :: arg

    K = 4096; N = 4096; reps = 15
    if (command_argument_count() >= 1) then; call get_command_argument(1, arg); read (arg, *) K; end if
    if (command_argument_count() >= 2) then; call get_command_argument(2, arg); read (arg, *) N; end if
    if (command_argument_count() >= 3) then; call get_command_argument(3, arg); read (arg, *) reps; end if

    allocate (A(M, K), W(K), Wp3((K/16)*7, N), Wp4(K/2, N), Wmx((K/32)*17, N), &
              C3(M, N), C4(M, N), Cmx(M, N), wr(K))
    do j = 1, K
        call random_number(u); A(1, j) = int(floor(u * 256.0) - 128, int8)
    end do
    do j = 1, N
        do i = 1, K
            call random_number(u)
            if (mod(i, 2) == 1) then; W(i) = int(floor(u * 16.0) - 8, int8)
            else;                     W(i) = int(floor(u * 8.0) - 4, int8); end if
            call random_number(u); wr(i) = (real(u, real64) - 0.5_real64) * 4.0_real64
        end do
        call pack_column(W, Wp3(:, j), K)
        call pack4_column(W, Wp4(:, j), K)     ! same values, valid 4-bit range
        call pack_mx_column(wr, Wmx(:, j), K)
    end do

    call matmul_packed3p5(A, Wp3, C3, M, N, K)  ! warm-up and cross-check
    call matmul_packed4(A, Wp4, C4, M, N, K)
    if (any(C3 /= C4)) error stop "3.5-bit and INT4 kernels disagree on identical weights"
    call matmul_mx(A, Wmx, Cmx, M, N, K)
    u8 = sum(Cmx)

    call time_it(1, s3); call time_it(2, s4); call time_it(3, sm)

    print '(A)', '{'
    print '(A,I0,A,I0,A,I0,A)', '  "K": ', K, ', "N": ', N, ', "reps": ', reps, ','
    print '(A,ES10.3,A,ES10.3,A)', '  "median_seconds_per_matvec": {"3p5bit": ', s3(1), ', "int4": ', s4(1), ','
    print '(A,ES10.3,A)', '      "mxfp4": ', sm(1), '},'
    print '(A,ES10.3,A,ES10.3,A,ES10.3,A)', '  "min_seconds_per_matvec": {"3p5bit": ', s3(2), ', "int4": ', s4(2), ', "mxfp4": ', sm(2), '},'
    print '(A,F7.3,A,F7.3,A)', '  "median_time_ratio_over_int4": {"3p5bit": ', s3(1) / s4(1), ', "mxfp4": ', sm(1) / s4(1), '},'
    print '(A,F8.4,A,F8.4,A)', '  "size_ratio_over_int4": {"3p5bit": ', real(size(Wp3), real64) / size(Wp4), &
                                ', "mxfp4": ', real(size(Wmx), real64) / size(Wp4), '},'
    print '(A,I0)', '  "checksum": ', sum(int(C3, int64)) + sum(int(C4, int64)) + nint(u8)
    print '(A)', '}'

contains
    ! stats = [median, min] over `reps` timed calls of kernel `which`.
    subroutine time_it(which, stats)
        integer, intent(in) :: which
        real(real64), intent(out) :: stats(2)
        real(real64) :: t(reps), tmp
        integer(int64) :: t0, t1, rate
        integer :: r, p, q
        call system_clock(count_rate=rate)
        do r = 1, reps
            call system_clock(t0)
            select case (which)
            case (1); call matmul_packed3p5(A, Wp3, C3, M, N, K)
            case (2); call matmul_packed4(A, Wp4, C4, M, N, K)
            case default; call matmul_mx(A, Wmx, Cmx, M, N, K)
            end select
            call system_clock(t1)
            t(r) = real(t1 - t0, real64) / real(rate, real64)
        end do
        do p = 1, reps - 1                              ! selection sort, reps is small
            do q = p + 1, reps
                if (t(q) < t(p)) then; tmp = t(p); t(p) = t(q); t(q) = tmp; end if
            end do
        end do
        stats(1) = t((reps + 1) / 2); stats(2) = t(1)
    end subroutine time_it
end program bench_matmul
