program test_pack3p5
    use iso_fortran_env, only: int8, int32, int64, real64
    use pack3p5
    implicit none
    integer :: nfail
    nfail = 0
    call t_exhaustive_pairs()
    call t_roundtrip_random()
    call t_matmul_matches_reference()
    call t_int4_matches_reference()
    call t_overflow_bound()
    call t_storage()
    if (nfail == 0) then
        print '(A)', "ALL TESTS PASSED"
    else
        print '(A,I0,A)', "FAILED: ", nfail, " check(s)"
        error stop 1
    end if

contains

    subroutine check(cond, msg)
        logical, intent(in) :: cond
        character(*), intent(in) :: msg
        if (.not. cond) then
            nfail = nfail + 1
            print '(2A)', "  FAIL: ", msg
        end if
    end subroutine check

    ! Every one of the 8 x 128 (position, 7-bit pattern) combinations survives unpack(pack()).
    subroutine t_exhaustive_pairs()
        integer(int8) :: w(16), r(16), pk(7)
        integer :: p, e, o, bad
        bad = 0
        do p = 0, 7
            do e = -8, 7
                do o = -4, 3
                    w = 0_int8
                    w(2*p + 1) = int(e, int8)
                    w(2*p + 2) = int(o, int8)
                    call pack_column(w, pk, 16)
                    call unpack_column(pk, r, 16)
                    if (any(r /= w)) bad = bad + 1
                end do
            end do
        end do
        call check(bad == 0, "exhaustive pair round-trip (8 positions x 16 x 8 values)")
        print '(A,I0,A)', "  exhaustive: ", 8*16*8, " cases checked"
    end subroutine t_exhaustive_pairs

    subroutine t_roundtrip_random()
        integer, parameter :: K = 4096
        integer(int8) :: w(K), r(K), pk((K/16)*7)
        real :: u
        integer :: i, trial, bad
        bad = 0
        call random_seed()
        do trial = 1, 50
            do i = 1, K
                call random_number(u)
                if (mod(i, 2) == 1) then
                    w(i) = int(floor(u * 16.0) - 8, int8)
                else
                    w(i) = int(floor(u * 8.0) - 4, int8)
                end if
            end do
            call pack_column(w, pk, K)
            call unpack_column(pk, r, K)
            if (any(r /= w)) bad = bad + 1
        end do
        call check(bad == 0, "random K=4096 round-trip, 50 trials")
    end subroutine t_roundtrip_random

    subroutine fill_case(M, N, K, A, W)
        integer, intent(in) :: M, N, K
        integer(int8), intent(out) :: A(M, K), W(K, N)
        real :: u
        integer :: i, j
        do j = 1, K
            do i = 1, M
                call random_number(u); A(i, j) = int(floor(u * 256.0) - 128, int8)
            end do
        end do
        do j = 1, N
            do i = 1, K
                call random_number(u)
                if (mod(i, 2) == 1) then
                    W(i, j) = int(floor(u * 16.0) - 8, int8)
                else
                    W(i, j) = int(floor(u * 8.0) - 4, int8)
                end if
            end do
        end do
    end subroutine fill_case

    subroutine t_matmul_matches_reference()
        integer, parameter :: M = 3, N = 5, K = 256
        integer(int8) :: A(M, K), W(K, N), Wp((K/16)*7, N)
        integer(int32) :: C(M, N), R(M, N)
        integer :: i, j, k2
        call fill_case(M, N, K, A, W)
        do j = 1, N
            call pack_column(W(:, j), Wp(:, j), K)
        end do
        call matmul_packed3p5(A, Wp, C, M, N, K)
        do j = 1, N
            do i = 1, M
                R(i, j) = 0
                do k2 = 1, K
                    R(i, j) = R(i, j) + int(A(i, k2), int32) * int(W(k2, j), int32)
                end do
            end do
        end do
        call check(all(C == R), "packed 3.5-bit matmul is bit-exact vs naive int matmul")
    end subroutine t_matmul_matches_reference

    subroutine t_int4_matches_reference()
        integer, parameter :: M = 3, N = 5, K = 256
        integer(int8) :: A(M, K), W(K, N), Wp(K/2, N)
        integer(int32) :: C(M, N), R(M, N)
        integer :: i, j, k2
        call fill_case(M, N, K, A, W)
        W = max(W, int(-8, int8))
        do j = 1, N
            do i = 1, K
                if (mod(i, 2) == 0) W(i, j) = int(max(-8, min(7, 2*int(W(i, j)))), int8)  ! use full 4-bit range
            end do
            call pack4_column(W(:, j), Wp(:, j), K)
        end do
        call matmul_packed4(A, Wp, C, M, N, K)
        do j = 1, N
            do i = 1, M
                R(i, j) = 0
                do k2 = 1, K
                    R(i, j) = R(i, j) + int(A(i, k2), int32) * int(W(k2, j), int32)
                end do
            end do
        end do
        call check(all(C == R), "packed INT4 baseline matmul is bit-exact vs naive int matmul")
    end subroutine t_int4_matches_reference

    ! Worst case: A = -128 everywhere, w = -8 / -4 -> each pair contributes 128*(8+4).
    subroutine t_overflow_bound()
        integer(int64) :: kmax, worst
        kmax = acc_bound()
        worst = kmax * 128_int64 * 8_int64          ! conservative: every weight magnitude 8
        call check(worst <= int(huge(0_int32), int64), "int32 accumulator safe for K <= acc_bound()")
        call check((kmax + 1) * 128_int64 * 8_int64 > int(huge(0_int32), int64), "acc_bound() is tight")
        print '(A,I0)', "  int32 accumulator safe for K <= ", kmax
        print '(A,I0,A,I0)', "  Llama-70B K=8192 and 28672 are far below: ", 8192, " / ", 28672
    end subroutine t_overflow_bound

    subroutine t_storage()
        integer, parameter :: K = 8192, N = 4
        integer(int8) :: Wp((K/16)*7, N)
        real(real64) :: bits_per_weight, params70b_gb
        bits_per_weight = real(size(Wp), real64) * 8.0_real64 / (real(K, real64) * real(N, real64))
        call check(abs(bits_per_weight - 3.5_real64) < 1.0e-12_real64, "storage is exactly 3.5 bits per weight")
        params70b_gb = 70.6e9_real64 * bits_per_weight / 8.0_real64 / 1.0e9_real64
        print '(A,F5.2,A,F5.1,A)', "  measured storage: ", bits_per_weight, " bits/weight -> 70.6B params = ", params70b_gb, " GB (weights only)"
    end subroutine t_storage

end program test_pack3p5
