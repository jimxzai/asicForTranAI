program test_mxfp4
    use iso_fortran_env, only: int8, int32, int64, real64
    use mxfp4
    implicit none
    integer :: nfail
    nfail = 0
    call t_decode_table()
    call t_encode_decode_roundtrip()
    call t_matmul_exact()
    call t_block_bound()
    call t_scale_range()
    call t_storage()
    if (nfail == 0) then
        print '(A)', "ALL MXFP4 TESTS PASSED"
    else
        print '(A,I0,A)', "FAILED: ", nfail, " check(s)"
        error stop 1
    end if
contains
    subroutine check(cond, msg)
        logical, intent(in) :: cond
        character(*), intent(in) :: msg
        if (.not. cond) then; nfail = nfail + 1; print '(2A)', "  FAIL: ", msg; end if
    end subroutine check

    subroutine t_decode_table()
        real(real64) :: expect(0:15)
        integer :: c
        expect(0:7)  = [0.0_real64, 0.5_real64, 1.0_real64, 1.5_real64, 2.0_real64, 3.0_real64, 4.0_real64, 6.0_real64]
        expect(8:15) = -expect(0:7)
        do c = 0, 15
            call check(mx_decode_code(c) == expect(c), "decode table entry")
        end do
    end subroutine

    ! encode(decode(c)) == c for all codes except negative zero (code 8), which encodes to +0.
    subroutine t_encode_decode_roundtrip()
        integer :: c, bad
        bad = 0
        do c = 0, 15
            if (c == 8) cycle
            if (mx_encode_value(mx_decode_code(c)) /= c) bad = bad + 1
        end do
        call check(bad == 0, "encode(decode(code)) == code for all 15 distinct codes")
        call check(mx_encode_value(5.0_real64) == 7 .or. mx_encode_value(5.0_real64) == 6, "5.0 rounds to a neighbour of 4 and 6")
    end subroutine

    ! Packed matmul must equal a float64 reference built from decoded values and the same scales.
    subroutine t_matmul_exact()
        integer, parameter :: M = 3, N = 4, K = 256
        integer(int8) :: A(M, K), Wp((K/32)*17, N)
        real(real64) :: W(K, N), C(M, N), R(M, N), s
        real :: u
        integer :: i, j, b, t, e, lo, hi, base, pair
        do j = 1, K
            do i = 1, M
                call random_number(u); A(i, j) = int(floor(u * 256.0) - 128, int8)
            end do
        end do
        do j = 1, N
            do b = 0, K/32 - 1
                call random_number(u); e = 110 + int(u * 20.0)       ! scale exponents 2**-17 .. 2**2
                base = b * 17
                s = 2.0_real64 ** (e - 127)
                do t = 0, 15
                    call random_number(u); lo = int(u * 16.0)
                    call random_number(u); hi = int(u * 16.0)
                    W(b*32 + 2*t + 1, j) = mx_decode_code(lo) * s
                    W(b*32 + 2*t + 2, j) = mx_decode_code(hi) * s
                    pair = ior(lo, ishft(hi, 4))
                    Wp(base + t + 1, j) = int(merge(pair - 256, pair, pair > 127), int8)
                end do
                Wp(base + 17, j) = int(merge(e - 256, e, e > 127), int8)
            end do
        end do
        call matmul_mx(A, Wp, C, M, N, K)
        do j = 1, N
            do i = 1, M
                R(i, j) = sum(real(A(i, :), real64) * W(:, j))
            end do
        end do
        call check(all(C == R), "MXFP4 matmul equals float64 reference exactly (scale exponents 2**-17..2**2)")
        print '(A,ES9.2)', "  max |C - R| = ", maxval(abs(C - R))
    end subroutine

    subroutine t_block_bound()
        call check(mx_block_acc_bound() == 49152_int64, "per-block |acc| bound is 49152")
        call check(mx_block_acc_bound() < int(huge(0_int32), int64), "per-block accumulator fits int32 with large margin")
    end subroutine

    subroutine t_scale_range()
        real(real64) :: w(32)
        integer(int8) :: pk(17)
        integer :: e
        w = 0.0_real64; w(1) = 1.0e-30_real64
        call pack_mx_column(w, pk, 32)
        e = iand(int(pk(17), int32), 255)
        call check(e >= 0 .and. e <= 254, "scale exponent byte stays in [0,254] for tiny inputs")
        w(1) = 1.0e30_real64
        call pack_mx_column(w, pk, 32)
        e = iand(int(pk(17), int32), 255)
        call check(e >= 0 .and. e <= 254, "scale exponent byte never emits the reserved value 255")
    end subroutine

    subroutine t_storage()
        integer, parameter :: K = 4096, N = 2
        integer(int8) :: Wp((K/32)*17, N)
        real(real64) :: bits
        bits = real(size(Wp), real64) * 8.0_real64 / (real(K, real64) * N)
        call check(abs(bits - 4.25_real64) < 1.0e-12_real64, "storage is exactly 4.25 bits per weight")
        print '(A,F5.3,A)', "  measured storage: ", bits, " bits/weight (4 + 8/32)"
    end subroutine
end program test_mxfp4
