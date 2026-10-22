! Reference 3.5-bit weight packing and matmul.
!
! Format: weights come in pairs. The even-indexed weight of a pair is a 4-bit
! two's-complement integer in [-8, 7]; the odd-indexed one is 3-bit in [-4, 3].
! A pair is 7 bits, so 3.5 bits per weight on average. Eight pairs (16 weights)
! fill exactly 56 bits = 7 bytes, so storage really is 3.5 bits per weight with
! no padding. Pair p of a group sits at bit offset 7*p of a little-endian
! 56-bit word. K must be a multiple of 16.
module pack3p5
    use iso_fortran_env, only: int8, int32, int64
    implicit none
    private
    public :: GROUP_W, GROUP_B, pack_column, unpack_column, matmul_packed3p5, &
              matmul_packed4, pack4_column, acc_bound
    integer, parameter :: GROUP_W = 16   ! weights per group
    integer, parameter :: GROUP_B = 7    ! bytes per group

contains

    pure function sext(x, nbits) result(v)
        integer(int32), intent(in) :: x
        integer, intent(in) :: nbits
        integer(int32) :: v
        v = iand(x, ishft(1_int32, nbits) - 1_int32)
        if (v >= ishft(1_int32, nbits - 1)) v = v - ishft(1_int32, nbits)
    end function sext

    ! w: K integer weights, even positions (1,3,..) in [-8,7], odd (2,4,..) in [-4,3].
    ! packed: (K/16)*7 bytes.
    pure subroutine pack_column(w, packed, K)
        integer, intent(in) :: K
        integer(int8), intent(in) :: w(K)
        integer(int8), intent(out) :: packed((K / GROUP_W) * GROUP_B)
        integer :: g, p, b
        integer(int64) :: word, raw7
        do g = 0, K / GROUP_W - 1
            word = 0_int64
            do p = 0, 7
                raw7 = ior(ishft(int(iand(int(w(g*16 + 2*p + 1), int32), 15_int32), int64), 3), &
                           int(iand(int(w(g*16 + 2*p + 2), int32), 7_int32), int64))
                word = ior(word, ishft(raw7, 7 * p))
            end do
            do b = 0, 6
                packed(g*7 + b + 1) = int(iand(ishft(word, -8 * b), 255_int64) - &
                                          merge(256_int64, 0_int64, &
                                                iand(ishft(word, -8 * b), 255_int64) > 127_int64), int8)
            end do
        end do
    end subroutine pack_column

    pure subroutine unpack_column(packed, w, K)
        integer, intent(in) :: K
        integer(int8), intent(in) :: packed((K / GROUP_W) * GROUP_B)
        integer(int8), intent(out) :: w(K)
        integer :: g, p, b
        integer(int64) :: word, raw7
        do g = 0, K / GROUP_W - 1
            word = 0_int64
            do b = 0, 6
                word = ior(word, ishft(iand(int(packed(g*7 + b + 1), int64), 255_int64), 8 * b))
            end do
            do p = 0, 7
                raw7 = iand(ishft(word, -7 * p), 127_int64)
                w(g*16 + 2*p + 1) = int(sext(int(ishft(raw7, -3), int32), 4), int8)
                w(g*16 + 2*p + 2) = int(sext(int(iand(raw7, 7_int64), int32), 3), int8)
            end do
        end do
    end subroutine unpack_column

    ! C(i,j) = sum_k A(i,k) * w(k,j), with w stored packed per column.
    ! Wp: (K/16*7, N). Accumulates in int32; see acc_bound for the safe K.
    subroutine matmul_packed3p5(A, Wp, C, M, N, K)
        integer, intent(in) :: M, N, K
        integer(int8), intent(in) :: A(M, K)
        integer(int8), intent(in) :: Wp((K / GROUP_W) * GROUP_B, N)
        integer(int32), intent(out) :: C(M, N)
        integer :: i, j, g, p, b, k0
        integer(int64) :: word, raw7
        integer(int32) :: n1, n2, acc
        do j = 1, N
            do i = 1, M
                acc = 0
                do g = 0, K / GROUP_W - 1
                    word = 0_int64
                    do b = 0, 6
                        word = ior(word, ishft(iand(int(Wp(g*7 + b + 1, j), int64), 255_int64), 8 * b))
                    end do
                    k0 = g * 16
                    do p = 0, 7
                        raw7 = iand(ishft(word, -7 * p), 127_int64)
                        n1 = sext(int(ishft(raw7, -3), int32), 4)
                        n2 = sext(int(iand(raw7, 7_int64), int32), 3)
                        acc = acc + int(A(i, k0 + 2*p + 1), int32) * n1 &
                                  + int(A(i, k0 + 2*p + 2), int32) * n2
                    end do
                end do
                C(i, j) = acc
            end do
        end do
    end subroutine matmul_packed3p5

    ! INT4 baseline: two 4-bit weights per byte (low nibble = even index).
    pure subroutine pack4_column(w, packed, K)
        integer, intent(in) :: K
        integer(int8), intent(in) :: w(K)
        integer(int8), intent(out) :: packed(K / 2)
        integer :: k2
        integer(int32) :: v
        do k2 = 1, K / 2
            v = ior(iand(int(w(2*k2 - 1), int32), 15_int32), ishft(iand(int(w(2*k2), int32), 15_int32), 4))
            if (v > 127) v = v - 256
            packed(k2) = int(v, int8)
        end do
    end subroutine pack4_column

    subroutine matmul_packed4(A, Wp, C, M, N, K)
        integer, intent(in) :: M, N, K
        integer(int8), intent(in) :: A(M, K)
        integer(int8), intent(in) :: Wp(K / 2, N)
        integer(int32), intent(out) :: C(M, N)
        integer :: i, j, k2
        integer(int32) :: acc, byte, n1, n2
        do j = 1, N
            do i = 1, M
                acc = 0
                do k2 = 1, K / 2
                    byte = iand(int(Wp(k2, j), int32), 255_int32)
                    n1 = sext(byte, 4)
                    n2 = sext(ishft(byte, -4), 4)
                    acc = acc + int(A(i, 2*k2 - 1), int32) * n1 + int(A(i, 2*k2), int32) * n2
                end do
                C(i, j) = acc
            end do
        end do
    end subroutine matmul_packed4

    ! Largest K for which an int32 accumulator cannot overflow:
    ! |A| <= 128, |w| <= 8, so each term is at most 1024 in magnitude.
    pure function acc_bound() result(kmax)
        integer(int64) :: kmax
        kmax = int(huge(0_int32), int64) / 1024_int64
    end function acc_bound

end module pack3p5
