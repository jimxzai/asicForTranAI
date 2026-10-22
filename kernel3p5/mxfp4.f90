! Reference MXFP4 (OCP Microscaling) weight blocks and matmul.
!
! A block holds 32 weights as 16 bytes of E2M1 codes (low nibble = even index) plus one
! E8M0 scale byte e, scale = 2**(e-127); e = 255 is reserved (NaN) and rejected.
! E2M1 code: bit 3 = sign, bits 2..0 index {0, .5, 1, 1.5, 2, 3, 4, 6}.
! Twice an E2M1 value is an integer in {0,1,2,3,4,6,8,12}, so a block dot product with
! int8 activations is an EXACT integer sum, scaled once per block by a power of two:
!     contribution = acc * 2**(e-127) / 2,   |acc| <= 32 * 128 * 12 = 49152.
module mxfp4
    use iso_fortran_env, only: int8, int32, int64, real64
    implicit none
    private
    public :: BLOCK_W, BLOCK_B, mx_decode_code, mx_encode_value, pack_mx_column, &
              matmul_mx, mx_block_acc_bound
    integer, parameter :: BLOCK_W = 32, BLOCK_B = 17      ! 16 code bytes + 1 scale byte
    ! Signed doubled E2M1 value for each 4-bit code: a 16-entry table beats branching on the sign bit.
    integer(int32), parameter :: SVT(0:15) = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
    real(real64), parameter :: E2M1(0:7) = [0.0_real64, 0.5_real64, 1.0_real64, 1.5_real64, &
                                            2.0_real64, 3.0_real64, 4.0_real64, 6.0_real64]
contains

    pure function mx_decode_code(code) result(v)          ! unscaled E2M1 value
        integer, intent(in) :: code                       ! 0..15
        real(real64) :: v
        v = E2M1(iand(code, 7))
        if (iand(code, 8) /= 0) v = -v
    end function mx_decode_code

    pure function mx_encode_value(x) result(code)         ! nearest E2M1 code (ties to the smaller magnitude)
        real(real64), intent(in) :: x
        integer :: code, i, best
        real(real64) :: d, bd
        best = 0; bd = huge(1.0_real64)
        do i = 0, 7
            d = abs(abs(x) - E2M1(i))
            if (d < bd) then; bd = d; best = i; end if
        end do
        code = best
        if (x < 0.0_real64 .and. best /= 0) code = ior(code, 8)
    end function mx_encode_value

    ! w: K real weights (K multiple of 32). packed: (K/32)*17 bytes. Scale per OCP spec:
    ! e = floor(log2(max|w|)) - 2 + 127, clipped to [0,254].
    pure subroutine pack_mx_column(w, packed, K)
        integer, intent(in) :: K
        real(real64), intent(in) :: w(K)
        integer(int8), intent(out) :: packed((K / BLOCK_W) * BLOCK_B)
        integer :: b, t, e, lo, hi, base
        real(real64) :: amax, s
        do b = 0, K / BLOCK_W - 1
            base = b * BLOCK_B
            amax = maxval(abs(w(b*32 + 1 : b*32 + 32)))
            if (amax > 0.0_real64) then
                e = floor(log(amax) / log(2.0_real64)) - 2 + 127
            else
                e = 127
            end if
            e = max(0, min(254, e))
            s = 2.0_real64 ** (e - 127)
            do t = 0, 15
                lo = mx_encode_value(max(-6.0_real64, min(6.0_real64, w(b*32 + 2*t + 1) / s)))
                hi = mx_encode_value(max(-6.0_real64, min(6.0_real64, w(b*32 + 2*t + 2) / s)))
                packed(base + t + 1) = byte_of(ior(lo, ishft(hi, 4)))
            end do
            packed(base + 17) = byte_of(e)
        end do
    end subroutine pack_mx_column

    pure function byte_of(v) result(b)
        integer, intent(in) :: v
        integer(int8) :: b
        b = int(merge(v - 256, v, v > 127), int8)
    end function byte_of

    ! C(i,j) = sum over blocks of 2**(e-127) * (sum_k A(i,k) * E2M1(code_k)), accumulated in float64.
    ! Per block the sum is an exact int32; the only rounding is in the float64 accumulation.
    subroutine matmul_mx(A, Wp, C, M, N, K)
        integer, intent(in) :: M, N, K
        integer(int8), intent(in) :: A(M, K)
        integer(int8), intent(in) :: Wp((K / BLOCK_W) * BLOCK_B, N)
        real(real64), intent(out) :: C(M, N)
        integer :: i, j, b, t, base, k0
        integer(int32) :: acc, byte, lo, hi
        real(real64) :: tot
        do j = 1, N
            do i = 1, M
                tot = 0.0_real64
                do b = 0, K / BLOCK_W - 1
                    base = b * BLOCK_B; k0 = b * 32
                    acc = 0
                    do t = 0, 15
                        byte = iand(int(Wp(base + t + 1, j), int32), 255_int32)
                        lo = iand(byte, 15_int32); hi = ishft(byte, -4)
                        acc = acc + int(A(i, k0 + 2*t + 1), int32) * SVT(lo) &
                                  + int(A(i, k0 + 2*t + 2), int32) * SVT(hi)
                    end do
                    tot = tot + scale(real(acc, real64), int(iand(int(Wp(base + 17, j), int32), 255_int32)) - 128)
                end do
                C(i, j) = tot
            end do
        end do
    end subroutine matmul_mx


    pure function mx_block_acc_bound() result(b)
        integer(int64) :: b
        b = 32_int64 * 128_int64 * 12_int64
    end function mx_block_acc_bound
end module mxfp4
