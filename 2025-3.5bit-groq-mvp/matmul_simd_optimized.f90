! SIMD-Optimized INT4/3.5-bit MatMul - Target: 2.3× speedup
! Uses OpenMP SIMD directives + manual vectorization
! Pure Fortran 2023 with SIMD extensions

module matmul_simd_optimized
    use iso_fortran_env, only: int8, int32, real32
    use omp_lib
    implicit none

    private
    public :: matmul_int4_simd, dequantize_output_simd

    ! Precomputed lookup tables for sign extension
    integer(int32), parameter :: SIGN_EXTEND_4BIT(0:15) = [ &
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1 ]

    integer(int32), parameter :: SIGN_EXTEND_3BIT(0:7) = [ &
        0, 1, 2, 3, -4, -3, -2, -1 ]

contains

    !> SIMD-Optimized INT4 matrix multiplication with AWQ
    !> Uses OpenMP SIMD + manual unrolling for maximum throughput
    subroutine matmul_int4_simd(A, W_Q, W_scales, C, M, N, K_dim)
        integer(int8), intent(in) :: A(:,:)           ! [M, K]
        integer(int8), intent(in) :: W_Q(:,:)         ! [K/8, N]
        real(real32), intent(in) :: W_scales(:)       ! [N]
        integer(int32), intent(out) :: C(:,:)         ! [M, N]
        integer(int32), intent(in) :: M, N, K_dim

        integer(int32) :: i, j, k_idx, k_packed
        integer(int32) :: packed_byte, qval1, qval2
        integer(int32) :: accum
        integer(int32) :: qvals(8)  ! For SIMD processing
        integer(int32) :: a_vals(8)
        integer(int32) :: k

        ! Outer parallelization with OpenMP
        !$omp parallel do private(i,j,k_idx,k_packed,packed_byte,qval1,qval2,accum,qvals,a_vals,k) schedule(static)
        do j = 1, N
            do i = 1, M
                accum = 0

                ! Process 8 values at a time with SIMD
                !$omp simd reduction(+:accum)
                do k_idx = 1, K_dim - 7, 8
                    ! Unpack 8 values from 4 packed bytes
                    ! Byte 1: values 1,2
                    k_packed = (k_idx + 1) / 2
                    packed_byte = int(W_Q(k_packed, j), int32)
                    qvals(1) = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    qvals(2) = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))

                    ! Byte 2: values 3,4
                    k_packed = (k_idx + 3) / 2
                    packed_byte = int(W_Q(k_packed, j), int32)
                    qvals(3) = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    qvals(4) = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))

                    ! Byte 3: values 5,6
                    k_packed = (k_idx + 5) / 2
                    packed_byte = int(W_Q(k_packed, j), int32)
                    qvals(5) = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    qvals(6) = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))

                    ! Byte 4: values 7,8
                    k_packed = (k_idx + 7) / 2
                    packed_byte = int(W_Q(k_packed, j), int32)
                    qvals(7) = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    qvals(8) = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))

                    ! Load activations and compute (compiler will vectorize)
                    a_vals(1) = int(A(i, k_idx), int32)
                    a_vals(2) = int(A(i, k_idx+1), int32)
                    a_vals(3) = int(A(i, k_idx+2), int32)
                    a_vals(4) = int(A(i, k_idx+3), int32)
                    a_vals(5) = int(A(i, k_idx+4), int32)
                    a_vals(6) = int(A(i, k_idx+5), int32)
                    a_vals(7) = int(A(i, k_idx+6), int32)
                    a_vals(8) = int(A(i, k_idx+7), int32)

                    ! SIMD multiply-accumulate (8 operations in parallel)
                    accum = accum + a_vals(1) * qvals(1) + &
                                    a_vals(2) * qvals(2) + &
                                    a_vals(3) * qvals(3) + &
                                    a_vals(4) * qvals(4) + &
                                    a_vals(5) * qvals(5) + &
                                    a_vals(6) * qvals(6) + &
                                    a_vals(7) * qvals(7) + &
                                    a_vals(8) * qvals(8)
                end do

                ! Handle remaining elements
                do k_idx = ((K_dim / 8) * 8) + 1, K_dim, 2
                    k_packed = (k_idx + 1) / 2
                    packed_byte = int(W_Q(k_packed, j), int32)

                    qval1 = SIGN_EXTEND_4BIT(iand(packed_byte, 15))
                    accum = accum + int(A(i, k_idx), int32) * qval1

                    if (k_idx + 1 <= K_dim) then
                        qval2 = SIGN_EXTEND_4BIT(iand(ishft(packed_byte, -4), 15))
                        accum = accum + int(A(i, k_idx + 1), int32) * qval2
                    end if
                end do

                C(i, j) = accum
            end do
        end do
        !$omp end parallel do

    end subroutine matmul_int4_simd

    !> SIMD-optimized dequantization
    subroutine dequantize_output_simd(C, W_scales, Out, M, N)
        integer(int32), intent(in) :: C(:,:)      ! [M, N]
        real(real32), intent(in) :: W_scales(:)   ! [N]
        real(real32), intent(out) :: Out(:,:)     ! [M, N]
        integer(int32), intent(in) :: M, N

        integer(int32) :: i, j

        !$omp parallel do private(i,j) schedule(static)
        do j = 1, N
            !$omp simd
            do i = 1, M
                Out(i, j) = real(C(i, j), real32) * W_scales(j)
            end do
        end do
        !$omp end parallel do

    end subroutine dequantize_output_simd

end module matmul_simd_optimized
