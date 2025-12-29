! BLAS-Optimized INT4/3.5-bit MatMul - Target: 50-100× speedup
! Uses OpenBLAS SGEMM for hardware-optimized matrix multiplication
! Pure Fortran 2023 with BLAS interface
!
! Performance strategy:
!   1. Dequantize INT4 weights to FP32 (one-time cost, can be cached)
!   2. Convert INT8 activations to FP32
!   3. Call OpenBLAS SGEMM (multi-threaded, cache-blocked, SIMD-vectorized)
!   4. Return FP32 output (skip INT32 accumulation for BLAS path)
!
! Expected speedup: 50-100× vs baseline on multi-core CPU
! Memory overhead: K*N*4 bytes for dequantized weights (temporary)

module matmul_blas_optimized
    use iso_fortran_env, only: int8, int32, real32
    implicit none

    private
    public :: matmul_int4_blas, matmul_int4_blas_cached, dequantize_weights_int4

    ! BLAS interface for single-precision general matrix multiply
    ! C = alpha * A * B + beta * C
    interface
        subroutine sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            character(len=1), intent(in) :: transa, transb
            integer, intent(in) :: m, n, k, lda, ldb, ldc
            real, intent(in) :: alpha, beta
            real, intent(in) :: a(lda, *)
            real, intent(in) :: b(ldb, *)
            real, intent(inout) :: c(ldc, *)
        end subroutine sgemm
    end interface

contains

    !> BLAS-Optimized INT4 matrix multiplication
    !> Dequantizes weights on-the-fly and uses OpenBLAS SGEMM
    !>
    !> @param A - Input activations [M, K] in INT8
    !> @param W_Q - Quantized weights [K/2, N] packed 4-bit
    !> @param W_scales - Per-column dequantization scales [N]
    !> @param Out - Output [M, N] in FP32 (direct output, no INT32 accumulation)
    !> @param M - Number of rows in A and Out
    !> @param N - Number of columns in W and Out
    !> @param K_dim - Inner dimension (K)
    subroutine matmul_int4_blas(A, W_Q, W_scales, Out, M, N, K_dim)
        integer(int8), intent(in) :: A(:,:)           ! [M, K]
        integer(int8), intent(in) :: W_Q(:,:)         ! [K/2, N] packed
        real(real32), intent(in) :: W_scales(:)       ! [N]
        real(real32), intent(out) :: Out(:,:)         ! [M, N]
        integer(int32), intent(in) :: M, N, K_dim

        ! Temporary arrays for BLAS
        real(real32), allocatable :: A_fp32(:,:)      ! [M, K] FP32 activations
        real(real32), allocatable :: W_fp32(:,:)      ! [K, N] FP32 dequantized weights

        integer(int32) :: i, j, k_idx, k_packed
        integer(int32) :: packed_byte, qval

        ! Allocate temporary buffers
        allocate(A_fp32(M, K_dim))
        allocate(W_fp32(K_dim, N))

        ! Step 1: Convert INT8 activations to FP32 (simple cast)
        !$omp parallel do collapse(2) schedule(static)
        do j = 1, K_dim
            do i = 1, M
                A_fp32(i, j) = real(A(i, j), real32)
            end do
        end do
        !$omp end parallel do

        ! Step 2: Dequantize INT4 weights to FP32
        !$omp parallel do private(k_idx,k_packed,packed_byte,qval) schedule(static)
        do j = 1, N
            do k_idx = 1, K_dim, 2
                k_packed = (k_idx + 1) / 2
                packed_byte = int(W_Q(k_packed, j), int32)

                ! Extract first 4-bit value (lower bits)
                qval = iand(packed_byte, 15)
                if (qval >= 8) qval = qval - 16  ! Sign extend
                W_fp32(k_idx, j) = real(qval, real32) * W_scales(j)

                ! Extract second 4-bit value (upper bits)
                if (k_idx + 1 <= K_dim) then
                    qval = iand(ishft(packed_byte, -4), 15)
                    if (qval >= 8) qval = qval - 16
                    W_fp32(k_idx + 1, j) = real(qval, real32) * W_scales(j)
                end if
            end do
        end do
        !$omp end parallel do

        ! Step 3: Call OpenBLAS SGEMM - This is where the magic happens!
        ! Out = 1.0 * A_fp32 * W_fp32 + 0.0 * Out
        ! SGEMM performs: C = alpha*A*B + beta*C
        ! We want: Out[M,N] = A_fp32[M,K] * W_fp32[K,N]
        call sgemm( &
            'N', 'N', &        ! No transpose for A or B
            M, N, K_dim, &     ! Matrix dimensions
            1.0, &             ! alpha = 1.0
            A_fp32, M, &       ! A matrix, leading dimension M
            W_fp32, K_dim, &   ! B matrix, leading dimension K
            0.0, &             ! beta = 0.0 (don't add to existing Out)
            Out, M)            ! C matrix, leading dimension M

        ! Cleanup
        deallocate(A_fp32)
        deallocate(W_fp32)

    end subroutine matmul_int4_blas


    !> BLAS-Optimized INT4 matmul with cached dequantized weights
    !> Use this when weights are reused across many forward passes
    !>
    !> @param A - Input activations [M, K] in INT8
    !> @param W_fp32 - Pre-dequantized weights [K, N] in FP32 (CACHED)
    !> @param Out - Output [M, N] in FP32
    !> @param M - Number of rows
    !> @param N - Number of columns
    !> @param K_dim - Inner dimension
    subroutine matmul_int4_blas_cached(A, W_fp32, Out, M, N, K_dim)
        integer(int8), intent(in) :: A(:,:)           ! [M, K]
        real(real32), intent(in) :: W_fp32(:,:)       ! [K, N] PRE-DEQUANTIZED
        real(real32), intent(out) :: Out(:,:)         ! [M, N]
        integer(int32), intent(in) :: M, N, K_dim

        real(real32), allocatable :: A_fp32(:,:)      ! [M, K]
        integer(int32) :: i, j

        ! Allocate activation buffer
        allocate(A_fp32(M, K_dim))

        ! Convert INT8 activations to FP32
        !$omp parallel do collapse(2) schedule(static)
        do j = 1, K_dim
            do i = 1, M
                A_fp32(i, j) = real(A(i, j), real32)
            end do
        end do
        !$omp end parallel do

        ! Call OpenBLAS SGEMM with pre-dequantized weights
        ! This is MUCH faster for inference (weights reused per token)
        call sgemm( &
            'N', 'N', &
            M, N, K_dim, &
            1.0, &
            A_fp32, M, &
            W_fp32, K_dim, &
            0.0, &
            Out, M)

        deallocate(A_fp32)

    end subroutine matmul_int4_blas_cached


    !> Helper: Dequantize INT4 weights to FP32 (for weight caching)
    !> Call this once at model load time, then use matmul_int4_blas_cached()
    !>
    !> @param W_Q - Quantized weights [K/2, N] packed 4-bit
    !> @param W_scales - Per-column scales [N]
    !> @param W_fp32 - Output dequantized weights [K, N] in FP32
    !> @param N - Number of columns
    !> @param K_dim - Number of rows (unpacked)
    subroutine dequantize_weights_int4(W_Q, W_scales, W_fp32, N, K_dim)
        integer(int8), intent(in) :: W_Q(:,:)         ! [K/2, N]
        real(real32), intent(in) :: W_scales(:)       ! [N]
        real(real32), intent(out) :: W_fp32(:,:)      ! [K, N]
        integer(int32), intent(in) :: N, K_dim

        integer(int32) :: j, k_idx, k_packed
        integer(int32) :: packed_byte, qval

        !$omp parallel do private(k_idx,k_packed,packed_byte,qval) schedule(static)
        do j = 1, N
            do k_idx = 1, K_dim, 2
                k_packed = (k_idx + 1) / 2
                packed_byte = int(W_Q(k_packed, j), int32)

                ! First value
                qval = iand(packed_byte, 15)
                if (qval >= 8) qval = qval - 16
                W_fp32(k_idx, j) = real(qval, real32) * W_scales(j)

                ! Second value
                if (k_idx + 1 <= K_dim) then
                    qval = iand(ishft(packed_byte, -4), 15)
                    if (qval >= 8) qval = qval - 16
                    W_fp32(k_idx + 1, j) = real(qval, real32) * W_scales(j)
                end if
            end do
        end do
        !$omp end parallel do

    end subroutine dequantize_weights_int4

end module matmul_blas_optimized
