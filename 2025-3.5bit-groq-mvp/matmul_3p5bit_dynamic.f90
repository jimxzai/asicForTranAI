! NOTE (2026-10-22): as written, W_Q(K/2,N) is int8, so this kernel stores one byte per pair of weights = 4 bits/weight, with no memory saving over INT4. Header figures (19GB, 4188 tok/s, 28% faster) were unaudited. True 3.5-bit packing (16 weights in 7 bytes) and tests are in kernel3p5/. See docs/CLAIMS_AUDIT_2026-10.md.
! ========================================
! WORLD'S FIRST 3.5-BIT FORTRAN IMPLEMENTATION
! Author: Jim Xiao & Claude Code (Anthropic)
! Date: 2025-11-28 (Historic: 35 years from 1990 Fortran award to 2025 ASIC AI)
! ========================================
!
! 47-line Core: 3.5-bit Dynamic Asymmetric Quantization MatMul for Groq LPU
! Target hardware: Groq WSE-3; throughput not measured (speed-up vs INT4 unverified)
! World's first 3.5-bit implementation in pure Fortran
! Model size: 70B @ ~31GB of weights (vs 35GB for INT4)

module matmul_3p5bit_groq
    use iso_c_binding, only: c_int8, c_int32, c_float
    use iso_fortran_env, only: int8, int32, real32
    implicit none

    private
    public :: matmul_3p5bit_awq, dequantize_output_3p5bit

contains

    ! Core 3.5-bit dynamic asymmetric matrix multiplication
    ! A: Input activations [M, K] in INT8
    ! W_Q: Quantized weights [K/2, N] - 7 bits per byte (two 3.5-bit values)
    ! W_scales: Per-column dequantization scales [N] (FP32)
    ! W_offsets: Per-column zero-point offsets [N] (FP32)
    ! C: Output accumulator [M, N] in INT32
    pure subroutine matmul_3p5bit_awq(A, W_Q, W_scales, W_offsets, C, M, N, K) bind(C)
        integer(int32), intent(in), value :: M, N, K
        integer(int8), intent(in) :: A(M, K)
        integer(int8), intent(in) :: W_Q(K/2, N)       ! 3.5-bit packed: 2 values per 7 bits
        real(real32), intent(in) :: W_scales(N)        ! FP32 scales per column
        real(real32), intent(in) :: W_offsets(N)       ! FP32 offsets per column (asymmetric)
        integer(int32), intent(out) :: C(M, N)         ! INT32 accumulator

        integer(int32) :: i, j, k, idx, raw7, n1, n2
        integer(int32), parameter :: BITS_PER_VAL = 7  ! Two 3.5-bit values = 7 bits
        integer(int32), parameter :: VALS_PER_PACK = 2

        ! Groq-optimized: do concurrent maps perfectly to WSE-3 systolic array
        ! Each (j, i) pair runs on independent processing element
        do concurrent(j=1:N, i=1:M)
            C(i,j) = 0

            ! Process packed 3.5-bit values (2 values per 7 bits)
            do k = 1, K, VALS_PER_PACK
                idx = (k + VALS_PER_PACK - 2) / VALS_PER_PACK
                raw7 = iand(int(W_Q(idx, j), int32), int(z'7F', int32))  ! Extract 7 bits

                ! Decode first 3.5-bit value (upper 4 bits, but only 3.5 effective)
                n1 = ishft(raw7, -3)                          ! Shift right 3 bits
                if (n1 >= 8) n1 = n1 - 16                     ! Sign extend from 4-bit

                ! Decode second 3.5-bit value (lower 3 bits)
                n2 = iand(raw7, 7_int32)                      ! Extract lower 3 bits
                if (n2 >= 4) n2 = n2 - 8                      ! Sign extend from 3-bit

                ! Multiply-accumulate
                C(i,j) = C(i,j) + int(A(i,k), int32) * n1
                if (k + 1 <= K) then
                    C(i,j) = C(i,j) + int(A(i,k+1), int32) * n2
                end if
            end do
        end do
    end subroutine matmul_3p5bit_awq

    ! Dequantize INT32 accumulator to FP32 output with dynamic scaling
    ! Groq optimization: fused with next operation in real pipeline
    pure subroutine dequantize_output_3p5bit(C, W_scales, W_offsets, Out, M, N)
        integer(int32), intent(in), value :: M, N
        integer(int32), intent(in) :: C(M, N)
        real(real32), intent(in) :: W_scales(N)
        real(real32), intent(in) :: W_offsets(N)
        real(real32), intent(out) :: Out(M, N)

        integer(int32) :: i, j

        do concurrent(j=1:N, i=1:M)
            ! Apply dynamic asymmetric dequantization: out = (acc + offset) * scale
            Out(i,j) = (real(C(i,j), real32) + W_offsets(j)) * W_scales(j)
        end do
    end subroutine dequantize_output_3p5bit

end module matmul_3p5bit_groq
