! Minimal test to isolate INT4 matmul issue with loaded weights
program test_int4_matmul
    use iso_fortran_env, only: int8, int32, real32
    use matmul_int4_groq
    use transformer_layer
    implicit none

    ! Test dimensions (small for debugging)
    integer(int32), parameter :: M = 2        ! Batch/sequence length
    integer(int32), parameter :: K_DIM = HIDDEN_DIM  ! 8192
    integer(int32), parameter :: N = NUM_HEADS * HEAD_DIM  ! 8192

    ! Arrays
    integer(int8), allocatable :: A(:,:)           ! Input activations [M, K_DIM]
    integer(int8), allocatable :: W_Q(:,:)         ! Quantized weights [K_DIM/2, N]
    real(real32), allocatable :: W_scales(:)       ! Scales [N]
    integer(int32), allocatable :: C(:,:)          ! Output accumulator [M, N]
    real(real32), allocatable :: output(:,:)       ! Final output [M, N]

    integer :: unit, ios, i, j

    print *, "==========================================="
    print *, "INT4 MatMul Test with Loaded Weights"
    print *, "==========================================="
    print *, "Test dimensions:"
    print *, "  M (batch):", M
    print *, "  K (input):", K_DIM
    print *, "  N (output):", N
    print *, "  Weight shape: [", K_DIM/2, ",", N, "]"
    print *, ""

    ! Allocate arrays
    print *, "Allocating arrays..."
    allocate(A(M, K_DIM))
    allocate(W_Q(K_DIM/2, N))
    allocate(W_scales(N))
    allocate(C(M, N))
    allocate(output(M, N))
    print *, "  ✓ Arrays allocated"
    print *, ""

    ! Initialize input with small values
    print *, "Initializing input activations..."
    do i = 1, M
        do j = 1, K_DIM
            A(i,j) = int(mod(i+j, 15) - 7, int8)  ! Values -7 to 7
        end do
    end do
    print *, "  ✓ Input initialized"
    print *, "  Sample values:", A(1, 1:5)
    print *, ""

    ! Load weights from test file
    print *, "Loading weights from test_weights_layer0.bin..."
    open(newunit=unit, file='test_weights_layer0.bin', form='unformatted', &
         access='stream', status='old', action='read', iostat=ios)

    if (ios /= 0) then
        print *, "  ✗ Error: Could not open test_weights_layer0.bin"
        print *, "  Run 'make gen-weights' first"
        stop 1
    end if

    ! Read Q weights (first in file)
    read(unit, iostat=ios) W_Q
    if (ios /= 0) then
        print *, "  ✗ Error reading W_Q"
        stop 1
    end if

    read(unit, iostat=ios) W_scales
    if (ios /= 0) then
        print *, "  ✗ Error reading W_scales"
        stop 1
    end if

    close(unit)
    print *, "  ✓ Weights loaded successfully"
    print *, "  Weight samples:", W_Q(1, 1:5)
    print *, "  Scale samples:", W_scales(1:5)
    print *, ""

    ! Test INT4 matmul
    print *, "Running INT4 matrix multiplication..."
    print *, "  Calling: matmul_int4_awq(A, W_Q, W_scales, C, M, N, K_DIM)"
    print *, "  This is where the segfault may occur..."
    print *, ""

    call matmul_int4_awq(A, W_Q, W_scales, C, M, N, K_DIM)

    print *, "  ✓ matmul_int4_awq completed!"
    print *, ""

    ! Dequantize output
    print *, "Dequantizing output..."
    call dequantize_output(C, W_scales, output, M, N)
    print *, "  ✓ Dequantization complete"
    print *, ""

    ! Show results
    print *, "Results:"
    print *, "  Output shape: [", M, ",", N, "]"
    print *, "  Sample outputs:", output(1, 1:5)
    print *, ""

    ! Cleanup
    deallocate(A, W_Q, W_scales, C, output)

    print *, "==========================================="
    print *, "✓ INT4 MatMul Test PASSED!"
    print *, "==========================================="

end program test_int4_matmul
