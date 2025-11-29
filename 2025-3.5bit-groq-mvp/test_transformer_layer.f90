! Test program for transformer layer
! Compile: gfortran -O3 matmul_int4_groq.f90 transformer_layer.f90 test_transformer_layer.f90 -o test_layer

program test_transformer_layer
    use iso_fortran_env, only: int32, real32
    use transformer_layer
    implicit none

    type(TransformerLayer) :: layer
    real(real32), allocatable :: x(:,:), output(:,:)
    integer(int32) :: seq_len
    integer(int32) :: i, j

    print *, "=========================================="
    print *, "LLaMA 70B Transformer Layer Test"
    print *, "Pure Fortran 2023 - ASIC Optimized"
    print *, "=========================================="
    print *, ""

    ! Test with small sequence
    seq_len = 4
    print *, "Test configuration:"
    print *, "  Sequence length:", seq_len
    print *, "  Hidden dim:", HIDDEN_DIM
    print *, "  Num heads:", NUM_HEADS
    print *, "  KV heads:", NUM_KV_HEADS
    print *, "  Head dim:", HEAD_DIM
    print *, ""

    ! Allocate buffers
    allocate(x(seq_len, HIDDEN_DIM))
    allocate(output(seq_len, HIDDEN_DIM))

    ! Allocate layer weights (simplified - normally loaded from file)
    allocate(layer%attn_norm(HIDDEN_DIM))
    allocate(layer%ffn_norm(HIDDEN_DIM))

    ! Initialize with simple test data
    layer%attn_norm = 1.0
    layer%ffn_norm = 1.0

    ! Initialize RoPE frequency cache (max_seq_len = 2048 for LLaMA)
    call init_rope_freqs(layer, 2048)

    ! Create test input (small random values)
    do i = 1, seq_len
        do j = 1, HIDDEN_DIM
            x(i,j) = 0.01 * real(mod(i*j, 100), real32)
        end do
    end do

    print *, "Input shape: [", seq_len, ",", HIDDEN_DIM, "]"
    print *, "Input sample (first position, first 8 dims):"
    write(*, '(8F10.6)') x(1, 1:8)
    print *, ""

    ! Apply transformer layer
    print *, "Running transformer layer..."
    call apply_transformer_layer(layer, x, output, seq_len)

    print *, ""
    print *, "Output shape: [", seq_len, ",", HIDDEN_DIM, "]"
    print *, "Output sample (first position, first 8 dims):"
    write(*, '(8F10.6)') output(1, 1:8)
    print *, ""

    print *, "✓ Transformer layer test completed!"
    print *, ""
    print *, "Next steps:"
    print *, "  1. Replace placeholder matmuls with INT4 quantized versions"
    print *, "  2. Load real LLaMA 70B weights"
    print *, "  3. Implement KV caching for generation"
    print *, "  4. Stack 80 layers for full model"

    ! Cleanup
    deallocate(x, output)
    deallocate(layer%attn_norm, layer%ffn_norm)
    if (allocated(layer%rope_freqs)) deallocate(layer%rope_freqs)

end program test_transformer_layer
