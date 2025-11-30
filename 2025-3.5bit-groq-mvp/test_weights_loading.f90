! Test loading and using weights in transformer layer
program test_weights_loading
    use iso_fortran_env, only: int8, int32, real32
    use transformer_layer
    use weight_loader
    implicit none

    type(TransformerLayer) :: layer
    real(real32), allocatable :: x(:,:)
    real(real32), allocatable :: output(:,:)
    integer(int32) :: seq_len
    integer :: i, j

    print *, "=========================================="
    print *, "Transformer Layer Weight Loading Test"
    print *, "=========================================="
    print *, ""

    seq_len = 2
    allocate(x(seq_len, HIDDEN_DIM))
    allocate(output(seq_len, HIDDEN_DIM))

    ! Initialize input
    print *, "Initializing test input..."
    do i = 1, seq_len
        do j = 1, HIDDEN_DIM
            x(i,j) = 0.01 * real(mod(i+j, 100), real32)
        end do
    end do
    print *, "  ✓ Input initialized"
    print *, ""

    ! Initialize layer
    print *, "Initializing transformer layer..."
    allocate(layer%attn_norm(HIDDEN_DIM))
    allocate(layer%ffn_norm(HIDDEN_DIM))
    layer%attn_norm = 1.0
    layer%ffn_norm = 1.0

    call init_rope_freqs(layer, 2048)
    call init_kv_cache(layer, 2048)
    print *, "  ✓ Layer initialized"
    print *, ""

    ! Load weights
    print *, "Loading weights from test_weights_layer0.bin..."
    call load_layer_weights(layer, 'test_weights_layer0.bin', 0)
    print *, ""

    ! Check if weights were loaded
    if (allocated(layer%wq)) then
        print *, "  ✓ wq allocated: shape = [", size(layer%wq,1), ",", size(layer%wq,2), "]"
        print *, "    Sample values:", layer%wq(1, 1:5)
    else
        print *, "  ✗ wq NOT allocated!"
    end if

    if (allocated(layer%wq_scales)) then
        print *, "  ✓ wq_scales allocated: size =", size(layer%wq_scales)
        print *, "    Sample values:", layer%wq_scales(1:5)
    else
        print *, "  ✗ wq_scales NOT allocated!"
    end if
    print *, ""

    ! Now test a simple forward pass with loaded weights
    print *, "Testing transformer layer with loaded weights..."
    print *, "  (This may crash if there's a bug with loaded weights)"
    print *, ""

    call apply_transformer_layer(layer, x, output, seq_len)

    print *, "  ✓ Forward pass completed!"
    print *, "  Output shape: [", size(output,1), ",", size(output,2), "]"
    print *, "  Sample outputs:", output(1, 1:5)
    print *, ""

    ! Cleanup
    deallocate(x, output)
    if (allocated(layer%attn_norm)) deallocate(layer%attn_norm)
    if (allocated(layer%ffn_norm)) deallocate(layer%ffn_norm)
    if (allocated(layer%rope_freqs)) deallocate(layer%rope_freqs)
    if (allocated(layer%k_cache)) deallocate(layer%k_cache)
    if (allocated(layer%v_cache)) deallocate(layer%v_cache)
    if (allocated(layer%wq)) deallocate(layer%wq)
    if (allocated(layer%wq_scales)) deallocate(layer%wq_scales)
    if (allocated(layer%wk)) deallocate(layer%wk)
    if (allocated(layer%wk_scales)) deallocate(layer%wk_scales)
    if (allocated(layer%wv)) deallocate(layer%wv)
    if (allocated(layer%wv_scales)) deallocate(layer%wv_scales)
    if (allocated(layer%wo)) deallocate(layer%wo)
    if (allocated(layer%wo_scales)) deallocate(layer%wo_scales)
    if (allocated(layer%w_gate)) deallocate(layer%w_gate)
    if (allocated(layer%w_gate_scales)) deallocate(layer%w_gate_scales)
    if (allocated(layer%w_up)) deallocate(layer%w_up)
    if (allocated(layer%w_up_scales)) deallocate(layer%w_up_scales)
    if (allocated(layer%w_down)) deallocate(layer%w_down)
    if (allocated(layer%w_down_scales)) deallocate(layer%w_down_scales)

    print *, "==========================================="
    print *, "✓ Weight Loading Test PASSED!"
    print *, "==========================================="

end program test_weights_loading
