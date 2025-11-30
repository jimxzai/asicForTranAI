! Generate random test weights for LLaMA 70B model
! This creates synthetic weights in the correct shapes for testing
! WITHOUT needing to download 140GB of real weights

program generate_test_weights
    use iso_fortran_env, only: int8, int32, real32
    use transformer_layer
    implicit none

    type(TransformerLayer) :: layer
    integer(int32) :: i, j, seed_val
    real(real32) :: rand_val

    print *, "=========================================="
    print *, "Generating Test Weights for LLaMA 70B"
    print *, "=========================================="
    print *, ""

    ! Initialize random seed
    call random_seed()

    print *, "Allocating weight arrays..."

    ! Allocate Q, K, V projection weights (INT4 packed)
    ! Q: [HIDDEN_DIM, NUM_HEADS * HEAD_DIM] = [8192, 8192]
    ! Packed to INT4: [8192/2, 8192] (2 values per byte)
    allocate(layer%wq(HIDDEN_DIM/2, NUM_HEADS * HEAD_DIM))
    allocate(layer%wq_scales(NUM_HEADS * HEAD_DIM))

    ! K: [HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM] = [8192, 1024]
    allocate(layer%wk(HIDDEN_DIM/2, NUM_KV_HEADS * HEAD_DIM))
    allocate(layer%wk_scales(NUM_KV_HEADS * HEAD_DIM))

    ! V: [HIDDEN_DIM, NUM_KV_HEADS * HEAD_DIM] = [8192, 1024]
    allocate(layer%wv(HIDDEN_DIM/2, NUM_KV_HEADS * HEAD_DIM))
    allocate(layer%wv_scales(NUM_KV_HEADS * HEAD_DIM))

    ! O: [HIDDEN_DIM, HIDDEN_DIM] = [8192, 8192]
    allocate(layer%wo(HIDDEN_DIM/2, HIDDEN_DIM))
    allocate(layer%wo_scales(HIDDEN_DIM))

    ! FFN weights
    ! Gate & Up: [HIDDEN_DIM, INTERMEDIATE_DIM] = [8192, 28672]
    allocate(layer%w_gate(HIDDEN_DIM/2, INTERMEDIATE_DIM))
    allocate(layer%w_gate_scales(INTERMEDIATE_DIM))

    allocate(layer%w_up(HIDDEN_DIM/2, INTERMEDIATE_DIM))
    allocate(layer%w_up_scales(INTERMEDIATE_DIM))

    ! Down: [INTERMEDIATE_DIM, HIDDEN_DIM] = [28672, 8192]
    allocate(layer%w_down(INTERMEDIATE_DIM/2, HIDDEN_DIM))
    allocate(layer%w_down_scales(HIDDEN_DIM))

    print *, "✓ Arrays allocated"
    print *, ""
    print *, "Generating random INT4 weights..."

    ! Generate random INT4 weights (values -7 to 7)
    ! Note: INT4 values are packed 2 per byte in INT8
    do i = 1, size(layer%wq, 1)
        do j = 1, size(layer%wq, 2)
            call random_number(rand_val)
            layer%wq(i,j) = int(rand_val * 30 - 15, int8)  ! -15 to 15
        end do
    end do

    do i = 1, size(layer%wk, 1)
        do j = 1, size(layer%wk, 2)
            call random_number(rand_val)
            layer%wk(i,j) = int(rand_val * 30 - 15, int8)
        end do
    end do

    do i = 1, size(layer%wv, 1)
        do j = 1, size(layer%wv, 2)
            call random_number(rand_val)
            layer%wv(i,j) = int(rand_val * 30 - 15, int8)
        end do
    end do

    do i = 1, size(layer%wo, 1)
        do j = 1, size(layer%wo, 2)
            call random_number(rand_val)
            layer%wo(i,j) = int(rand_val * 30 - 15, int8)
        end do
    end do

    do i = 1, size(layer%w_gate, 1)
        do j = 1, size(layer%w_gate, 2)
            call random_number(rand_val)
            layer%w_gate(i,j) = int(rand_val * 30 - 15, int8)
        end do
    end do

    do i = 1, size(layer%w_up, 1)
        do j = 1, size(layer%w_up, 2)
            call random_number(rand_val)
            layer%w_up(i,j) = int(rand_val * 30 - 15, int8)
        end do
    end do

    do i = 1, size(layer%w_down, 1)
        do j = 1, size(layer%w_down, 2)
            call random_number(rand_val)
            layer%w_down(i,j) = int(rand_val * 30 - 15, int8)
        end do
    end do

    print *, "✓ INT4 weights generated"
    print *, ""
    print *, "Generating random FP32 scales..."

    ! Generate random scales (typically small positive values)
    do i = 1, size(layer%wq_scales)
        call random_number(rand_val)
        layer%wq_scales(i) = rand_val * 0.01  ! Scale factors 0.0 to 0.01
    end do

    do i = 1, size(layer%wk_scales)
        call random_number(rand_val)
        layer%wk_scales(i) = rand_val * 0.01
    end do

    do i = 1, size(layer%wv_scales)
        call random_number(rand_val)
        layer%wv_scales(i) = rand_val * 0.01
    end do

    do i = 1, size(layer%wo_scales)
        call random_number(rand_val)
        layer%wo_scales(i) = rand_val * 0.01
    end do

    do i = 1, size(layer%w_gate_scales)
        call random_number(rand_val)
        layer%w_gate_scales(i) = rand_val * 0.01
    end do

    do i = 1, size(layer%w_up_scales)
        call random_number(rand_val)
        layer%w_up_scales(i) = rand_val * 0.01
    end do

    do i = 1, size(layer%w_down_scales)
        call random_number(rand_val)
        layer%w_down_scales(i) = rand_val * 0.01
    end do

    print *, "✓ Scales generated"
    print *, ""
    print *, "Writing weights to binary file..."

    ! Save to binary file
    open(unit=10, file='test_weights_layer0.bin', form='unformatted', &
         access='stream', status='replace')

    ! Write Q weights
    write(10) layer%wq
    write(10) layer%wq_scales

    ! Write K weights
    write(10) layer%wk
    write(10) layer%wk_scales

    ! Write V weights
    write(10) layer%wv
    write(10) layer%wv_scales

    ! Write O weights
    write(10) layer%wo
    write(10) layer%wo_scales

    ! Write FFN weights
    write(10) layer%w_gate
    write(10) layer%w_gate_scales

    write(10) layer%w_up
    write(10) layer%w_up_scales

    write(10) layer%w_down
    write(10) layer%w_down_scales

    close(10)

    print *, "✓ Weights saved to: test_weights_layer0.bin"
    print *, ""
    print *, "Weight Statistics:"
    print *, "  Q weights:   ", size(layer%wq), " INT8 values (packed INT4)"
    print *, "  K weights:   ", size(layer%wk), " INT8 values (packed INT4)"
    print *, "  V weights:   ", size(layer%wv), " INT8 values (packed INT4)"
    print *, "  O weights:   ", size(layer%wo), " INT8 values (packed INT4)"
    print *, "  Gate weights:", size(layer%w_gate), " INT8 values (packed INT4)"
    print *, "  Up weights:  ", size(layer%w_up), " INT8 values (packed INT4)"
    print *, "  Down weights:", size(layer%w_down), " INT8 values (packed INT4)"
    print *, ""
    print *, "File size: ~", (size(layer%wq) + size(layer%wk) + size(layer%wv) + &
                               size(layer%wo) + size(layer%w_gate) + size(layer%w_up) + &
                               size(layer%w_down)) / 1024 / 1024, " MB per layer"
    print *, ""
    print *, "✓ Test weights generated successfully!"
    print *, ""
    print *, "Next steps:"
    print *, "  1. Create load_weights() function to read this file"
    print *, "  2. Test inference with random weights"
    print *, "  3. Verify shapes and computations work"
    print *, "  4. Replace with real LLaMA weights later"

    ! Cleanup
    deallocate(layer%wq, layer%wq_scales)
    deallocate(layer%wk, layer%wk_scales)
    deallocate(layer%wv, layer%wv_scales)
    deallocate(layer%wo, layer%wo_scales)
    deallocate(layer%w_gate, layer%w_gate_scales)
    deallocate(layer%w_up, layer%w_up_scales)
    deallocate(layer%w_down, layer%w_down_scales)

end program generate_test_weights
