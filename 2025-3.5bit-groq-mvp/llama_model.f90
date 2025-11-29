! LLaMA 70B Full Model - 80 Transformer Layers
! Pure Fortran 2023 - ASIC Optimized
! Target: 3100+ tokens/sec on Groq LPU

module llama_model
    use iso_fortran_env, only: int32, real32
    use transformer_layer
    implicit none

    private
    public :: LLaMAModel, init_llama_model, forward_llama, cleanup_llama_model

    ! LLaMA 70B architecture
    integer(int32), parameter :: NUM_LAYERS = 80
    integer(int32), parameter :: VOCAB_SIZE = 32000
    integer(int32), parameter :: MAX_SEQ_LEN = 2048

    ! Full model structure
    type :: LLaMAModel
        ! Token embedding
        real(real32), allocatable :: token_embeddings(:,:)  ! [VOCAB_SIZE, HIDDEN_DIM]

        ! 80 transformer layers
        type(TransformerLayer), allocatable :: layers(:)  ! [NUM_LAYERS]

        ! Final normalization
        real(real32), allocatable :: final_norm(:)  ! [HIDDEN_DIM]

        ! Output projection (LM head)
        real(real32), allocatable :: output_weights(:,:)  ! [HIDDEN_DIM, VOCAB_SIZE]

        ! Model metadata
        integer(int32) :: num_layers
        integer(int32) :: max_seq_len
    end type LLaMAModel

contains

    !===========================================================================
    ! Initialize LLaMA 70B model with 80 layers
    !===========================================================================
    subroutine init_llama_model(model)
        type(LLaMAModel), intent(inout) :: model
        integer(int32) :: i

        print *, "=========================================="
        print *, "Initializing LLaMA 70B Model"
        print *, "=========================================="
        print *, "Architecture:"
        print *, "  Layers:", NUM_LAYERS
        print *, "  Hidden dim:", HIDDEN_DIM
        print *, "  Intermediate:", INTERMEDIATE_DIM
        print *, "  Vocab size:", VOCAB_SIZE
        print *, "  Max seq len:", MAX_SEQ_LEN
        print *, ""

        ! Set model config
        model%num_layers = NUM_LAYERS
        model%max_seq_len = MAX_SEQ_LEN

        ! Allocate token embeddings
        allocate(model%token_embeddings(VOCAB_SIZE, HIDDEN_DIM))
        model%token_embeddings = 0.0

        ! Allocate 80 transformer layers
        allocate(model%layers(NUM_LAYERS))

        ! Initialize each layer
        print *, "Initializing layers..."
        do i = 1, NUM_LAYERS
            ! Allocate layer normalization weights
            allocate(model%layers(i)%attn_norm(HIDDEN_DIM))
            allocate(model%layers(i)%ffn_norm(HIDDEN_DIM))
            model%layers(i)%attn_norm = 1.0
            model%layers(i)%ffn_norm = 1.0

            ! Initialize RoPE frequencies
            call init_rope_freqs(model%layers(i), MAX_SEQ_LEN)

            ! Initialize KV cache
            call init_kv_cache(model%layers(i), MAX_SEQ_LEN)

            ! Note: Weight loading (wq, wk, wv, wo, w_gate, w_up, w_down)
            ! will be done separately from safetensors file

            if (mod(i, 10) == 0) then
                print '(A,I3,A)', "  ✓ Initialized ", i, " layers"
            end if
        end do

        ! Allocate final norm and output weights
        allocate(model%final_norm(HIDDEN_DIM))
        allocate(model%output_weights(HIDDEN_DIM, VOCAB_SIZE))
        model%final_norm = 1.0
        model%output_weights = 0.0

        print *, "✓ Model initialization complete!"
        print *, ""

    end subroutine init_llama_model

    !===========================================================================
    ! Forward pass through all 80 layers
    !===========================================================================
    subroutine forward_llama(model, token_ids, output_logits, seq_len)
        type(LLaMAModel), intent(inout) :: model
        integer(int32), intent(in) :: token_ids(:)  ! Input token IDs
        real(real32), intent(out) :: output_logits(:,:)  ! [seq_len, VOCAB_SIZE]
        integer(int32), intent(in), value :: seq_len

        ! Allocatable to avoid stack overflow
        real(real32), allocatable :: x(:,:)
        real(real32), allocatable :: layer_out(:,:)
        real(real32), allocatable :: x_norm(:,:)
        integer(int32) :: i, j, layer_idx

        ! Allocate working arrays
        allocate(x(seq_len, HIDDEN_DIM))
        allocate(layer_out(seq_len, HIDDEN_DIM))
        allocate(x_norm(seq_len, HIDDEN_DIM))

        ! 1. Token embedding lookup
        do i = 1, seq_len
            x(i, :) = model%token_embeddings(token_ids(i), :)
        end do

        ! 2. Pass through all 80 transformer layers
        do layer_idx = 1, NUM_LAYERS
            call apply_transformer_layer(model%layers(layer_idx), x, layer_out, seq_len)
            x = layer_out  ! Feed output to next layer
        end do

        ! 3. Final RMSNorm
        call rms_norm(x, model%final_norm, x_norm, seq_len, HIDDEN_DIM)

        ! 4. Output projection to vocabulary
        ! output_logits[i,j] = sum_k(x_norm[i,k] * output_weights[k,j])
        do concurrent(i = 1:seq_len, j = 1:VOCAB_SIZE)
            output_logits(i,j) = dot_product(x_norm(i,:), model%output_weights(:,j))
        end do

        ! Clean up
        if (allocated(x)) deallocate(x)
        if (allocated(layer_out)) deallocate(layer_out)
        if (allocated(x_norm)) deallocate(x_norm)

    end subroutine forward_llama

    !===========================================================================
    ! Cleanup model resources
    !===========================================================================
    subroutine cleanup_llama_model(model)
        type(LLaMAModel), intent(inout) :: model
        integer(int32) :: i

        ! Clean up embeddings and output
        if (allocated(model%token_embeddings)) deallocate(model%token_embeddings)
        if (allocated(model%final_norm)) deallocate(model%final_norm)
        if (allocated(model%output_weights)) deallocate(model%output_weights)

        ! Clean up each layer
        if (allocated(model%layers)) then
            do i = 1, model%num_layers
                if (allocated(model%layers(i)%attn_norm)) &
                    deallocate(model%layers(i)%attn_norm)
                if (allocated(model%layers(i)%ffn_norm)) &
                    deallocate(model%layers(i)%ffn_norm)
                if (allocated(model%layers(i)%rope_freqs)) &
                    deallocate(model%layers(i)%rope_freqs)
                if (allocated(model%layers(i)%k_cache)) &
                    deallocate(model%layers(i)%k_cache)
                if (allocated(model%layers(i)%v_cache)) &
                    deallocate(model%layers(i)%v_cache)
                ! Note: Weight arrays (wq, wk, etc.) cleanup would go here
            end do
            deallocate(model%layers)
        end if

    end subroutine cleanup_llama_model

end module llama_model
