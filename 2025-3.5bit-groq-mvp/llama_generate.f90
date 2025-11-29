! LLaMA 70B End-to-End Text Generation
! Integrates: Tokenizer → 80-layer Model → Sampling → Detokenizer
! Pure Fortran 2023 with Python tokenizer wrapper

program llama_generate
    use iso_fortran_env, only: int32, real32
    use llama_model
    use sampling
    implicit none

    ! Configuration
    integer(int32), parameter :: VOCAB_SIZE = 32000
    integer(int32), parameter :: MAX_SEQ_LEN = 2048
    integer(int32), parameter :: MAX_NEW_TOKENS = 100
    real(real32), parameter :: TEMPERATURE = 0.8
    integer(int32), parameter :: TOP_K = 40
    real(real32), parameter :: TOP_P = 0.95

    ! Model and generation state
    type(LLaMAModel) :: model
    integer(int32), allocatable :: token_ids(:)
    integer(int32) :: prompt_len, total_len, i, next_token
    real(real32), allocatable :: logits_2d(:,:)  ! [seq_len, VOCAB_SIZE]
    real(real32), allocatable :: logits_1d(:)     ! [VOCAB_SIZE] for sampling
    character(len=1024) :: prompt, generated_text
    character(len=256) :: tokenizer_cmd
    integer :: ios

    ! Timing
    real(real32) :: start_time, end_time, elapsed_time, tokens_per_sec

    print *, "=========================================="
    print *, "LLaMA 70B Text Generation Pipeline"
    print *, "Pure Fortran 2023 - ASIC Optimized"
    print *, "=========================================="
    print *, ""

    ! Step 1: Get user prompt
    print *, "Enter prompt (or press Enter for default):"
    read(*, '(A)', iostat=ios) prompt
    if (ios /= 0 .or. len_trim(prompt) == 0) then
        prompt = "Explain quantum computing in simple terms:"
        print *, "Using default prompt: ", trim(prompt)
    end if
    print *, ""

    ! Step 2: Tokenize input using Python wrapper
    print *, "Tokenizing prompt..."
    call tokenize_text(trim(prompt), token_ids)
    prompt_len = size(token_ids)
    print *, "  Prompt tokens:", prompt_len
    print *, "  Token IDs:", token_ids(1:min(10, prompt_len))
    print *, ""

    ! Step 3: Initialize LLaMA model
    print *, "Initializing LLaMA 70B model..."
    call init_llama_model(model)
    print *, "  ✓ Model initialized (80 layers)"
    print *, ""

    ! Step 4: Load weights (if available)
    print *, "Loading weights..."
    call load_model_weights(model)
    print *, ""

    ! Step 5: Autoregressive generation loop
    print *, "Generating text (max", MAX_NEW_TOKENS, "tokens)..."
    print *, "==========================================="
    print *, trim(prompt), " "

    allocate(logits_1d(VOCAB_SIZE))
    total_len = prompt_len

    call cpu_time(start_time)

    do i = 1, MAX_NEW_TOKENS
        ! For KV caching: first pass processes all prompt tokens, subsequent passes only new token
        if (i == 1) then
            ! First pass: process entire prompt
            if (allocated(logits_2d)) deallocate(logits_2d)
            allocate(logits_2d(total_len, VOCAB_SIZE))
            call forward_llama(model, token_ids, logits_2d, total_len)
            logits_1d = logits_2d(total_len, :)
        else
            ! Subsequent passes: process only the last (newest) token
            ! KV cache handles attending to all previous tokens
            if (allocated(logits_2d)) deallocate(logits_2d)
            allocate(logits_2d(1, VOCAB_SIZE))
            call forward_llama(model, token_ids(total_len:total_len), logits_2d, 1)
            logits_1d = logits_2d(1, :)
        end if

        ! Sample next token using temperature + top-p
        if (TEMPERATURE > 0.0) then
            next_token = sample_top_p(logits_1d, VOCAB_SIZE, TOP_P)
        else
            next_token = sample_greedy(logits_1d, VOCAB_SIZE)
        end if

        ! Check for end-of-sequence token (2 for LLaMA)
        if (next_token == 2) then
            print *, ""
            print *, "[EOS]"
            exit
        end if

        ! Append new token to sequence
        total_len = total_len + 1
        if (total_len > MAX_SEQ_LEN) then
            print *, ""
            print *, "[MAX LENGTH REACHED]"
            exit
        end if

        ! Reallocate token_ids array to include new token
        token_ids = [token_ids, next_token]

        ! Detokenize and print (simple version - prints token ID)
        ! In production, call Python detokenizer here
        if (mod(i, 10) == 0) then
            write(*, '(A,I4,A,I4,A)', advance='no') " [", i, "/", MAX_NEW_TOKENS, "]"
        end if
    end do

    call cpu_time(end_time)
    elapsed_time = end_time - start_time
    tokens_per_sec = real(total_len - prompt_len, real32) / elapsed_time

    print *, ""
    print *, "==========================================="
    print *, ""
    print *, "Generation Statistics:"
    print *, "  Prompt tokens:", prompt_len
    print *, "  Generated tokens:", total_len - prompt_len
    print *, "  Total tokens:", total_len
    write(*, '(A,F8.2,A)') "  Time elapsed: ", elapsed_time, " seconds"
    write(*, '(A,F8.2,A)') "  Throughput: ", tokens_per_sec, " tokens/sec"
    write(*, '(A,F8.2,A)') "  Latency: ", 1000.0/tokens_per_sec, " ms/token"
    print *, ""

    ! Step 6: Detokenize full output
    print *, "Detokenizing output..."
    call detokenize_tokens(token_ids, generated_text)
    print *, ""
    print *, "Final output:"
    print *, "==========================================="
    print *, trim(generated_text)
    print *, "==========================================="

    ! Cleanup
    call cleanup_llama_model(model)
    if (allocated(token_ids)) deallocate(token_ids)
    if (allocated(logits_1d)) deallocate(logits_1d)
    if (allocated(logits_2d)) deallocate(logits_2d)

contains

    !===========================================================================
    ! Tokenize text using Python SentencePiece wrapper
    ! Writes prompt to temp file, calls tokenizer.py, reads token IDs back
    !===========================================================================
    subroutine tokenize_text(text, tokens)
        character(len=*), intent(in) :: text
        integer(int32), allocatable, intent(out) :: tokens(:)

        integer :: unit, n_tokens, i, ios
        character(len=256) :: cmd

        ! Write prompt to temp file
        open(newunit=unit, file='prompt.txt', status='replace', action='write')
        write(unit, '(A)') trim(text)
        close(unit)

        ! Call Python tokenizer
        cmd = 'python3 scripts/tokenizer.py encode prompt.txt tokens.bin'
        call execute_command_line(trim(cmd), exitstat=ios)

        if (ios /= 0) then
            print *, "  ⚠ Warning: Tokenizer failed, using placeholder tokens"
            ! Fallback: use simple placeholder tokens
            allocate(tokens(10))
            tokens = [(i, i=1,10)]
            return
        end if

        ! Read token IDs from binary file
        open(newunit=unit, file='tokens.bin', form='unformatted', &
             access='stream', status='old', action='read', iostat=ios)

        if (ios /= 0) then
            print *, "  ⚠ Warning: Could not read tokens, using placeholder"
            allocate(tokens(10))
            tokens = [(i, i=1,10)]
            return
        end if

        ! Read number of tokens
        read(unit, iostat=ios) n_tokens
        if (ios /= 0 .or. n_tokens <= 0 .or. n_tokens > 10000) then
            print *, "  ⚠ Warning: Invalid token count, using placeholder"
            close(unit)
            allocate(tokens(10))
            tokens = [(i, i=1,10)]
            return
        end if

        ! Read token IDs
        allocate(tokens(n_tokens))
        read(unit, iostat=ios) tokens
        close(unit)

        if (ios /= 0) then
            print *, "  ⚠ Warning: Token read failed, using placeholder"
            deallocate(tokens)
            allocate(tokens(10))
            tokens = [(i, i=1,10)]
        end if

    end subroutine tokenize_text

    !===========================================================================
    ! Detokenize tokens using Python SentencePiece wrapper
    !===========================================================================
    subroutine detokenize_tokens(tokens, text)
        integer(int32), intent(in) :: tokens(:)
        character(len=*), intent(out) :: text

        integer :: unit, i, ios
        character(len=256) :: cmd

        ! Write tokens to binary file
        open(newunit=unit, file='output_tokens.bin', form='unformatted', &
             access='stream', status='replace', action='write')
        write(unit) size(tokens)
        write(unit) tokens
        close(unit)

        ! Call Python detokenizer
        cmd = 'python3 scripts/tokenizer.py decode output_tokens.bin output.txt'
        call execute_command_line(trim(cmd), exitstat=ios)

        if (ios /= 0) then
            text = "[Detokenization failed]"
            return
        end if

        ! Read generated text
        open(newunit=unit, file='output.txt', status='old', action='read', iostat=ios)
        if (ios /= 0) then
            text = "[Could not read output]"
            return
        end if

        read(unit, '(A)', iostat=ios) text
        close(unit)

        if (ios /= 0) then
            text = "[Read error]"
        end if

    end subroutine detokenize_tokens

    !===========================================================================
    ! Load model weights from binary files (if available)
    !===========================================================================
    subroutine load_model_weights(model)
        type(LLaMAModel), intent(inout) :: model
        integer :: layer_idx
        character(len=256) :: weight_file
        logical :: file_exists

        ! Check if test weights exist
        inquire(file='test_weights_layer0.bin', exist=file_exists)

        if (.not. file_exists) then
            print *, "  ⚠ No weight files found - using random initialization"
            print *, "  Run 'make gen-weights' to generate test weights"
            return
        end if

        print *, "  Loading test weights..."
        ! For now, just load layer 0 as a test
        ! Full implementation would load all 80 layers
        ! call load_layer_weights(model%layers(1), 'test_weights_layer0.bin', 0)
        print *, "  ⚠ Weight loading temporarily disabled to avoid INT4 segfault"
        print *, "  Model will use random/placeholder weights"

    end subroutine load_model_weights

end program llama_generate
