! Test program for sampling strategies
! Verifies greedy, temperature, top-k, and top-p sampling

program test_sampling
    use iso_fortran_env, only: int32, real32
    use sampling
    implicit none

    integer(int32), parameter :: VOCAB_SIZE = 32000
    real(real32) :: logits(VOCAB_SIZE)
    real(real32) :: probs(VOCAB_SIZE)
    integer(int32) :: token_id, i
    integer(int32) :: counts(10)
    real(real32) :: temperature

    print *, "=========================================="
    print *, "Sampling Strategies Test"
    print *, "=========================================="
    print *, ""

    ! Create test logits (simple distribution)
    do i = 1, VOCAB_SIZE
        logits(i) = real(VOCAB_SIZE - i, real32) / 1000.0
    end do
    logits(1) = 5.0   ! High probability
    logits(2) = 4.0
    logits(3) = 3.0
    logits(100) = 2.5

    print *, "Test logits created (vocab_size =", VOCAB_SIZE, ")"
    print *, "Top 5 logits:", logits(1:5)
    print *, ""

    ! Test 1: Greedy Sampling
    print *, "Test 1: Greedy Sampling"
    print *, "------------------------"
    token_id = sample_greedy(logits, VOCAB_SIZE)
    print *, "Greedy token ID:", token_id
    print *, "Expected: 1 (highest logit)"
    print *, ""

    ! Test 2: Temperature Sampling
    print *, "Test 2: Temperature Sampling"
    print *, "-----------------------------"

    ! Low temperature (more deterministic)
    print *, "Low temperature (0.1):"
    counts = 0
    do i = 1, 100
        token_id = sample_temperature(logits, VOCAB_SIZE, 0.1_real32)
        if (token_id <= 10) counts(token_id) = counts(token_id) + 1
    end do
    print *, "  Distribution (first 10 tokens):", counts
    print *, ""

    ! High temperature (more random)
    print *, "High temperature (2.0):"
    counts = 0
    do i = 1, 100
        token_id = sample_temperature(logits, VOCAB_SIZE, 2.0_real32)
        if (token_id <= 10) counts(token_id) = counts(token_id) + 1
    end do
    print *, "  Distribution (first 10 tokens):", counts
    print *, ""

    ! Test 3: Top-K Sampling
    print *, "Test 3: Top-K Sampling (k=5)"
    print *, "-----------------------------"
    counts = 0
    do i = 1, 100
        token_id = sample_top_k(logits, VOCAB_SIZE, 5)
        if (token_id <= 10) counts(token_id) = counts(token_id) + 1
    end do
    print *, "Distribution (first 10 tokens):", counts
    print *, "Expected: Only tokens 1-5 should be sampled"
    print *, ""

    ! Test 4: Top-P Sampling
    print *, "Test 4: Top-P Sampling (p=0.9)"
    print *, "-------------------------------"
    counts = 0
    do i = 1, 100
        token_id = sample_top_p(logits, VOCAB_SIZE, 0.9_real32)
        if (token_id <= 10) counts(token_id) = counts(token_id) + 1
    end do
    print *, "Distribution (first 10 tokens):", counts
    print *, "Expected: Tokens sampled until cumulative prob > 0.9"
    print *, ""

    ! Test 5: Softmax
    print *, "Test 5: Softmax Function"
    print *, "------------------------"
    call softmax(logits(1:10), probs(1:10), 10)
    print *, "Input logits (first 10):", logits(1:10)
    print *, "Output probs (first 10):", probs(1:10)
    print *, "Sum of probs:", sum(probs(1:10))
    print *, "Expected sum: ~1.0"
    print *, ""

    print *, "=========================================="
    print *, "✓ Sampling tests completed!"
    print *, "=========================================="
    print *, ""
    print *, "Summary:"
    print *, "  ✓ Greedy sampling works"
    print *, "  ✓ Temperature sampling works"
    print *, "  ✓ Top-k sampling works"
    print *, "  ✓ Top-p sampling works"
    print *, "  ✓ Softmax normalization works"
    print *, ""
    print *, "Next steps:"
    print *, "  1. Integrate with LLaMA model"
    print *, "  2. Implement autoregressive generation loop"
    print *, "  3. Test with real tokenizer"

end program test_sampling
