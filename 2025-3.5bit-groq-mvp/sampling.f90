! Text Generation Sampling Module
! Implements various sampling strategies for LLaMA text generation
! Pure Fortran 2023

module sampling
    use iso_fortran_env, only: int32, real32
    implicit none

    private
    public :: sample_greedy, sample_top_k, sample_top_p, sample_temperature
    public :: apply_temperature, softmax

contains

    !===========================================================================
    ! Apply temperature scaling to logits
    ! Higher temperature = more random, lower = more deterministic
    !===========================================================================
    pure subroutine apply_temperature(logits, temperature, scaled_logits, vocab_size)
        integer(int32), intent(in) :: vocab_size
        real(real32), intent(in) :: logits(vocab_size)
        real(real32), intent(in) :: temperature
        real(real32), intent(out) :: scaled_logits(vocab_size)

        if (temperature > 0.0) then
            scaled_logits = logits / temperature
        else
            scaled_logits = logits
        end if
    end subroutine apply_temperature

    !===========================================================================
    ! Softmax: Convert logits to probabilities
    !===========================================================================
    pure subroutine softmax(logits, probs, vocab_size)
        integer(int32), intent(in) :: vocab_size
        real(real32), intent(in) :: logits(vocab_size)
        real(real32), intent(out) :: probs(vocab_size)

        real(real32) :: max_logit, sum_exp
        integer(int32) :: i

        ! Numerical stability: subtract max
        max_logit = maxval(logits)

        ! Compute exp and sum
        sum_exp = 0.0
        do i = 1, vocab_size
            probs(i) = exp(logits(i) - max_logit)
            sum_exp = sum_exp + probs(i)
        end do

        ! Normalize
        if (sum_exp > 0.0) then
            probs = probs / sum_exp
        else
            probs = 1.0 / real(vocab_size, real32)
        end if
    end subroutine softmax

    !===========================================================================
    ! Greedy Sampling: Always pick the most likely token
    !===========================================================================
    function sample_greedy(logits, vocab_size) result(token_id)
        integer(int32), intent(in) :: vocab_size
        real(real32), intent(in) :: logits(vocab_size)
        integer(int32) :: token_id

        integer(int32) :: i, max_idx
        real(real32) :: max_val

        max_val = logits(1)
        max_idx = 1

        do i = 2, vocab_size
            if (logits(i) > max_val) then
                max_val = logits(i)
                max_idx = i
            end if
        end do

        token_id = max_idx
    end function sample_greedy

    !===========================================================================
    ! Temperature Sampling: Sample from temperature-scaled distribution
    !===========================================================================
    function sample_temperature(logits, vocab_size, temperature) result(token_id)
        integer(int32), intent(in) :: vocab_size
        real(real32), intent(in) :: logits(vocab_size)
        real(real32), intent(in) :: temperature
        integer(int32) :: token_id

        real(real32) :: scaled_logits(vocab_size)
        real(real32) :: probs(vocab_size)
        real(real32) :: rand_val, cumsum
        integer(int32) :: i

        ! Apply temperature scaling
        call apply_temperature(logits, temperature, scaled_logits, vocab_size)

        ! Convert to probabilities
        call softmax(scaled_logits, probs, vocab_size)

        ! Sample from distribution
        call random_number(rand_val)

        cumsum = 0.0
        token_id = vocab_size  ! Default to last token

        do i = 1, vocab_size
            cumsum = cumsum + probs(i)
            if (rand_val <= cumsum) then
                token_id = i
                exit
            end if
        end do
    end function sample_temperature

    !===========================================================================
    ! Top-K Sampling: Sample from top K most likely tokens
    !===========================================================================
    function sample_top_k(logits, vocab_size, k) result(token_id)
        integer(int32), intent(in) :: vocab_size, k
        real(real32), intent(in) :: logits(vocab_size)
        integer(int32) :: token_id

        real(real32) :: masked_logits(vocab_size)
        real(real32) :: probs(vocab_size)
        real(real32) :: kth_largest
        integer(int32) :: i, count
        real(real32) :: rand_val, cumsum

        ! Find k-th largest value (simple O(n*k) approach)
        masked_logits = logits
        kth_largest = -huge(1.0)

        ! Sort to find top-k threshold
        do count = 1, k
            kth_largest = maxval(masked_logits)
            do i = 1, vocab_size
                if (masked_logits(i) == kth_largest) then
                    masked_logits(i) = -huge(1.0)
                    exit
                end if
            end do
        end do

        ! Mask out tokens not in top-k
        do i = 1, vocab_size
            if (logits(i) < kth_largest) then
                masked_logits(i) = -huge(1.0)
            else
                masked_logits(i) = logits(i)
            end if
        end do

        ! Convert to probabilities
        call softmax(masked_logits, probs, vocab_size)

        ! Sample from distribution
        call random_number(rand_val)

        cumsum = 0.0
        token_id = vocab_size

        do i = 1, vocab_size
            if (probs(i) > 0.0) then
                cumsum = cumsum + probs(i)
                if (rand_val <= cumsum) then
                    token_id = i
                    exit
                end if
            end if
        end do
    end function sample_top_k

    !===========================================================================
    ! Top-P (Nucleus) Sampling: Sample from smallest set with cumulative prob > p
    !===========================================================================
    function sample_top_p(logits, vocab_size, p) result(token_id)
        integer(int32), intent(in) :: vocab_size
        real(real32), intent(in) :: logits(vocab_size), p
        integer(int32) :: token_id

        real(real32) :: probs(vocab_size)
        real(real32) :: sorted_probs(vocab_size)
        integer(int32) :: sorted_indices(vocab_size)
        real(real32) :: cumsum, threshold
        real(real32) :: masked_logits(vocab_size)
        integer(int32) :: i, j
        real(real32) :: rand_val, sample_cumsum

        ! Convert to probabilities
        call softmax(logits, probs, vocab_size)

        ! Simple selection sort to get sorted probabilities
        sorted_probs = probs
        do i = 1, vocab_size
            sorted_indices(i) = i
        end do

        ! Sort descending
        do i = 1, vocab_size - 1
            do j = i + 1, vocab_size
                if (sorted_probs(j) > sorted_probs(i)) then
                    ! Swap probabilities
                    threshold = sorted_probs(i)
                    sorted_probs(i) = sorted_probs(j)
                    sorted_probs(j) = threshold

                    ! Swap indices
                    token_id = sorted_indices(i)
                    sorted_indices(i) = sorted_indices(j)
                    sorted_indices(j) = token_id
                end if
            end do
        end do

        ! Find threshold: smallest cumulative prob > p
        cumsum = 0.0
        threshold = 0.0

        do i = 1, vocab_size
            cumsum = cumsum + sorted_probs(i)
            if (cumsum >= p) then
                threshold = sorted_probs(i)
                exit
            end if
        end do

        ! Mask tokens below threshold
        do i = 1, vocab_size
            if (probs(i) >= threshold) then
                masked_logits(i) = logits(i)
            else
                masked_logits(i) = -huge(1.0)
            end if
        end do

        ! Renormalize probabilities
        call softmax(masked_logits, probs, vocab_size)

        ! Sample from distribution
        call random_number(rand_val)

        sample_cumsum = 0.0
        token_id = vocab_size

        do i = 1, vocab_size
            if (probs(i) > 0.0) then
                sample_cumsum = sample_cumsum + probs(i)
                if (rand_val <= sample_cumsum) then
                    token_id = i
                    exit
                end if
            end if
        end do
    end function sample_top_p

end module sampling
