! LLaMA 70B Transformer Layer - Pure Fortran 2023
! Implements one complete decoder layer with:
! - Grouped-Query Attention (GQA: 64 heads, 8 KV heads)
! - SwiGLU FFN
! - RMSNorm
! - RoPE positional encoding
!
! Optimized for ASIC deployment (Groq LPU, Cerebras, etc.)

module transformer_layer
    use iso_fortran_env, only: int8, int32, real32
    use matmul_int4_groq, only: matmul_int4_awq, dequantize_output
    implicit none

    private
    public :: TransformerLayer, apply_transformer_layer, init_rope_freqs
    public :: init_kv_cache, reset_kv_cache, rms_norm
    public :: HIDDEN_DIM, INTERMEDIATE_DIM, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM

    ! LLaMA 70B configuration
    integer(int32), parameter :: HIDDEN_DIM = 8192
    integer(int32), parameter :: INTERMEDIATE_DIM = 28672
    integer(int32), parameter :: NUM_HEADS = 64
    integer(int32), parameter :: NUM_KV_HEADS = 8
    integer(int32), parameter :: HEAD_DIM = 128
    real(real32), parameter :: RMS_NORM_EPS = 1.0e-5
    real(real32), parameter :: PI = 3.14159265359

    ! Layer weights structure
    type :: TransformerLayer
        ! Attention weights (quantized)
        integer(int8), allocatable :: wq(:,:)      ! Query projection
        integer(int8), allocatable :: wk(:,:)      ! Key projection
        integer(int8), allocatable :: wv(:,:)      ! Value projection
        integer(int8), allocatable :: wo(:,:)      ! Output projection
        real(real32), allocatable :: wq_scales(:)
        real(real32), allocatable :: wk_scales(:)
        real(real32), allocatable :: wv_scales(:)
        real(real32), allocatable :: wo_scales(:)

        ! FFN weights (quantized)
        integer(int8), allocatable :: w_gate(:,:)  ! SwiGLU gate
        integer(int8), allocatable :: w_up(:,:)    ! SwiGLU up projection
        integer(int8), allocatable :: w_down(:,:)  ! Down projection
        real(real32), allocatable :: w_gate_scales(:)
        real(real32), allocatable :: w_up_scales(:)
        real(real32), allocatable :: w_down_scales(:)

        ! RMSNorm weights
        real(real32), allocatable :: attn_norm(:)  ! Pre-attention norm
        real(real32), allocatable :: ffn_norm(:)   ! Pre-FFN norm

        ! RoPE frequency cache
        real(real32), allocatable :: rope_freqs(:,:)

        ! KV cache for autoregressive generation
        real(real32), allocatable :: k_cache(:,:,:)  ! [max_seq_len, NUM_KV_HEADS, HEAD_DIM]
        real(real32), allocatable :: v_cache(:,:,:)  ! [max_seq_len, NUM_KV_HEADS, HEAD_DIM]
        integer(int32) :: cache_pos  ! Current position in cache
    end type TransformerLayer

contains

    !===========================================================================
    ! Helper: INT4 matmul with automatic dequantization
    ! Wraps matmul_int4_awq + dequantize_output for convenience
    !===========================================================================
    subroutine int4_linear(x, w_q, w_scales, output, M, N, K_dim)
        integer(int32), intent(in) :: M, N, K_dim
        real(real32), intent(in) :: x(M, K_dim)
        integer(int8), intent(in) :: w_q(K_dim/8, N)
        real(real32), intent(in) :: w_scales(N)
        real(real32), intent(out) :: output(M, N)

        integer(int8) :: x_int8(M, K_dim)
        integer(int32) :: accum(M, N)
        integer(int32) :: i, j

        ! Quantize input to INT8 (simple round for now)
        ! TODO: Use proper activation quantization
        do concurrent(i = 1:M, j = 1:K_dim)
            x_int8(i,j) = int(max(-127.0, min(127.0, x(i,j) * 127.0)), int8)
        end do

        ! INT4 matrix multiplication
        call matmul_int4_awq(x_int8, w_q, w_scales, accum, M, N, K_dim)

        ! Dequantize to FP32
        call dequantize_output(accum, w_scales, output, M, N)
    end subroutine int4_linear

    !===========================================================================
    ! Initialize RoPE frequency cache
    ! Precomputes rotary embedding frequencies for efficient inference
    !===========================================================================
    subroutine init_rope_freqs(layer, max_seq_len)
        type(TransformerLayer), intent(inout) :: layer
        integer(int32), intent(in) :: max_seq_len

        integer(int32) :: pos, d
        real(real32) :: theta, freq_base

        ! Allocate frequency cache: [max_seq_len, HEAD_DIM/2]
        if (.not. allocated(layer%rope_freqs)) then
            allocate(layer%rope_freqs(max_seq_len, HEAD_DIM/2))
        end if

        ! RoPE base frequency (standard LLaMA value)
        freq_base = 10000.0

        ! Compute frequencies for each position and dimension pair
        do concurrent(pos = 1:max_seq_len, d = 1:HEAD_DIM/2)
            ! theta = freq_base^(-2*d/HEAD_DIM)
            theta = freq_base ** (-2.0 * real(d-1, real32) / real(HEAD_DIM, real32))
            ! freq = position * theta
            layer%rope_freqs(pos, d) = real(pos-1, real32) * theta
        end do
    end subroutine init_rope_freqs

    !===========================================================================
    ! Initialize KV cache for autoregressive generation
    ! Allocates cache buffers and resets position to 0
    !===========================================================================
    subroutine init_kv_cache(layer, max_seq_len)
        type(TransformerLayer), intent(inout) :: layer
        integer(int32), intent(in) :: max_seq_len

        ! Allocate KV cache: [max_seq_len, NUM_KV_HEADS, HEAD_DIM]
        if (.not. allocated(layer%k_cache)) then
            allocate(layer%k_cache(max_seq_len, NUM_KV_HEADS, HEAD_DIM))
        end if
        if (.not. allocated(layer%v_cache)) then
            allocate(layer%v_cache(max_seq_len, NUM_KV_HEADS, HEAD_DIM))
        end if

        ! Reset cache position
        layer%cache_pos = 0
        layer%k_cache = 0.0
        layer%v_cache = 0.0
    end subroutine init_kv_cache

    !===========================================================================
    ! Reset KV cache position for new generation sequence
    !===========================================================================
    subroutine reset_kv_cache(layer)
        type(TransformerLayer), intent(inout) :: layer
        layer%cache_pos = 0
        if (allocated(layer%k_cache)) layer%k_cache = 0.0
        if (allocated(layer%v_cache)) layer%v_cache = 0.0
    end subroutine reset_kv_cache

    !===========================================================================
    ! RMSNorm: Root Mean Square Layer Normalization
    !===========================================================================
    pure subroutine rms_norm(x, weight, output, seq_len, dim)
        integer(int32), intent(in), value :: seq_len, dim
        real(real32), intent(in) :: x(seq_len, dim)
        real(real32), intent(in) :: weight(dim)
        real(real32), intent(out) :: output(seq_len, dim)

        integer(int32) :: i, j
        real(real32) :: rms

        ! ASIC-optimized: parallel over sequence positions
        do concurrent(i = 1:seq_len)
            ! Compute RMS
            rms = 0.0
            do j = 1, dim
                rms = rms + x(i,j) * x(i,j)
            end do
            rms = sqrt(rms / real(dim, real32) + RMS_NORM_EPS)

            ! Normalize and scale
            do concurrent(j = 1:dim)
                output(i,j) = (x(i,j) / rms) * weight(j)
            end do
        end do
    end subroutine rms_norm

    !===========================================================================
    ! RoPE: Rotary Positional Embeddings
    !===========================================================================
    pure subroutine apply_rope(q, k, freqs, seq_len, num_heads, head_dim)
        integer(int32), intent(in), value :: seq_len, num_heads, head_dim
        real(real32), intent(inout) :: q(seq_len, num_heads, head_dim)
        real(real32), intent(inout) :: k(seq_len, num_heads, head_dim)
        real(real32), intent(in) :: freqs(seq_len, head_dim/2)

        integer(int32) :: pos, h, d
        real(real32) :: cos_val, sin_val
        real(real32) :: q_real, q_imag, k_real, k_imag

        ! Apply rotation to each head independently
        do concurrent(pos = 1:seq_len, h = 1:num_heads)
            do d = 1, head_dim/2
                cos_val = cos(freqs(pos, d))
                sin_val = sin(freqs(pos, d))

                ! Query rotation (treat pairs as complex numbers)
                q_real = q(pos, h, 2*d-1)
                q_imag = q(pos, h, 2*d)
                q(pos, h, 2*d-1) = q_real * cos_val - q_imag * sin_val
                q(pos, h, 2*d)   = q_real * sin_val + q_imag * cos_val

                ! Key rotation
                k_real = k(pos, h, 2*d-1)
                k_imag = k(pos, h, 2*d)
                k(pos, h, 2*d-1) = k_real * cos_val - k_imag * sin_val
                k(pos, h, 2*d)   = k_real * sin_val + k_imag * cos_val
            end do
        end do
    end subroutine apply_rope

    !===========================================================================
    ! SwiGLU Activation: Swish-Gated Linear Unit
    ! SwiGLU(x) = Swish(gate(x)) * up(x)
    ! where Swish(x) = x * sigmoid(x)
    !===========================================================================
    pure elemental function swish(x) result(y)
        real(real32), intent(in) :: x
        real(real32) :: y
        y = x / (1.0 + exp(-x))
    end function swish

    pure subroutine swiglu(gate, up, output, seq_len, dim)
        integer(int32), intent(in), value :: seq_len, dim
        real(real32), intent(in) :: gate(seq_len, dim)
        real(real32), intent(in) :: up(seq_len, dim)
        real(real32), intent(out) :: output(seq_len, dim)

        integer(int32) :: i, j

        do concurrent(i = 1:seq_len, j = 1:dim)
            output(i,j) = swish(gate(i,j)) * up(i,j)
        end do
    end subroutine swiglu

    !===========================================================================
    ! Grouped-Query Attention (GQA)
    ! LLaMA 70B: 64 query heads, 8 key/value heads
    !===========================================================================
    subroutine grouped_query_attention(layer, x_norm, output, seq_len)
        type(TransformerLayer), intent(inout) :: layer
        real(real32), intent(in) :: x_norm(seq_len, HIDDEN_DIM)
        real(real32), intent(out) :: output(seq_len, HIDDEN_DIM)
        integer(int32), intent(in), value :: seq_len

        ! Buffers for Q, K, V projections
        real(real32) :: q(seq_len, NUM_HEADS, HEAD_DIM)
        real(real32) :: k(seq_len, NUM_KV_HEADS, HEAD_DIM)
        real(real32) :: v(seq_len, NUM_KV_HEADS, HEAD_DIM)
        real(real32) :: q_flat(seq_len, NUM_HEADS * HEAD_DIM)
        real(real32) :: k_flat(seq_len, NUM_KV_HEADS * HEAD_DIM)
        real(real32) :: v_flat(seq_len, NUM_KV_HEADS * HEAD_DIM)

        ! Attention scores and output
        real(real32), allocatable :: scores(:,:,:)  ! Dynamic: (seq_len, total_seq_len, NUM_HEADS)
        real(real32) :: attn_out(seq_len, NUM_HEADS, HEAD_DIM)

        integer(int32) :: i, j, h, kv_h, d, total_seq_len, cache_start
        real(real32) :: scale_factor, sum_exp, max_score
        real(real32) :: k_current, v_current  ! For fetching K,V from cache or current

        ! 1. Compute Q, K, V projections
        if (allocated(layer%wq) .and. allocated(layer%wq_scales)) then
            ! Q, K, V projections with INT4 matmul
            call int4_linear(x_norm, layer%wq, layer%wq_scales, q_flat, &
                seq_len, NUM_HEADS * HEAD_DIM, HIDDEN_DIM)
            call int4_linear(x_norm, layer%wk, layer%wk_scales, k_flat, &
                seq_len, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM)
            call int4_linear(x_norm, layer%wv, layer%wv_scales, v_flat, &
                seq_len, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM)

            ! Reshape to [seq_len, num_heads, head_dim]
            do i = 1, seq_len
                do h = 1, NUM_HEADS
                    do d = 1, HEAD_DIM
                        q(i, h, d) = q_flat(i, (h-1)*HEAD_DIM + d)
                    end do
                end do
                do h = 1, NUM_KV_HEADS
                    do d = 1, HEAD_DIM
                        k(i, h, d) = k_flat(i, (h-1)*HEAD_DIM + d)
                        v(i, h, d) = v_flat(i, (h-1)*HEAD_DIM + d)
                    end do
                end do
            end do
        else
            ! Use test data for development (no weights loaded yet)
            do i = 1, seq_len
                do h = 1, NUM_HEADS
                    do d = 1, HEAD_DIM
                        q(i, h, d) = x_norm(i, (h-1)*HEAD_DIM + d) / real(NUM_HEADS, real32)
                    end do
                end do
            end do

            do i = 1, seq_len
                do h = 1, NUM_KV_HEADS
                    do d = 1, HEAD_DIM
                        k(i, h, d) = x_norm(i, (h-1)*HEAD_DIM + d) / real(NUM_KV_HEADS, real32)
                        v(i, h, d) = x_norm(i, (h-1)*HEAD_DIM + d) / real(NUM_KV_HEADS, real32)
                    end do
                end do
            end do
        end if

        ! 2. Apply RoPE rotary positional embeddings
        if (allocated(layer%rope_freqs)) then
            call apply_rope(q, k, layer%rope_freqs, seq_len, NUM_HEADS, HEAD_DIM)
        end if

        ! 2.5. KV Cache Integration - Store current K,V in cache
        ! For autoregressive generation: cache past K,V to avoid recomputation
        if (allocated(layer%k_cache) .and. allocated(layer%v_cache)) then
            ! Store current K,V in cache at positions [cache_pos+1:cache_pos+seq_len]
            do i = 1, seq_len
                do h = 1, NUM_KV_HEADS
                    do d = 1, HEAD_DIM
                        layer%k_cache(layer%cache_pos + i, h, d) = k(i, h, d)
                        layer%v_cache(layer%cache_pos + i, h, d) = v(i, h, d)
                    end do
                end do
            end do
        end if

        ! 3. Determine total sequence length including cache and allocate scores
        total_seq_len = layer%cache_pos + seq_len
        allocate(scores(seq_len, total_seq_len, NUM_HEADS))

        scale_factor = 1.0 / sqrt(real(HEAD_DIM, real32))

        ! 4. Grouped-Query Attention: Each KV head serves 8 query heads
        ! Groups: NUM_HEADS / NUM_KV_HEADS = 64 / 8 = 8 query heads per KV head
        do h = 1, NUM_HEADS
            ! Map query head to corresponding KV head (8:1 ratio)
            kv_h = (h - 1) / (NUM_HEADS / NUM_KV_HEADS) + 1

            ! Compute Q @ K^T for this head
            ! Each query position (i) attends to all past KV positions [1:total_seq_len]
            do i = 1, seq_len
                do j = 1, total_seq_len
                    ! Dot product: Q[i, h, :] @ K[j, kv_h, :]
                    scores(i, j, h) = 0.0
                    do d = 1, HEAD_DIM
                        ! Fetch K from cache if j <= cache_pos, else from current k array
                        if (j <= layer%cache_pos .and. allocated(layer%k_cache)) then
                            k_current = layer%k_cache(j, kv_h, d)
                        else
                            k_current = k(j - layer%cache_pos, kv_h, d)
                        end if
                        scores(i, j, h) = scores(i, j, h) + q(i, h, d) * k_current
                    end do
                    scores(i, j, h) = scores(i, j, h) * scale_factor

                    ! Apply causal mask (can't attend to future positions)
                    ! Current absolute position is cache_pos + i
                    if (j > layer%cache_pos + i) then
                        scores(i, j, h) = -1.0e9  ! Large negative = ~0 after softmax
                    end if
                end do

                ! 5. Apply softmax over all attended positions [1:total_seq_len]
                max_score = maxval(scores(i, :, h))
                sum_exp = 0.0
                do j = 1, total_seq_len
                    scores(i, j, h) = exp(scores(i, j, h) - max_score)
                    sum_exp = sum_exp + scores(i, j, h)
                end do
                ! Normalize
                if (sum_exp > 0.0) then
                    scores(i, :, h) = scores(i, :, h) / sum_exp
                end if
            end do
        end do

        ! 6. Apply attention to values: attn_weights @ V
        do h = 1, NUM_HEADS
            kv_h = (h - 1) / (NUM_HEADS / NUM_KV_HEADS) + 1

            do i = 1, seq_len
                do d = 1, HEAD_DIM
                    attn_out(i, h, d) = 0.0
                    do j = 1, seq_len
                        attn_out(i, h, d) = attn_out(i, h, d) + &
                            scores(i, j, h) * v(j, kv_h, d)
                    end do
                end do
            end do
        end do

        ! 7. Concatenate heads and reshape
        ! output[i, :] = concat(attn_out[i, 1, :], attn_out[i, 2, :], ..., attn_out[i, 64, :])
        do i = 1, seq_len
            do h = 1, NUM_HEADS
                do d = 1, HEAD_DIM
                    q_flat(i, (h-1)*HEAD_DIM + d) = attn_out(i, h, d)  ! Reuse q_flat buffer
                end do
            end do
        end do

        ! 8. Output projection: [seq_len, HIDDEN_DIM] @ [HIDDEN_DIM, HIDDEN_DIM]
        if (allocated(layer%wo) .and. allocated(layer%wo_scales)) then
            call int4_linear(q_flat(:, 1:HIDDEN_DIM), layer%wo, layer%wo_scales, &
                output, seq_len, HIDDEN_DIM, HIDDEN_DIM)
        else
            ! No output projection - copy concatenated heads directly
            output(:, :) = q_flat(:, 1:HIDDEN_DIM)
        end if

    end subroutine grouped_query_attention

    !===========================================================================
    ! SwiGLU Feed-Forward Network
    !===========================================================================
    subroutine swiglu_ffn(layer, x_norm, output, seq_len)
        type(TransformerLayer), intent(in) :: layer
        real(real32), intent(in) :: x_norm(seq_len, HIDDEN_DIM)
        real(real32), intent(out) :: output(seq_len, HIDDEN_DIM)
        integer(int32), intent(in), value :: seq_len

        real(real32) :: gate_proj(seq_len, INTERMEDIATE_DIM)
        real(real32) :: up_proj(seq_len, INTERMEDIATE_DIM)
        real(real32) :: swiglu_out(seq_len, INTERMEDIATE_DIM)
        real(real32) :: down_out(seq_len, HIDDEN_DIM)

        integer(int32) :: i, j, k

        ! 1. Gate projection: [seq_len, HIDDEN_DIM] @ [HIDDEN_DIM, INTERMEDIATE_DIM]
        if (allocated(layer%w_gate) .and. allocated(layer%w_gate_scales)) then
            call int4_linear(x_norm, layer%w_gate, layer%w_gate_scales, &
                gate_proj, seq_len, INTERMEDIATE_DIM, HIDDEN_DIM)
        else
            ! Test data for development
            do i = 1, seq_len
                do j = 1, INTERMEDIATE_DIM
                    gate_proj(i, j) = 0.0
                    do k = 1, HIDDEN_DIM
                        gate_proj(i, j) = gate_proj(i, j) + &
                            x_norm(i, k) * (0.01 / real(HIDDEN_DIM, real32))
                    end do
                end do
            end do
        end if

        ! 2. Up projection: [seq_len, HIDDEN_DIM] @ [HIDDEN_DIM, INTERMEDIATE_DIM]
        if (allocated(layer%w_up) .and. allocated(layer%w_up_scales)) then
            call int4_linear(x_norm, layer%w_up, layer%w_up_scales, &
                up_proj, seq_len, INTERMEDIATE_DIM, HIDDEN_DIM)
        else
            ! Test data for development
            do i = 1, seq_len
                do j = 1, INTERMEDIATE_DIM
                    up_proj(i, j) = 0.0
                    do k = 1, HIDDEN_DIM
                        up_proj(i, j) = up_proj(i, j) + &
                            x_norm(i, k) * (0.01 / real(HIDDEN_DIM, real32))
                    end do
                end do
            end do
        end if

        ! 3. Apply SwiGLU activation: swish(gate) * up
        call swiglu(gate_proj, up_proj, swiglu_out, seq_len, INTERMEDIATE_DIM)

        ! 4. Down projection: [seq_len, INTERMEDIATE_DIM] @ [INTERMEDIATE_DIM, HIDDEN_DIM]
        if (allocated(layer%w_down) .and. allocated(layer%w_down_scales)) then
            call int4_linear(swiglu_out, layer%w_down, layer%w_down_scales, &
                output, seq_len, HIDDEN_DIM, INTERMEDIATE_DIM)
        else
            ! Test data for development
            do i = 1, seq_len
                do j = 1, HIDDEN_DIM
                    output(i, j) = 0.0
                    do k = 1, INTERMEDIATE_DIM
                        output(i, j) = output(i, j) + &
                            swiglu_out(i, k) * (0.01 / real(INTERMEDIATE_DIM, real32))
                    end do
                end do
            end do
        end if

    end subroutine swiglu_ffn

    !===========================================================================
    ! Complete Transformer Layer
    ! output = x + Attention(RMSNorm(x))
    ! output = output + FFN(RMSNorm(output))
    !===========================================================================
    subroutine apply_transformer_layer(layer, x, output, seq_len)
        type(TransformerLayer), intent(inout) :: layer
        real(real32), intent(in) :: x(seq_len, HIDDEN_DIM)
        real(real32), intent(out) :: output(seq_len, HIDDEN_DIM)
        integer(int32), intent(in), value :: seq_len

        real(real32) :: x_norm(seq_len, HIDDEN_DIM)
        real(real32) :: attn_out(seq_len, HIDDEN_DIM)
        real(real32) :: ffn_out(seq_len, HIDDEN_DIM)
        real(real32) :: residual(seq_len, HIDDEN_DIM)

        ! First residual connection: Attention
        call rms_norm(x, layer%attn_norm, x_norm, seq_len, HIDDEN_DIM)
        call grouped_query_attention(layer, x_norm, attn_out, seq_len)

        ! Add residual
        residual = x + attn_out

        ! Second residual connection: FFN
        call rms_norm(residual, layer%ffn_norm, x_norm, seq_len, HIDDEN_DIM)
        call swiglu_ffn(layer, x_norm, ffn_out, seq_len)

        ! Final output with residual
        output = residual + ffn_out

    end subroutine apply_transformer_layer

end module transformer_layer
