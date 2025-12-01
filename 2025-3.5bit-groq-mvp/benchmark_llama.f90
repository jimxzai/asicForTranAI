! Comprehensive Performance Benchmark for LLaMA 70B
! Measures: throughput, latency, per-layer timing, memory usage
! Target: 3100+ tokens/sec on Groq LPU

program benchmark_llama
    use iso_fortran_env, only: int32, real32
    use llama_model
    use transformer_layer
    implicit none

    type(LLaMAModel) :: model
    integer(int32), parameter :: NUM_WARMUP_ITERS = 3
    integer(int32), parameter :: NUM_BENCH_ITERS = 10
    integer(int32), parameter :: BATCH_SIZE = 1
    integer(int32), parameter :: SEQ_LEN = 128  ! Typical generation length

    integer(int32) :: token_ids(SEQ_LEN)
    real(real32), allocatable :: logits(:,:)
    real(real32) :: start_time, end_time, total_time
    real(real32) :: warmup_time, bench_time
    real(real32) :: avg_latency, throughput, tokens_per_sec
    integer :: i, j, iter
    integer(int32) :: num_layers_total, vocab_size_total

    print *, "=========================================="
    print *, "LLaMA 70B Performance Benchmark"
    print *, "Pure Fortran 2023 - ASIC Optimized"
    print *, "=========================================="
    print *, ""

    ! Initialize model
    print *, "Initializing model..."
    call init_llama_model(model)

    ! Get configuration from initialized model
    num_layers_total = model%num_layers
    vocab_size_total = 32000  ! VOCAB_SIZE from llama_model

    print *, "  ✓ Model initialized"
    print *, ""
    print *, "Configuration:"
    print *, "  Layers:      ", num_layers_total
    print *, "  Hidden dim:  ", HIDDEN_DIM
    print *, "  Vocab size:  ", vocab_size_total
    print *, "  Batch size:  ", BATCH_SIZE
    print *, "  Seq length:  ", SEQ_LEN
    print *, "  Warmup iters:", NUM_WARMUP_ITERS
    print *, "  Bench iters: ", NUM_BENCH_ITERS
    print *, ""

    ! Create test input tokens
    do i = 1, SEQ_LEN
        token_ids(i) = mod(i, vocab_size_total) + 1
    end do

    ! Allocate output
    allocate(logits(SEQ_LEN, vocab_size_total))

    ! ===== WARMUP PHASE =====
    print *, "=========================================="
    print *, "WARMUP PHASE"
    print *, "=========================================="
    print *, "Running", NUM_WARMUP_ITERS, "warmup iterations..."

    call cpu_time(start_time)
    do iter = 1, NUM_WARMUP_ITERS
        call forward_llama(model, token_ids, logits, SEQ_LEN)
        write(*, '(A,I0,A,I0)', advance='no') "  Warmup ", iter, "/", NUM_WARMUP_ITERS
        if (iter < NUM_WARMUP_ITERS) write(*, '(A)') ""
    end do
    call cpu_time(end_time)
    warmup_time = end_time - start_time

    print *, ""
    print *, "✓ Warmup complete"
    write(*, '(A,F8.2,A)') "  Total warmup time: ", warmup_time, " seconds"
    print *, ""

    ! ===== BENCHMARK PHASE =====
    print *, "=========================================="
    print *, "BENCHMARK PHASE"
    print *, "=========================================="
    print *, "Running", NUM_BENCH_ITERS, "benchmark iterations..."
    print *, ""

    total_time = 0.0

    do iter = 1, NUM_BENCH_ITERS
        call cpu_time(start_time)
        call forward_llama(model, token_ids, logits, SEQ_LEN)
        call cpu_time(end_time)

        total_time = total_time + (end_time - start_time)

        write(*, '(A,I2,A,I2,A,F8.3,A)') &
            "  Iteration ", iter, "/", NUM_BENCH_ITERS, &
            ": ", (end_time - start_time), " sec"
    end do

    print *, ""
    print *, "✓ Benchmark complete"
    print *, ""

    ! ===== RESULTS =====
    print *, "=========================================="
    print *, "BENCHMARK RESULTS"
    print *, "=========================================="
    print *, ""

    avg_latency = total_time / real(NUM_BENCH_ITERS, real32)
    tokens_per_sec = real(SEQ_LEN, real32) / avg_latency
    throughput = real(SEQ_LEN * NUM_BENCH_ITERS, real32) / total_time

    print *, "Overall Performance:"
    write(*, '(A,F10.3,A)') "  Average latency:    ", avg_latency, " sec/forward"
    write(*, '(A,F10.3,A)') "  Throughput:         ", throughput, " tokens/sec"
    write(*, '(A,F10.3,A)') "  Per-token latency:  ", avg_latency*1000.0/SEQ_LEN, " ms/token"
    write(*, '(A,F10.3,A)') "  Total time:         ", total_time, " sec"
    print *, ""

    print *, "Per-sequence metrics:"
    write(*, '(A,I0,A)') "  Tokens per sequence: ", SEQ_LEN
    write(*, '(A,F10.3,A)') "  Time per sequence:   ", avg_latency, " sec"
    write(*, '(A,F10.3,A)') "  Sequences per sec:   ", 1.0/avg_latency, " seq/sec"
    print *, ""

    print *, "Model throughput comparison:"
    write(*, '(A,F10.1,A)') "  Current:            ", throughput, " tok/s"
    write(*, '(A,F10.1,A)') "  Groq LPU Target:    ", 3100.0, " tok/s"
    write(*, '(A,F10.1,A)') "  Speedup needed:     ", 3100.0/throughput, "x"
    print *, ""

    ! Memory estimates
    print *, "Memory Usage (estimates):"
    write(*, '(A,F10.1,A)') "  Model params:       ", 70.0, " GB (70B × FP32)"
    write(*, '(A,F10.1,A)') "  INT4 quantized:     ", 8.75, " GB (70B × 4-bit)"
    write(*, '(A,F10.1,A)') "  KV cache (2048):    ", 2.56, " GB (per sequence)"
    write(*, '(A,F10.1,A)') "  Activations:        ", 0.25, " GB (working memory)"
    print *, ""

    print *, "=========================================="
    print *, "PERFORMANCE BREAKDOWN"
    print *, "=========================================="
    print *, ""

    ! Estimate per-layer time (total_time / num_layers_total / NUM_BENCH_ITERS)
    write(*, '(A,F10.3,A)') "  Avg time per layer: ", &
        avg_latency/real(num_layers_total, real32)*1000.0, " ms"
    write(*, '(A,I0,A,F10.3,A)') "  ", num_layers_total, " layers total:    ", &
        avg_latency, " sec"
    print *, ""

    ! Bottleneck analysis
    print *, "Expected bottlenecks:"
    print *, "  1. INT4 matmul (naive implementation)"
    print *, "  2. Attention computation (no FlashAttention)"
    print *, "  3. FFN operations (no BLAS acceleration)"
    print *, ""

    print *, "Optimization opportunities:"
    print *, "  • Replace naive matmul with optimized BLAS"
    print *, "  • Implement SIMD vectorization for INT4"
    print *, "  • Use FlashAttention for attention computation"
    print *, "  • Parallelize across layers (pipeline parallelism)"
    print *, "  • Optimize memory access patterns"
    print *, ""

    ! Cleanup
    call cleanup_llama_model(model)
    deallocate(logits)

    print *, "=========================================="
    print *, "Benchmark complete!"
    print *, "=========================================="

end program benchmark_llama
