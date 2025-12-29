! cuBLAS GPU-Accelerated INT4 MatMul - Target: 100-500× speedup
! Uses NVIDIA cuBLAS for hardware-optimized GPU matrix multiplication
! Fortran-C interop with CUDA cuBLAS library
!
! Performance on RTX 2080 Ti (4352 CUDA cores):
!   Expected: 0.05-0.1 ms per 8192×8192 matmul (vs 6.99 ms CPU)
!   Speedup: 70-140× vs SIMD+OpenMP
!   Throughput: ~125-250 tokens/sec for 70B model

module matmul_cublas
    use iso_fortran_env, only: int8, int32, real32
    use iso_c_binding
    implicit none

    private
    public :: matmul_int4_cublas, matmul_int4_cublas_cached
    public :: cublas_init, cublas_shutdown
    public :: dequantize_weights_gpu, allocate_gpu_memory

    ! cuBLAS handle (opaque pointer)
    type(c_ptr) :: cublas_handle = c_null_ptr

    ! GPU memory pointers
    type(c_ptr) :: d_A_fp32 = c_null_ptr       ! Device activations
    type(c_ptr) :: d_W_fp32 = c_null_ptr       ! Device weights
    type(c_ptr) :: d_Out = c_null_ptr          ! Device output

    ! cuBLAS C interface
    interface
        ! cublasCreate
        function cublasCreate_v2(handle) bind(c, name='cublasCreate_v2')
            use iso_c_binding
            type(c_ptr) :: handle
            integer(c_int) :: cublasCreate_v2
        end function

        ! cublasDestroy
        function cublasDestroy_v2(handle) bind(c, name='cublasDestroy_v2')
            use iso_c_binding
            type(c_ptr), value :: handle
            integer(c_int) :: cublasDestroy_v2
        end function

        ! cublasSgemm - Single precision matrix multiply
        function cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, &
                                A, lda, B, ldb, beta, C, ldc) bind(c, name='cublasSgemm_v2')
            use iso_c_binding
            type(c_ptr), value :: handle
            integer(c_int), value :: transa, transb
            integer(c_int), value :: m, n, k
            real(c_float) :: alpha, beta
            type(c_ptr), value :: A, B, C
            integer(c_int), value :: lda, ldb, ldc
            integer(c_int) :: cublasSgemm_v2
        end function
    end interface

    ! CUDA memory management interface
    interface
        ! cudaMalloc
        function cudaMalloc(devPtr, size) bind(c, name='cudaMalloc')
            use iso_c_binding
            type(c_ptr) :: devPtr
            integer(c_size_t), value :: size
            integer(c_int) :: cudaMalloc
        end function

        ! cudaFree
        function cudaFree(devPtr) bind(c, name='cudaFree')
            use iso_c_binding
            type(c_ptr), value :: devPtr
            integer(c_int) :: cudaFree
        end function

        ! cudaMemcpy
        function cudaMemcpy(dst, src, count, kind) bind(c, name='cudaMemcpy')
            use iso_c_binding
            type(c_ptr), value :: dst, src
            integer(c_size_t), value :: count
            integer(c_int), value :: kind
            integer(c_int) :: cudaMemcpy
        end function

        ! cudaDeviceSynchronize
        function cudaDeviceSynchronize() bind(c, name='cudaDeviceSynchronize')
            use iso_c_binding
            integer(c_int) :: cudaDeviceSynchronize
        end function
    end interface

    ! CUDA memory copy kinds
    integer(c_int), parameter :: cudaMemcpyHostToDevice = 1
    integer(c_int), parameter :: cudaMemcpyDeviceToHost = 2
    integer(c_int), parameter :: cudaMemcpyDeviceToDevice = 3

    ! cuBLAS operation codes
    integer(c_int), parameter :: CUBLAS_OP_N = 0  ! No transpose
    integer(c_int), parameter :: CUBLAS_OP_T = 1  ! Transpose
    integer(c_int), parameter :: CUBLAS_OP_C = 2  ! Conjugate transpose

contains

    !> Initialize cuBLAS library
    !> Call once at program start
    subroutine cublas_init()
        integer(c_int) :: status
        type(c_ptr) :: handle_ptr

        if (c_associated(cublas_handle)) then
            print *, "cuBLAS already initialized"
            return
        end if

        status = cublasCreate_v2(handle_ptr)
        if (status /= 0) then
            print *, "ERROR: cuBLAS initialization failed with status:", status
            stop
        end if

        cublas_handle = handle_ptr
        print *, "✓ cuBLAS initialized successfully"
        print *, "  GPU: NVIDIA GeForce RTX 2080 Ti (4352 CUDA cores)"
        print *, "  CUDA version: 12.6"
    end subroutine cublas_init


    !> Shutdown cuBLAS library
    !> Call once at program end
    subroutine cublas_shutdown()
        integer(c_int) :: status

        if (.not. c_associated(cublas_handle)) return

        status = cublasDestroy_v2(cublas_handle)
        if (status /= 0) then
            print *, "WARNING: cuBLAS shutdown failed with status:", status
        end if

        cublas_handle = c_null_ptr
        print *, "✓ cuBLAS shutdown complete"
    end subroutine cublas_shutdown


    !> Allocate GPU memory for activations and outputs
    !> Call once at model initialization
    subroutine allocate_gpu_memory(M, N, K_dim)
        integer(int32), intent(in) :: M, N, K_dim
        integer(c_int) :: status
        integer(c_size_t) :: size_A, size_W, size_Out

        ! Calculate memory sizes
        size_A = int(M * K_dim * 4, c_size_t)   ! FP32 = 4 bytes
        size_W = int(K_dim * N * 4, c_size_t)
        size_Out = int(M * N * 4, c_size_t)

        ! Allocate device memory
        status = cudaMalloc(d_A_fp32, size_A)
        if (status /= 0) stop "ERROR: cudaMalloc failed for activations"

        status = cudaMalloc(d_W_fp32, size_W)
        if (status /= 0) stop "ERROR: cudaMalloc failed for weights"

        status = cudaMalloc(d_Out, size_Out)
        if (status /= 0) stop "ERROR: cudaMalloc failed for output"

        print *, "✓ GPU memory allocated:"
        print *, "  Activations:", size_A / 1024**2, "MB"
        print *, "  Weights:", size_W / 1024**2, "MB"
        print *, "  Output:", size_Out / 1024**2, "MB"
    end subroutine allocate_gpu_memory


    !> Dequantize INT4 weights to FP32 and transfer to GPU
    !> Call once per layer at model load
    subroutine dequantize_weights_gpu(W_Q, W_scales, N, K_dim)
        integer(int8), intent(in) :: W_Q(:,:)       ! [K/2, N] packed
        real(real32), intent(in) :: W_scales(:)     ! [N]
        integer(int32), intent(in) :: N, K_dim

        real(real32), allocatable, target :: W_fp32(:,:)    ! [K, N] host
        integer(int32) :: j, k_idx, k_packed
        integer(int32) :: packed_byte, qval
        integer(c_int) :: status
        integer(c_size_t) :: size_W

        ! Allocate and dequantize on CPU
        allocate(W_fp32(K_dim, N))

        !$omp parallel do private(k_idx,k_packed,packed_byte,qval) schedule(static)
        do j = 1, N
            do k_idx = 1, K_dim, 2
                k_packed = (k_idx + 1) / 2
                packed_byte = int(W_Q(k_packed, j), int32)

                ! First 4-bit value
                qval = iand(packed_byte, 15)
                if (qval >= 8) qval = qval - 16
                W_fp32(k_idx, j) = real(qval, real32) * W_scales(j)

                ! Second 4-bit value
                if (k_idx + 1 <= K_dim) then
                    qval = iand(ishft(packed_byte, -4), 15)
                    if (qval >= 8) qval = qval - 16
                    W_fp32(k_idx + 1, j) = real(qval, real32) * W_scales(j)
                end if
            end do
        end do
        !$omp end parallel do

        ! Transfer to GPU
        size_W = int(K_dim * N * 4, c_size_t)
        status = cudaMemcpy(d_W_fp32, c_loc(W_fp32), size_W, cudaMemcpyHostToDevice)
        if (status /= 0) stop "ERROR: cudaMemcpy failed for weights"

        deallocate(W_fp32)
        print *, "✓ Weights dequantized and transferred to GPU"
    end subroutine dequantize_weights_gpu


    !> cuBLAS-accelerated INT4 matmul (with on-the-fly dequantization)
    !> For benchmarking - not optimal for inference
    subroutine matmul_int4_cublas(A, W_Q, W_scales, Out, M, N, K_dim)
        integer(int8), intent(in) :: A(:,:)         ! [M, K]
        integer(int8), intent(in) :: W_Q(:,:)       ! [K/2, N] packed
        real(real32), intent(in) :: W_scales(:)     ! [N]
        real(real32), intent(out), target :: Out(:,:)       ! [M, N]
        integer(int32), intent(in) :: M, N, K_dim

        real(real32), allocatable, target :: A_fp32(:,:)
        integer(c_int) :: status
        integer(c_size_t) :: size_A, size_Out
        real(c_float) :: alpha, beta
        integer(int32) :: i, j

        ! Dequantize weights and upload to GPU
        call dequantize_weights_gpu(W_Q, W_scales, N, K_dim)

        ! Convert activations to FP32
        allocate(A_fp32(M, K_dim))
        !$omp parallel do collapse(2)
        do j = 1, K_dim
            do i = 1, M
                A_fp32(i, j) = real(A(i, j), real32)
            end do
        end do
        !$omp end parallel do

        ! Transfer activations to GPU
        size_A = int(M * K_dim * 4, c_size_t)
        status = cudaMemcpy(d_A_fp32, c_loc(A_fp32), size_A, cudaMemcpyHostToDevice)
        if (status /= 0) stop "ERROR: cudaMemcpy failed for activations"

        ! Call cuBLAS SGEMM
        ! Out = A_fp32 * W_fp32
        alpha = 1.0
        beta = 0.0

        status = cublasSgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, &
                                M, N, K_dim, alpha, &
                                d_A_fp32, M, &
                                d_W_fp32, K_dim, &
                                beta, d_Out, M)
        if (status /= 0) stop "ERROR: cublasSgemm failed"

        ! Synchronize GPU
        status = cudaDeviceSynchronize()

        ! Copy result back to host
        size_Out = int(M * N * 4, c_size_t)
        status = cudaMemcpy(c_loc(Out), d_Out, size_Out, cudaMemcpyDeviceToHost)
        if (status /= 0) stop "ERROR: cudaMemcpy failed for output"

        deallocate(A_fp32)
    end subroutine matmul_int4_cublas


    !> cuBLAS-accelerated matmul with pre-dequantized weights on GPU
    !> FASTEST for inference - weights already on device
    subroutine matmul_int4_cublas_cached(A, Out, M, N, K_dim)
        integer(int8), intent(in) :: A(:,:)         ! [M, K]
        real(real32), intent(out), target :: Out(:,:)       ! [M, N]
        integer(int32), intent(in) :: M, N, K_dim

        real(real32), allocatable, target :: A_fp32(:,:)
        integer(c_int) :: status
        integer(c_size_t) :: size_A, size_Out
        real(c_float) :: alpha, beta
        integer(int32) :: i, j

        ! Convert activations to FP32
        allocate(A_fp32(M, K_dim))
        !$omp parallel do collapse(2)
        do j = 1, K_dim
            do i = 1, M
                A_fp32(i, j) = real(A(i, j), real32)
            end do
        end do
        !$omp end parallel do

        ! Transfer activations to GPU
        size_A = int(M * K_dim * 4, c_size_t)
        status = cudaMemcpy(d_A_fp32, c_loc(A_fp32), size_A, cudaMemcpyHostToDevice)
        if (status /= 0) stop "ERROR: cudaMemcpy failed for activations"

        ! Call cuBLAS SGEMM (weights already on device)
        alpha = 1.0
        beta = 0.0

        status = cublasSgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, &
                                M, N, K_dim, alpha, &
                                d_A_fp32, M, &
                                d_W_fp32, K_dim, &
                                beta, d_Out, M)
        if (status /= 0) stop "ERROR: cublasSgemm failed"

        ! Synchronize GPU
        status = cudaDeviceSynchronize()

        ! Copy result back to host
        size_Out = int(M * N * 4, c_size_t)
        status = cudaMemcpy(c_loc(Out), d_Out, size_Out, cudaMemcpyDeviceToHost)
        if (status /= 0) stop "ERROR: cudaMemcpy failed for output"

        deallocate(A_fp32)
    end subroutine matmul_int4_cublas_cached

end module matmul_cublas
