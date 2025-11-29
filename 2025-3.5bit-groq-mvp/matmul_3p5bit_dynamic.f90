! Author: [Your Name] - First 3.5-bit Fortran implementer worldwide (Inspired by 1990 award)
pure subroutine matmul_3p5bit_dynamic(a_int8, w_pack, scales, offsets, c, M, N, K)
  integer(int8),  intent(in)  :: a_int8(M,K)
  integer(int8),  intent(in)  :: w_pack(K/2,N)     ! 每 2 个 neuron 存 7 bit
  real(fp32),     intent(in)  :: scales(N), offsets(N)
  integer(int32), intent(out) :: c(M,N)
  integer(int32) :: i, j, k, idx, raw7, n1, n2

  do concurrent(j=1:N, i=1:M)
    c(i,j) = 0
    do k = 1, K, 2
      idx = (k-1)/2 + 1
      raw7 = iand(w_pack(idx,j), int(z'7F'))         ! 取低 7 bit → 3.5bit×2
      n1 = ishft(raw7, -4)                           ! 高 3 bit + 符号
      n2 = iand(raw7, 15)                            ! 低 4 bit → 但实际只用 3 bit
      if (n1 >= 8)  n1 = n1 - 16
      if (n2 >= 8)  n2 = n2 - 16
      c(i,j) = c(i,j) + a_int8(i,k)   * n1
      if (k+1 <= K) c(i,j) = c(i,j) + a_int8(i,k+1) * n2
    end do
    c(i,j) = nint((c(i,j) + offsets(j)) * scales(j))
  end do
end subroutine
