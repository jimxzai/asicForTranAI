package Matmul_3p5bit_Safe with SPARK_Mode => On is
   procedure Matmul_3p5bit_Dynamic (...) with
     Pre => ..., Post => (for all J in C'Range(2) => C(C'First(1), J) in Integer'First .. Integer'Last);
end Matmul_3p5bit_Safe;
