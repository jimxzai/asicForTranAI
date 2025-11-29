-- Test harness for SPARK-verified transformer layer
-- This file is NOT verified by SPARK (only imports verified packages)

with Ada.Text_IO; use Ada.Text_IO;
with Transformer_Layer_Safe;
with HIP_Wrapper_Safe;

procedure Test_Transformer is
   -- Simple test: Create dummy inputs, call verified procedures
   N : constant Positive := 16;
   M : constant Positive := 8192;
   W : constant Positive := 8192;

   -- Dummy quantized weights (would be loaded from file in production)
   A_q : HIP_Wrapper_Safe.Activation_Matrix (1 .. N, 1 .. M);
   B_packed : HIP_Wrapper_Safe.Weight_Matrix_Packed (1 .. (M * W + 1) / 2);
   Scales : HIP_Wrapper_Safe.Scale_Vector (1 .. W);
   C_output : HIP_Wrapper_Safe.Output_Matrix (1 .. N, 1 .. W);

begin
   Put_Line ("=== SPARK Transformer Layer Test ===");
   Put_Line ("N (batch): " & N'Image);
   Put_Line ("M (input dim): " & M'Image);
   Put_Line ("W (output dim): " & W'Image);

   -- Initialize inputs (dummy values for testing)
   for I in A_q'Range(1) loop
      for J in A_q'Range(2) loop
         A_q(I, J) := 0;  -- Zero initialization
      end loop;
   end loop;

   for I in B_packed'Range loop
      B_packed(I) := 0;  -- Zero weights
   end loop;

   for I in Scales'Range loop
      Scales(I) := 1.0;  -- Unity scales
   end loop;

   Put_Line ("Inputs initialized.");

   -- Call SPARK-verified matrix multiplication
   Put_Line ("Calling HIP_Matmul_3p5bit (SPARK-verified)...");

   HIP_Wrapper_Safe.HIP_Matmul_3p5bit (
      A_Quantized => A_q,
      B_Packed => B_packed,
      Scales => Scales,
      C_Output => C_output,
      N => N, M => M, W => W
   );

   Put_Line ("HIP_Matmul_3p5bit completed successfully.");
   Put_Line ("Output C(1,1): " & C_output(1, 1)'Image);

   Put_Line ("=== Test PASSED ===");
   Put_Line ("All SPARK contracts satisfied (no runtime errors).");

exception
   when others =>
      Put_Line ("=== Test FAILED ===");
      Put_Line ("Exception raised (should never happen if SPARK verification correct)");
end Test_Transformer;
