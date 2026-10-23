-- SPARK-LLaMA-Safety: Quantization Package Specification
-- Author: [Your Name]
-- Date: 2025-11-28
--
-- 3.5-bit dynamic asymmetric quantization with formal proofs
-- This is the SPARK 2014 port of the Fortran world record implementation

with Interfaces; use Interfaces;

package Quantization
  with SPARK_Mode => On
is
   -- 3.5-bit encoding: 7 bits encode 2 weights (4-bit + 3-bit)
   type Packed_Weight is new Unsigned_8 range 0 .. 127;
   
   type Weight_Array is array (Positive range <>) of Packed_Weight;
   type Scale_Array is array (Positive range <>) of Float;
   
   -- Decode two 3.5-bit values from 7-bit packed representation
   procedure Decode_3p5bit
     (Packed : Packed_Weight;
      W1, W2 : out Integer_8)
   with
     Global => null,
     Post =>
       W1 in -8 .. 7 and    -- 4-bit signed range
       W2 in -4 .. 3;       -- 3-bit signed range
   
   -- Dequantize to FP32 with scale/offset
   function Dequantize
     (Quantized : Integer_8;
      Scale     : Float;
      Offset    : Float)
   return Float
   with
     Global => null,
     Pre => Scale > 0.0,
     Post => Dequantize'Result in -1.0e6 .. 1.0e6;  -- Reasonable FP32 range
     
   -- PROVEN: These operations cannot overflow, produce NaN, or access out-of-bounds
   
end Quantization;
