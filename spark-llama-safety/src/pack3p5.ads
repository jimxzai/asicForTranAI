--  Reference 3.5-bit pair code and int32 dot product, mirroring kernel3p5/pack3p5.f90.
--  A pair is one 7-bit code: even weight (4-bit two's complement) in bits 6..3,
--  odd weight (3-bit two's complement) in bits 2..0.

with Interfaces;

package Pack3p5 with SPARK_Mode, Pure is

   subtype Code7  is Natural range 0 .. 127;
   subtype Even_W is Integer range -8 .. 7;
   subtype Odd_W  is Integer range -4 .. 3;

   function Even_Weight (C : Code7) return Even_W is
     (if C / 8 >= 8 then C / 8 - 16 else C / 8);

   function Odd_Weight (C : Code7) return Odd_W is
     (if C mod 8 >= 4 then C mod 8 - 8 else C mod 8);

   --  decode (encode e o) = (e, o)
   function Encode (E : Even_W; O : Odd_W) return Code7 is
     (((E + 16) mod 16) * 8 + (O + 8) mod 8)
   with Post => Even_Weight (Encode'Result) = E
                and then Odd_Weight (Encode'Result) = O;

   --  The int32 accumulator of the Fortran kernel cannot overflow for this many terms
   --  (|a| <= 128, |w| <= 8, so each term is at most 1024 in magnitude).
   Max_Terms : constant := 2_097_151;

   type Act_Array  is array (Positive range <>) of Interfaces.Integer_8;
   type Code_Array is array (Positive range <>) of Code7;

   --  sum_k A(k) * w(k), two weights per code. Integer is 32 bits in GNAT, so the
   --  overflow checks proved here are the int32 overflow checks of the Fortran kernel.
   function Dot (A : Act_Array; Codes : Code_Array) return Integer
   with
     Pre  => A'First = 1 and then Codes'First = 1
             and then A'Length = 2 * Codes'Length
             and then A'Length <= Max_Terms,
     Post => abs Dot'Result <= 1024 * A'Length;

end Pack3p5;
