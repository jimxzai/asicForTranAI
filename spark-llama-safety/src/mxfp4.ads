--  MXFP4 (OCP microscaling) block dot product, mirroring kernel3p5/mxfp4.f90.
--  Twice an E2M1 value is an integer in {0,1,2,3,4,6,8,12}; bit 3 of the 4-bit code is the sign.

with Interfaces;

package Mxfp4 with SPARK_Mode, Pure is

   subtype Code4   is Natural range 0 .. 15;
   subtype Doubled is Integer range -12 .. 12;

   Table : constant array (Code4) of Doubled :=
     (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);

   Block_Size : constant := 32;
   type Act_Block  is array (1 .. Block_Size) of Interfaces.Integer_8;
   type Code_Block is array (1 .. Block_Size) of Code4;

   --  Exact integer block sum; the block scale is applied once afterwards as a power of two.
   function Block_Dot (A : Act_Block; C : Code_Block) return Integer
   with Post => abs Block_Dot'Result <= 49_152;   --  32 * 128 * 12

end Mxfp4;
