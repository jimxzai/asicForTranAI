--  Byte layout of one 16-weight group: 8 seven-bit pair codes <-> 7 bytes. Mirrors pack_column /
--  unpack_column in kernel3p5/pack3p5.f90: pair P occupies bits 7P .. 7P+6 of a little-endian
--  56-bit word; byte B holds bits 8B .. 8B+7.
--
--  GNATprove proves here that splitting the word into bytes and re-joining is the identity.
--  The remaining half (each 7-bit field of the word is the code placed there) did not close under
--  the SMT provers (CVC5, Z3, Alt-Ergo, 2-6 minutes); it is proved in Lean instead
--  (lean-kernel/LeanKernel/Layout.lean, SAT-based bit-vector solving).

with Interfaces; use Interfaces;
with Pack3p5;    use Pack3p5;

package Pack_Layout with SPARK_Mode, Pure is

   subtype Pair_Index is Natural range 0 .. 7;
   subtype Byte_Index is Natural range 0 .. 6;

   type Codes  is array (Pair_Index) of Code7;
   type Bytes7 is array (Byte_Index) of Unsigned_8;

   function Word_Of_Codes (C : Codes) return Unsigned_64 is
     (Unsigned_64 (C (0))
      or Shift_Left (Unsigned_64 (C (1)), 7)
      or Shift_Left (Unsigned_64 (C (2)), 14)
      or Shift_Left (Unsigned_64 (C (3)), 21)
      or Shift_Left (Unsigned_64 (C (4)), 28)
      or Shift_Left (Unsigned_64 (C (5)), 35)
      or Shift_Left (Unsigned_64 (C (6)), 42)
      or Shift_Left (Unsigned_64 (C (7)), 49));

   function Byte_Of (W : Unsigned_64; B : Byte_Index) return Unsigned_8 is
     (Unsigned_8 (Shift_Right (W, 8 * B) and 255));

   function Pack (C : Codes) return Bytes7 is
     (declare
        W : constant Unsigned_64 := Word_Of_Codes (C);
      begin
        (0 => Byte_Of (W, 0), 1 => Byte_Of (W, 1), 2 => Byte_Of (W, 2),
         3 => Byte_Of (W, 3), 4 => Byte_Of (W, 4), 5 => Byte_Of (W, 5),
         6 => Byte_Of (W, 6)));

   function Word_Of_Bytes (B : Bytes7) return Unsigned_64 is
     (Unsigned_64 (B (0))
      or Shift_Left (Unsigned_64 (B (1)), 8)
      or Shift_Left (Unsigned_64 (B (2)), 16)
      or Shift_Left (Unsigned_64 (B (3)), 24)
      or Shift_Left (Unsigned_64 (B (4)), 32)
      or Shift_Left (Unsigned_64 (B (5)), 40)
      or Shift_Left (Unsigned_64 (B (6)), 48));

   --  Splitting the 56-bit word into 7 bytes and re-joining is the identity.
   procedure Lemma_Word (C : Codes)
   with Ghost, Post => Word_Of_Bytes (Pack (C)) = Word_Of_Codes (C);

end Pack_Layout;
