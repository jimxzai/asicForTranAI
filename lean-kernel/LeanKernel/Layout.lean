import Std.Tactic.BVDecide
import LeanKernel.Pack3p5

/-!
# 16-weight group layout: 8 pair codes ↔ 7 bytes

Mirrors `pack_column` / `unpack_column` in `kernel3p5/pack3p5.f90`. Pair `p` occupies bits
`7p .. 7p+6` of a little-endian 56-bit word; byte `b` holds bits `8b .. 8b+7`.

`layout_roundtrip` is proved with `bv_decide` (bit-blasting to SAT, with the SAT certificate checked
by Lean's verified LRAT checker). That adds the axiom `Lean.ofReduceBool` to the trusted base.
-/

namespace LeanKernel

/-- Little-endian 56-bit word: code 0 in the lowest 7 bits. -/
def packWord (c0 c1 c2 c3 c4 c5 c6 c7 : BitVec 7) : BitVec 56 :=
  c7 ++ c6 ++ c5 ++ c4 ++ c3 ++ c2 ++ c1 ++ c0

def byteOf (w : BitVec 56) (b : Nat) : BitVec 8 := w.extractLsb' (8 * b) 8

def wordOfBytes (b0 b1 b2 b3 b4 b5 b6 : BitVec 8) : BitVec 56 :=
  b6 ++ b5 ++ b4 ++ b3 ++ b2 ++ b1 ++ b0

def codeAt (w : BitVec 56) (p : Nat) : BitVec 7 := w.extractLsb' (7 * p) 7

/-- Packing 8 codes into 7 bytes and unpacking returns the same 8 codes. -/
theorem layout_roundtrip (c0 c1 c2 c3 c4 c5 c6 c7 : BitVec 7) :
    let w  := packWord c0 c1 c2 c3 c4 c5 c6 c7
    let w' := wordOfBytes (byteOf w 0) (byteOf w 1) (byteOf w 2) (byteOf w 3)
                          (byteOf w 4) (byteOf w 5) (byteOf w 6)
    w' = w ∧ codeAt w' 0 = c0 ∧ codeAt w' 1 = c1 ∧ codeAt w' 2 = c2 ∧ codeAt w' 3 = c3 ∧
      codeAt w' 4 = c4 ∧ codeAt w' 5 = c5 ∧ codeAt w' 6 = c6 ∧ codeAt w' 7 = c7 := by
  intro w w'
  simp only [w, w', packWord, wordOfBytes, byteOf, codeAt]
  bv_decide

/-- End to end for one group: 8 (even, odd) weight pairs → codes → 7 bytes → codes → weights. -/
theorem group_roundtrip
    (e0 e1 e2 e3 e4 e5 e6 e7 : BitVec 4) (o0 o1 o2 o3 o4 o5 o6 o7 : BitVec 3) :
    let w  := packWord (encodePair e0 o0) (encodePair e1 o1) (encodePair e2 o2) (encodePair e3 o3)
                       (encodePair e4 o4) (encodePair e5 o5) (encodePair e6 o6) (encodePair e7 o7)
    let w' := wordOfBytes (byteOf w 0) (byteOf w 1) (byteOf w 2) (byteOf w 3)
                          (byteOf w 4) (byteOf w 5) (byteOf w 6)
    (decodeEven (codeAt w' 0) = e0.toInt ∧ decodeOdd (codeAt w' 0) = o0.toInt) ∧
    (decodeEven (codeAt w' 1) = e1.toInt ∧ decodeOdd (codeAt w' 1) = o1.toInt) ∧
    (decodeEven (codeAt w' 2) = e2.toInt ∧ decodeOdd (codeAt w' 2) = o2.toInt) ∧
    (decodeEven (codeAt w' 3) = e3.toInt ∧ decodeOdd (codeAt w' 3) = o3.toInt) ∧
    (decodeEven (codeAt w' 4) = e4.toInt ∧ decodeOdd (codeAt w' 4) = o4.toInt) ∧
    (decodeEven (codeAt w' 5) = e5.toInt ∧ decodeOdd (codeAt w' 5) = o5.toInt) ∧
    (decodeEven (codeAt w' 6) = e6.toInt ∧ decodeOdd (codeAt w' 6) = o6.toInt) ∧
    (decodeEven (codeAt w' 7) = e7.toInt ∧ decodeOdd (codeAt w' 7) = o7.toInt) := by
  intro w w'
  obtain ⟨_, h0, h1, h2, h3, h4, h5, h6, h7⟩ :=
    layout_roundtrip (encodePair e0 o0) (encodePair e1 o1) (encodePair e2 o2) (encodePair e3 o3)
                     (encodePair e4 o4) (encodePair e5 o5) (encodePair e6 o6) (encodePair e7 o7)
  rw [h0, h1, h2, h3, h4, h5, h6, h7]
  exact ⟨decode_encode e0 o0, decode_encode e1 o1, decode_encode e2 o2, decode_encode e3 o3,
         decode_encode e4 o4, decode_encode e5 o5, decode_encode e6 o6, decode_encode e7 o7⟩

end LeanKernel
