/-!
# 3.5-bit pair encoding: decode ∘ encode = id

Mirrors `kernel3p5/pack3p5.f90`. A pair is a 4-bit two's-complement even weight and a
3-bit two's-complement odd weight, stored as one 7-bit code (even in the high 4 bits).
Core Lean only, no Mathlib. Every theorem here is fully proved.
-/

namespace LeanKernel

/-- 7-bit code of a pair: even weight in bits 6..3, odd weight in bits 2..0. -/
def encodePair (e : BitVec 4) (o : BitVec 3) : BitVec 7 := e ++ o

/-- Even weight of a 7-bit code (signed 4-bit). -/
def decodeEven (c : BitVec 7) : Int := (c.extractLsb' 3 4).toInt

/-- Odd weight of a 7-bit code (signed 3-bit). -/
def decodeOdd (c : BitVec 7) : Int := (c.extractLsb' 0 3).toInt

/-- Decoding a code recovers exactly the two signed weights it was built from. -/
theorem decode_encode (e : BitVec 4) (o : BitVec 3) :
    decodeEven (encodePair e o) = e.toInt ∧ decodeOdd (encodePair e o) = o.toInt := by
  unfold decodeEven decodeOdd encodePair
  revert e o
  decide

/-- Encoding the decoded halves of any 7-bit code gives the code back (all 128 codes). -/
theorem encode_decode (c : BitVec 7) :
    encodePair (c.extractLsb' 3 4) (c.extractLsb' 0 3) = c := by
  unfold encodePair
  revert c
  decide

/-- The representable even weights are exactly [-8, 7]. -/
theorem even_range (e : BitVec 4) : -8 ≤ e.toInt ∧ e.toInt ≤ 7 := by
  revert e; decide

/-- The representable odd weights are exactly [-4, 3]. -/
theorem odd_range (o : BitVec 3) : -4 ≤ o.toInt ∧ o.toInt ≤ 3 := by
  revert o; decide

end LeanKernel
