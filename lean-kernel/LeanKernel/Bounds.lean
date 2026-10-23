/-!
# int32 accumulator bounds for the quantized dot products

Mirrors `acc_bound()` in `kernel3p5/pack3p5.f90` and the MXFP4 block bound in `mxfp4.f90`.
Activations are int8 (|a| ≤ 128). 3.5-bit weights have |w| ≤ 8. MXFP4 weights, doubled, have
|2·v| ≤ 12. Core Lean only; every theorem is fully proved.
-/

namespace LeanKernel

/-- Dot product of two integer lists (the code's multiply-accumulate loop). -/
def dot : List Int → List Int → Int
  | a :: as, w :: ws => a * w + dot as ws
  | _, _ => 0

/-- Generic bound: if every |a| ≤ A and every |w| ≤ B, a dot product of K equal-length terms
has magnitude at most A·B·K. -/
theorem dot_natAbs_le (A B : Nat) :
    ∀ (as ws : List Int),
      (∀ a ∈ as, a.natAbs ≤ A) → (∀ w ∈ ws, w.natAbs ≤ B) → as.length = ws.length →
      (dot as ws).natAbs ≤ A * B * as.length
  | [], [], _, _, _ => by simp [dot]
  | [], _ :: _, _, _, h => by simp at h
  | _ :: _, [], _, _, h => by simp at h
  | a :: as, w :: ws, ha, hw, hl => by
      have hl' : as.length = ws.length := by simpa using hl
      have ih := dot_natAbs_le A B as ws
        (fun x hx => ha x (List.mem_cons_of_mem _ hx))
        (fun x hx => hw x (List.mem_cons_of_mem _ hx)) hl'
      have hterm : (a * w).natAbs ≤ A * B := by
        rw [Int.natAbs_mul]
        exact Nat.mul_le_mul (ha a (List.mem_cons_self ..)) (hw w (List.mem_cons_self ..))
      have hsum := Int.natAbs_add_le (a * w) (dot as ws)
      simp only [dot, List.length_cons]
      calc (a * w + dot as ws).natAbs
          ≤ (a * w).natAbs + (dot as ws).natAbs := hsum
        _ ≤ A * B + A * B * as.length := Nat.add_le_add hterm ih
        _ = A * B * (as.length + 1) := by rw [Nat.mul_succ]; omega

/-! ### 3.5-bit weights: |a| ≤ 128, |w| ≤ 8 → 1024 per term -/

/-- K ≤ 2,097,151 terms: the int32 accumulator cannot overflow. -/
theorem int32_safe_3p5 (as ws : List Int)
    (ha : ∀ a ∈ as, a.natAbs ≤ 128) (hw : ∀ w ∈ ws, w.natAbs ≤ 8)
    (hl : as.length = ws.length) (hK : as.length ≤ 2097151) :
    -2147483648 ≤ dot as ws ∧ dot as ws ≤ 2147483647 := by
  have h := dot_natAbs_le 128 8 as ws ha hw hl
  omega

/-- The bound is tight: at K = 2,097,152 the worst case exceeds int32. -/
theorem int32_tight_3p5 : 128 * 8 * (2097152 : Nat) > 2147483647 := by decide

/-! ### MXFP4: |a| ≤ 128, |2·v| ≤ 12, blocks of 32 -/

/-- Per-block accumulator magnitude bound 32·128·12 = 49152. -/
theorem mx_block_bound (as ws : List Int)
    (ha : ∀ a ∈ as, a.natAbs ≤ 128) (hw : ∀ w ∈ ws, w.natAbs ≤ 12)
    (hl : as.length = ws.length) (h32 : as.length = 32) :
    (dot as ws).natAbs ≤ 49152 := by
  have h := dot_natAbs_le 128 12 as ws ha hw hl
  rw [h32] at h
  exact h

/-- That bound fits int32 with a large margin. -/
theorem mx_block_fits_int32 : (49152 : Nat) ≤ 2147483647 := by decide

end LeanKernel
