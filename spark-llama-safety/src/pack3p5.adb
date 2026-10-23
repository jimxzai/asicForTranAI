package body Pack3p5 with SPARK_Mode is

   function Dot (A : Act_Array; Codes : Code_Array) return Integer is
      Acc : Integer := 0;
   begin
      for I in Codes'Range loop
         pragma Loop_Invariant (abs Acc <= 2048 * (I - 1));
         Acc := Acc
                + Integer (A (2 * I - 1)) * Even_Weight (Codes (I))
                + Integer (A (2 * I))     * Odd_Weight (Codes (I));
      end loop;
      return Acc;
   end Dot;

end Pack3p5;
