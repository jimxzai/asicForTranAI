package body Mxfp4 with SPARK_Mode is

   function Block_Dot (A : Act_Block; C : Code_Block) return Integer is
      Acc : Integer := 0;
   begin
      for I in Act_Block'Range loop
         pragma Loop_Invariant (abs Acc <= 1536 * (I - 1));
         Acc := Acc + Integer (A (I)) * Table (C (I));
      end loop;
      return Acc;
   end Block_Dot;

end Mxfp4;
