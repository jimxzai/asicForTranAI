structure MCTSNode where visits : Nat := 0; totalValue : Float := 0.0
def puct (node : MCTSNode) (child : MCTSNode) : Float := ...  -- AlphaZero PUCT
