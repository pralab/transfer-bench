from transferbench import AttackEval
from transferbench.attacks_zoo.query_based.subspace import SubSpace

evals = AttackEval(SubSpace)
evals.set_scenarios("subspace")
evals.run(device="mps", batch_size=5)
