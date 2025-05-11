from transferbench import AttackEval

evals = AttackEval("GFCS")
evals.set_scenarios("gfcs")
evals.run(device="mps", batch_size=5)
