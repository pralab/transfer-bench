from transferbench import AttackEval

evals = AttackEval("SimbaODS")
evals.set_scenarios("gfcs")
evals.run(device="mps", batch_size=5)
