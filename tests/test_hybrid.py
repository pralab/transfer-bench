
from transferbench.attack_evaluation import AttackEval
from transferbench.attacks_zoo import HybridAttack

evaluator = AttackEval(HybridAttack)
evaluator.set_scenarios("omeo-imagenet-inf")
result = evaluator.run(batch_size=10, device="cuda:0")

