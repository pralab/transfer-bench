from torch.utils.data import Subset
from transferbench.attack_evaluation import AttackEval
from transferbench.datasets.datasets import ImageNetT
from transferbench.scenarios import load_attack_scenario

evaluator = AttackEval("MGAA")
scenario = load_attack_scenario("debug")[0]
scenario.dataset = Subset(ImageNetT(), range(100))
evaluator.set_scenarios(scenario)
result = evaluator.run(batch_size=2, device="cpu")
print(result)
