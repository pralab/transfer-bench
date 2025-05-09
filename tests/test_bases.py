from torch.utils.data import Subset
from transferbench.attack_evaluation import AttackEval
from transferbench.attacks_zoo import BASES
from transferbench.datasets.datasets import ImageNetT
from transferbench.scenarios import load_attack_scenario

evaluator = AttackEval(BASES)
scenario = load_attack_scenario("bases")[0]
scenario.dataset = Subset(ImageNetT(), range(100))
evaluator.set_scenarios(scenario, scenario)
result = evaluator.run(batch_size=3, device="cuda:0")
print(result)
