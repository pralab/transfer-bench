from torch.utils.data import Subset
from transferbench.attack_evaluation import AttackEval
from transferbench.attacks_zoo import NaiveAvg
from transferbench.datasets.datasets import ImageNetT
from transferbench.scenarios import load_attack_scenario

evaluator = AttackEval(NaiveAvg)
scenario = load_attack_scenario("debug")[0]
scenario.dataset = Subset(ImageNetT(), range(10))
evaluator.set_scenarios(scenario)
result = evaluator.run(batch_size=3, device="cuda:0")
print(result)
