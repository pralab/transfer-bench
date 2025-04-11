from torch.utils.data import Subset
from transferbench.attack_evaluation import AttackEval
from transferbench.attacks_zoo.dswea import DSWEA
from transferbench.datasets.datasets import ImageNetT
from transferbench.scenarios import load_attack_scenario

evaluator = AttackEval(DSWEA)
scenario = load_attack_scenario("dswea")[0]
scenario.dataset = Subset(ImageNetT(), range(10))
evaluator.set_scenarios(scenario, scenario)
result = evaluator.run(batch_size=2, device="cuda:2")
print(result)
