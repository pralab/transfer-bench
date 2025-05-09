from torch.utils.data import Subset
from transferbench.attack_evaluation import AttackEval
from transferbench.attacks_zoo.query_based.simba_ods import SimbaODS
from transferbench.datasets.datasets import ImageNetT
from transferbench.scenarios import load_attack_scenario

evaluator = AttackEval(SimbaODS)
scenario = load_attack_scenario("SimbaODS")[0]
scenario.dataset = Subset(ImageNetT(), range(100))
evaluator.set_scenarios(scenario)
result = evaluator.run(batch_size=3, device="cuda:0")
print(result)
