from torch.utils.data import Subset
from transferbench.attack_evaluation import AttackEval
from transferbench.attacks_zoo import NES
from transferbench.datasets.datasets import CIFAR10T, ImageNetT
from transferbench.scenarios import load_attack_scenario

# Use default NES attack with original hyperparameters
evaluator = AttackEval(NES)
scenarios = load_attack_scenario("nes_debug")
scenario = scenarios[0]
scenario.dataset = Subset(ImageNetT(), range(100))
evaluator.set_scenarios(scenario)
result = evaluator.run(device="mps", batch_size=10)
print(result)