import gc
import torch
from torch.utils.data import Subset
from transferbench.attack_evaluation import AttackEval
from transferbench.attacks_zoo import HybridAttack
from transferbench.datasets.datasets import ImageNetT
from transferbench.scenarios import load_attack_scenario

# Clear any existing cache
torch.mps.empty_cache()
gc.collect()

from dataclasses import replace

evaluator = AttackEval(HybridAttack)
scenario = load_attack_scenario("debug")[0]
scenario.hp = replace(scenario.hp, maximum_queries=10000)
evaluator.set_scenarios(scenario)
result = evaluator.run(batch_size=10, device="mps")
print(result)

# Clean up after test
torch.mps.empty_cache()
gc.collect()