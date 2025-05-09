from torch.utils.data import Subset
from transferbench import attacks_zoo
from transferbench.attack_evaluation import AttackEval
from transferbench.datasets.datasets import ImageNetT
from transferbench.scenarios import load_attack_scenario


def main():
    oneshot_attacks = [
        "AdaEA",
        "CWA",
        "ENS",
        "LGV",
        "MBA",
        "SASD_WS",
        "SMER",
        "SVRE",
    ]
    for attack in oneshot_attacks:
        print(f"Testing {attack}...")
        attack_cls = getattr(attacks_zoo, attack)
        evaluator = AttackEval(attack_cls)
        scenario = load_attack_scenario("debug")[0]
        scenario.dataset = Subset(ImageNetT(), range(10))
        evaluator.set_scenarios(scenario)
        try:
            _ = evaluator.run(batch_size=2, device="cuda:1")
        except Exception as e:
            print(f"Failed to run {attack}: {e}")
