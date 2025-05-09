from torch.utils.data import Subset
from transferbench.attack_evaluation import AttackEval
from transferbench.attacks_zoo import SASD_WS
from transferbench.datasets.datasets import ImageNetT
from transferbench.scenarios import load_attack_scenario


def main():
    evaluator = AttackEval(SASD_WS)
    scenario = load_attack_scenario("sasd_ws")[0]
    scenario.dataset = Subset(ImageNetT(), range(100))
    evaluator.set_scenarios(scenario, scenario)
    result = evaluator.run(batch_size=2, device="cpu")


main()
