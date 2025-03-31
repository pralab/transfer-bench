from torchvision.models import resnet18, resnet50, resnet101, resnet152
from transferbench.evaluate_transferability import TransferabilityEvaluator

victim_model = resnet50(weights="DEFAULT")
surrogate_models = [
    resnet18(weights="DEFAULT"),
    resnet101(weights="DEFAULT"),
    resnet152(weights="DEFAULT"),
]

evaluator = TransferabilityEvaluator(victim_model, *surrogate_models)
result = evaluator.run(batch_size=4, device="cuda:1")
print(result)
