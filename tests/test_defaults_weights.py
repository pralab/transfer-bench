r"""Test for checking deafult weights of models."""

import torch
from transferbench.datasets import get_loader
from transferbench.models import get_model, list_models

loader = get_loader("ImageNetT", batch_size=256, device="cuda")


@torch.no_grad()
def eval_accuracy(model):
    total = 0
    correct = 0
    for i, (inputs, labels, _) in enumerate(get_loader("ImageNetT")):
        model.eval()
        model.to("cuda")
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        # Forward pass
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Print accuracy
    return 100 * correct / total


print(list_models())
for model_id in list_models():
    model = get_model(model_id)
    acc = eval_accuracy(model)
    print(f"Model: {model_id}, Accuracy:{acc}")
