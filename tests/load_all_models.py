r"""Test the pubdef models."""

import torch
from transferbench.datasets import get_loader
from transferbench.models import get_model, list_models


# Evaluate the models accuracy on ImageNet


@torch.no_grad()
def acc(model_name: str, batch_size: int = 200) -> float:
    """Get the accuracy of the model on ImageNet."""
    model = get_model(model_name)
    model.eval().to("cuda:2")
    loader = get_loader("ImageNetT", batch_size=batch_size, device="cuda:2")
    acc = 0.0
    for inputs, labels, _ in loader:
        outputs = model(inputs)
        acc += (outputs.argmax(1) == labels).float().sum()
    return acc / 1000


def check_models():
    available_models = list_models()[::-1]
    for model_name in available_models:
        print(f"Loading model: {model_name}: ...", end="\r")
        _ = get_model(model_name)
        print(f"Loading model: {model_name}: Done. Accuracy: {acc(model_name)}")

    print("All models loaded successfully.")


if __name__ == "__main__":
    check_models()
