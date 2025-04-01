r"""Check accuracy for models with  deafult weights."""

import pandas as pd
import torch
from transferbench.datasets import get_loader
from transferbench.models import get_model, list_models


@torch.no_grad()
def eval_accuracy(model):
    total = 0
    correct = 0
    for _, (inputs, labels, _) in enumerate(get_loader("ImageNetT")):
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


if __name__ == "__main__":
    loader = get_loader("ImageNetT", batch_size=256, device="cuda")
    # Test the accuracy of each modes
    acc_dict = {"model_id": [], "accuracy": []}
    for model_id in list_models():
        model = get_model(model_id)
        acc = eval_accuracy(model)
        acc_dict["model_id"].append(model_id)
        acc_dict["accuracy"].append(acc)

    # Save the accuracy to a file

    df_accuracy = pd.DataFrame(acc_dict)
    df_accuracy.to_csv("accuracy.csv", index=False)
