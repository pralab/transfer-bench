r"""Test the pubdef models."""

from transferbench.models import get_model, list_models

available_models = list_models()[::-1]  # reverse the order to load the optional first
for model_name in available_models:
    print(f"Loading model: {model_name}: ...", end="\r")
    _ = get_model(model_name)
    print(f"Loading model: {model_name}: Done")
