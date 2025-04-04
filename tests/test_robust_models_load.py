from transferbench.models.robust_models import get_models, list_models

for model_name in list_models():
    print(f"Loading model: {model_name}")
    model = get_models(model_name)
    print(f"Model {model_name} loaded successfully.")
    print(model)
    print("-" * 50)
