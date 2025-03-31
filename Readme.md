# Transfer-Bench

Transfer-Bench is a Python package designed to evaluate black-box attacks using one or more surrogate models. It provides tools to assess the transferability of adversarial examples and analyze the effectiveness of attacks in various scenarios.

## Features

- Evaluate black-box attacks with surrogate models.
- Analyze transferability of adversarial examples.
- Flexible and extensible framework for research and experimentation.

## Installation

To install Transfer-Bench, clone the repository and install the dependencies:

```bash
git clone https://github.com/your-repo/transfer-bench.git
cd transfer-bench
pip install -r requirements.txt
```

## Usage

Here is a basic example of how to use Transfer-Bench:

```python
from transferbench import evaluate_transferability
from transferbench.utils import visualize_statistics

# Define your black-box model and surrogate models
black_box_model = ...
surrogate_models = [...]

# Evaluate the attack
results = evaluate_transferability(black_box_model, surrogate_models, attack_method="NaiveAvg")
print(results)
```

For detailed documentation and examples, please refer to the [documentation](docs/).


Here is a basic example of how to evaluate the methods on the default settings

```python
from transferbench import evaluate_attack
def my_attack(self, model, surrogates, inputs, labels, targets = None):
    ...
    return adv_inputs

results = evaluate_attack()

```
## Proposed transferability metrics
- NaiveAvg, poor dummy but simple
- Learned 

## Proposed surrogates
- The dimensionality of the surrogates can be fixed (6 is fair enough)
- The victim in the pool, measuring the selection capabilities
- Different level of similarity
- 

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
