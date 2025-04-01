# TransferBench

TransferBench is a Python package designed to evaluate black-box attacks using one or more surrogate models. It provides tools to assess the transferability of adversarial examples and analyze the effectiveness of attacks in various scenarios.

## Features

- Evaluate black-box attacks with surrogate models.
- Analyze transferability of adversarial examples.
- Flexible and extensible framework for research and experimentation.

## Installation

- Python 3.8 or higher
- torch 1.9.0 or higher
- torchvision 0.10.0 or higher
- tqdm

### Using pip

```bash
pip install -e git+ssh://git@github.com/fabiobrau/transfer-bench.git
```

## Usage

Here is a basic example of how to use `tranferbench`for the evaluation of the transferability between models

```python
from transferbench import TransferEval

evaluator = TransferEval("resnet50",
[
    "resnet18",
    "resnet101",
    "resnet152"
])
result = evaluator.run(batch_size=4, device="cuda:1")
print(result)
```

To run the evaluation on a customized scenario consider importing the ```TransferScenario```class as follows
```python
from transferbench import TransferEval
from transferbench.attacks import BaseHyperParameters
from transferbench.evaluations import TransferScenario

myscenario = TransferScenario(
    hp=BaseHyperParameters(eps=0.3, p=2, maximum_queries=10),
    attack_step="NaiveAvg",
    dataset="ImageNetT",
)

evaluator = TransferEval(
    "resnet50",
[
    "resnet18",
    "resnet101",
    "resnet152"
],
)
evaluator.set_scenarios(myscenario, "oneshot")
result = evaluator.run(batch_size=4, device="cuda:1")
```
where the ```BaseHyperParameters``` class is used to define the common hyper-parameters to evaluate the performances.

If further customization is needed, you can use customized models and customized datasets, consider the following example for an evaluation on **CIFAR-100** dataset.

```python
import torch
from torchvision import datasets, transforms
from transferbench import TransferEval
from transferbench.attacks import BaseHyperParameters
from transferbench.evaluations import TransferScenario
from transferbench.models.utils import add_normalization

# Load a dataset
transform = transforms.Compose([transforms.ToTensor()])
cifar100 = datasets.CIFAR100(
    root="./data/datasets",
    train=False,
    download=True,
    transform=transform,
)
cifar100_mean = [0.5, 0.5, 0.5]
cifar100_std = [1.0, 1.0, 1.0]

REPO_LINK = "chenyaofo/pytorch-cifar-models"
# Load models and normalize them
def get_model(model):
    return torch.hub.load(
    REPO_LINK, "cifar100_" + model, pretrained=True
)

# Use the dataset in the TransferEval
victim_model = add_normalization(get_model("resnet56"), cifar100_mean, cifar100_std)
surrgoate_models = [
    add_normalization(get_model("vgg11_bn"), cifar100_mean, cifar100_std),
    add_normalization(get_model("vgg13_bn"), cifar100_mean, cifar100_std),
    add_normalization(get_model("vgg16_bn"), cifar100_mean, cifar100_std),
    add_normalization(get_model("vgg19_bn"), cifar100_mean, cifar100_std),
]

evaluator = TransferEval(
    victim_model,
    surrgoate_models,
)
myscenario = TransferScenario(
    hp=BaseHyperParameters(eps=0.3, p=2, maximum_queries=10),
    attack_step="NaiveAvg",
    dataset=cifar100,
)
evaluator.set_scenarios(myscenario)
result = evaluator.run(batch_size=4, device="cuda:1")
print(result)
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
## Proposed scenarios for transferability evaluations

- **oneshot** This is the fastest evaluation method on targeted imagenet, where a naive avg attack is performed for only one-query

- **full** This is the most complete evalaution, use it only if you have sufficient time/resourches. A naive avg attack is performed with a maximum of $50$ queries on both 
**NIPS2017-Targeted**.

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
