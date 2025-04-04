# TransferBench 🚀

TransferBench is a Python package designed to evaluate black-box transfer attacks using one or more surrogate models. It provides tools for testing the effectiveness of attacks in various scenarios that involve different victim models, surrogate models, and datasets. Optional robust victim models can also be included.

## Features ✅

- Evaluate black-box attacks on predefined scenarios with minimal effort.
- Highly customizable evaluation through user-defined scenarios.
- Standard scenarios require a minimal set of dependencies, while additional scenarios can be accessed by installing the full package.

## Installation ⚙️

For the standard installation, run:

```bash
pip install -e git+ssh://git@github.com/fabiobrau/transfer-bench.git
```

For extended experiments, such as robust scenarios (which require additional dependencies), install the full version:

```bash
pip install -e git+ssh://git@github.com/fabiobrau/transfer-bench.git#egg=transfer-bench[full]
```

## Usage 📌

Here is a basic example of how to evaluate attack methods using the default settings:

```python
from transferbench.attack_evaluation import AttackEval
from transferbench.attacks_zoo import NaiveAvg

evaluator = AttackEval(NaiveAvg)
# Display default scenarios
print(evaluator.scenarios)
# Run the evaluation
result = evaluator.run(batch_size=4, device="cuda:1")
print(result)
```

## Proposed Scenarios for Attack Evaluations 🎯

### **hetero-imagenet-inf** (Default Scenario)

In this scenario, the attack is evaluated on the following cases:

```yaml
hetero-imagenet-inf:
  - hp: 
      maximum_queries: 50
      p: "inf"
      eps: 0.062745 # 16/255
    victim_model: "vgg19"
    surrogate_models: ["resnet50", "resnext50_32x4d", "densenet121", "swin_b", "swin_t", "vit_b_32"]  # CNNPool
    dataset: "ImageNetT" 

  - hp: 
      maximum_queries: 50
      p: "inf"
      eps: 0.062745 # 16/255
    victim_model: "resnext101_32x8d"
    surrogate_models: ["inceptionv3", "convnext_base", "vgg16",  "swin_b", "swin_t", "vit_b_32"]  # ResPool
    dataset: "ImageNetT"

  - hp: 
      maximum_queries: 50
      p: "inf"
      eps: 0.062745 # 16/255
    victim_model: "vit_b_16"
    surrogate_models: ["inceptionv3", "convnext_base", "vgg16", "resnet50", "resnext50_32x4d", "densenet121"]  # ViTPool
    dataset: "ImageNetT"
```

Additional included scenarios:
- **homo-imagenet-inf**
- **robust-imagenet-inf** (optional, requires additional dependencies)

## Contributing 🤝

We welcome contributions! For detailed guidelines on contributing to the `attacks_zoo`, please refer to the [Contribution Guide for attacks_zoo](attacks_zoo/README.md).

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

