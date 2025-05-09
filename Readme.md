# TransferBench 🚀

TransferBench is a Python package designed for evaluating black-box transfer attacks using one or more surrogate models. It provides a flexible and streamlined interface for testing attack effectiveness across a variety of scenarios involving different victim models, surrogate models, and datasets. Optional robust victim models are also supported.

## Features ✅

- Effortless evaluation of black-box attacks on predefined scenarios.
- Fully customizable via user-defined evaluation scenarios.
- Lightweight core dependencies; extended scenarios available via optional extras.

## Installation ⚙️

To install the standard version:

```bash
pip install git+https://git@github.com/pralab/transfer-bench.git
```

To enable evaluation on robust scenarios or cifar datasets, install the additional dependencies:

```bash
pip install "git+https://git@github.com/pralab/transfer-bench.git#egg=transferbench[robust]"
```
or
```bash
pip install "git+https://git@github.com/pralab/transfer-bench.git#egg=transferbench[cifar]"
```
**quotes are needed**

## Quickstart 📌

Here's a minimal example to evaluate an attack using the default settings:

```python
from transferbench import AttackEval
from transferbench.attacks_zoo import NaiveAvg

evaluator = AttackEval(NaiveAvg)
print(evaluator.scenarios)  # Display default scenarios
result = evaluator.run(batch_size=4, device="cuda:1")  # Run evaluation
print(result)
```

For more advanced examples and customization options, see the [tutorial notebook](examples/example-attack-evaluation.ipynb).

## Evaluation Scenarios 🎯

Attack evaluations are grouped into **campaigns**, each defining a different set of victim-surrogate model configurations:

- `etero`: heterogeneous surrogates
- `omeo`: homogeneous surrogates
- `robust`: robust victim models (optional)

### Default Scenario: `etero-imagenet-inf`

```yaml
hetero-imagenet-inf:
  - hp:
      maximum_queries: 50
      p: "inf"
      eps: 0.062745  # 16/255
    victim_model: "vgg19"
    surrogate_models: ["resnet50", "resnext50_32x4d", "densenet121", "swin_b", "swin_t", "vit_b_32"]  # CNNPool
    dataset: "ImageNetT"

  - hp:
      maximum_queries: 50
      p: "inf"
      eps: 0.062745
    victim_model: "resnext101_32x8d"
    surrogate_models: ["inception_v3", "convnext_base", "vgg16",  "swin_b", "swin_t", "vit_b_32"]  # ResPool
    dataset: "ImageNetT"

  - hp:
      maximum_queries: 50
      p: "inf"
      eps: 0.062745
    victim_model: "vit_b_16"
    surrogate_models: ["inception_v3", "convnext_base", "vgg16", "resnet50", "resnext50_32x4d", "densenet121"]  # ViTPool
    dataset: "ImageNetT"
```

Other included scenarios:
- `omeo-imagenet-inf`
- `robust-imagenet-inf` *(optional, requires `[robust]` installation)*

## Command-Line Interface: `trbench`

For full pipeline control, use the `trbench` CLI script. It helps manage experiment runs, tracks progress, and saves results automatically.

See the [`trbench` guide](transferbench/benchmark_tools/Readme.md) for more details.

## Contributing 🤝

We welcome contributions! To contribute to the `attacks_zoo` or other components, please read our [contribution guide](transferbench/attacks_zoo/README.md).

## License 📜

TransferBench is released under the MIT License. See the [LICENSE](LICENSE) file for full details.

