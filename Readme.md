# TransferBench 🚀

TransferBench is a Python package designed for evaluating black-box transfer attacks using one or more surrogate models. It provides a flexible and streamlined interface for testing attack effectiveness across a variety of scenarios involving different victim models, surrogate models, and datasets. Optional robust victim models are also supported.

## Features ✅

- Effortless evaluation of black-box attacks on predefined scenarios.
- Fully customizable via user-defined evaluation scenarios, models, datasets.
- Lightweight core dependencies; extended scenarios available via optional extras.
- Connection with Weight & Biases for inspecting running, missing, and finished evaluations.

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


Scenarios are stored in the directory `transferbench/config/scenarios` where also scenarios involved in the original papers have been included for comparison.

Scenarios informations are aggregated in a yaml file as follows
```yaml
etero-imagenet-inf:
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
- `omeo-cifar10-inf`  *(optional, requires `[cifar]` installation)*
- `etero-cifar10-inf`  *(optional, requires `[cifar]` installation)*
- `robust-cifar10-inf`  *(optional, requires `[cifar,robust]` installation)*


## Command-Line Interface: `trbench`

For full pipeline control, use the `trbench` CLI script. It helps manage experiment runs, tracks progress, and saves results automatically.

See the [`trbench` guide](transferbench/benchmark_tools/Readme.md) for more details.

## Contributing to the Attack Zoo 🤝

We welcome contributions! To contribute to the `attacks_zoo` or other components, please read our [contribution guide](transferbench/attacks_zoo/README.md).

### Implemented Attacks

Implemented attacks, all the implementation allows batch-wise computation.

| **Attack**    | **Venue**   | m  | Heterogenous | Robust | Targeted | p        | ε                | ASR [%] | 𝑞̄    |
|-------------------------------|----|----|--------|--------|----------|----------|------------------|---------|-------|
| SubSpace [Guo et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/2cad8fa47bbef282badbb8de5374b894-Paper.pdf)  | NeurIPS     | 3  | Yes     | No     | No       | ∞        | 13/255           | 98.9%   | 462   |
| SimbaODS [Tashiro et al., 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/30da227c6b5b9e2482b6b221c711edfd-Paper.pdf) |NeurIPS  | 4  | No     | No     | Yes       | ∞        | 13/255           | 92.0%   | 985   |
| GFCS [Lord et al., 2022](https://openreview.net/pdf?id=Zf4ZdI4OQPV)  | ICLR        | 4  | No     | No     | Yes       | 2        | $\sqrt{0.001d}$¹         | 60.0%   | 20   |
| BASES [Cai et al., 2022](https://openreview.net/pdf?id=lSfrwyww-FR)  | NeurIPS     | 20 | No     | No     | Yes       | ∞        | 16/255           | 99.7%   | 1.8   |
| GAA [Yang et al., 2024](https://doi.org/10.1016/j.ins.2024.121013)   | PR      | 4  | No     | No     | Yes       | ∞        | 16/255           | 46.0%   | 3.9   |
| DSWEA [Hu et al., 2025](https://doi.org/10.1016/j.patcog.2024.111263) |PR           | 10 | No     | No     | Yes       | ∞        | 16/255           | 96.6%   | 2.7   |
-----------------------------------------------------------------------------------------------------------------------
¹ Images included in the experiments have $d=3\cdot 299\cdot299$ pixels, from which $\varepsilon\approx16.37$

## Paper 📄 *Benchmarking Ensemble-based Black-box Transfer Attacks*
 
 *TransferBench: Benchmarking Ensemble-based Black-box Transfer Attacks*
 Fabio Brau, Maura Pintor, Antonio Emanuele Cinà, Raffaele Mura, Luca Scionis, Luca Oneto, Fabio Roli, Battista Biggio

*Under revision for NeurIPS 2025 Datasets and Benchmarks TrackPaper*

TL;DR: TransferBench reveals limitations in surrogate model choices, robustness generalization, and query efficiency.
## License 📜

TransferBench is released under the MIT License. See the [LICENSE](LICENSE.md) file for full details.

