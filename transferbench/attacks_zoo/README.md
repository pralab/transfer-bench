# 🚀 Transfer-Based Black-Box Attacks

To install these attacks, run:

```bash
pip install -e 'git+https://github.com/fabiobrau/transfer-bench.git[full]'
```

## 🤝 Contributing

Want to add an attack? Submit a pull request! ✅ Approved attacks will be included in future analyses.

### 🛠️ How to Add an Attack

Implement a function following this protocol:

```python
from typing import Protocol, Optional
from torch import Tensor

class Model(Protocol):
    r"""A model is a callable that takes a tensor and returns a tensor."""

    def __call__(self, inputs: Tensor) -> Tensor: ...

class AttackStep(Protocol):
    r"""Defines the protocol for an attack step."""

    def __call__(
        self,
        victim_model: Model,
        surrogate_models: list[Model],
        inputs: Tensor,
        labels: Tensor,
        targets: Optional[Tensor] = None,
        eps: Optional[float] = None,
        p: Optional[float | str] = None,
        maximum_queries: Optional[int] = None,
    ) -> Tensor:
        r"""Executes the attack on a batch of data.

        Parameters
        ----------
        - victim_model: The target model being attacked.
        - surrogate_models: Surrogate models used to craft the attack.
        - inputs: Input samples.
        - labels: True labels.
        - targets: (Optional) Target labels for targeted attacks.
        - eps: (Optional) Perturbation constraint.
        - p: (Optional) Norm type for the perturbation.
        - maximum_queries: (Optional) Max queries allowed.

        Returns
        -------
        - Tensor: The adversarial examples.

        Example:
        def attack_step(
            victim_model: Model,
            *surrogate_models: Model,
            inputs: Tensor,
            labels: Tensor,
            targets: Optional[Tensor] = None,
            eps: Optional[float] = None,
            p: Optional[float | str] = None,
            maximum_queries: Optional[int] = None,
        ) -> Tensor:
            ...
        """
```

### 🔗 Model Association

Each attack must be linked to the models used in the referenced paper. The `(victim, surrogates)` pair should be documented. If the models are from [`torchvision`](https://pytorch.org/vision/main/models.html), specify the model name. Otherwise, provide an `nn.Module` instance.

**ℹ️ Note:** Input images are assumed to have pixel values in the range [0,1]. Use any normalization method you prefer, or apply `add_normalization` from `transferbench.models.utils` for standardization.

