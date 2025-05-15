# 🚀 Transfer-Based Black-Box Attacks zoo

To install these attacks, run:

```bash
pip install git+https://git@github.com/pralab/transfer-bench.git
```

## Implemented Attacks

Implemented attacks, all the implementation accept batch-wise computation.

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
## 🤝 Contributing

Want to add an attack? Submit a pull request! ✅ Approved attacks will be included in future analyses.

### 🛠️ How to Add an Attack

Implement a function following this protocol:

```python
from typing import Protocol, Optional
from torch import nn, Tensor

class CallableModel(Protocol):
    r"""A model is a callable that takes a tensor and returns a tensor."""

    def __call__(self, inputs: Tensor, forward_mask: Optional[Tensor] = None) -> Tensor:
        r"""Callable that take a tensor as input and optionally a binary mask.

        The forward_mask is used to counting the actual forward passes.
        """


@runtime_checkable
class TransferAttack(Protocol):
    r"""Attack step protocol."""

    def __call__(
        self,
        victim_model: CallableModel,
        surrogate_models: list[Module],
        inputs: Tensor,
        labels: Tensor,
        targets: Optional[Tensor] = None,
        eps: Optional[float] = None,
        p: Optional[float | str] = None,
        maximum_queries: Optional[int] = None,
    ) -> Tensor:
        r"""Perform the attack on a batch of data.

        Parameters
        ----------
        - victim_model (CallableModel): The victim model.
        - surrogate_models (list[Modules]): The surrogate models.
        - inputs (Tensor): The input samples.
        - labels (Tensor): The labels.
        - targets (Tensor): The target labels for targeted-attack.
        - eps (float): The epsilon of the constraint.
        - p (float): The norm of the constraint.
        - maximum_queries (int): The maximum number of queries.

        Returns
        -------
        - Tensor: The adversarial examples.

        The transfer_attack function should have the following signature:
        '''
        def my_transfer_attack(
            victim_model: CallableModel,
            surrogate_models: list[Module],
            inputs: Tensor,
            labels: Tensor,
            targets: Optional[Tensor] = None,
            eps: Optional[float] = None,
            p: Optional[float | str] = None,
            maximum_queries: Optional[int] = None,
        ) -> Tensor:
            ...
        '''
        N.B the attack can work either in batch or single sample mode, nevertheless the
        queries of the victim are counted sample-wise only if a mask containing the
        information of the samples that needs to be computed is provided. Without a
        mask the queries are computed, batch-wise.
        """
```


**ℹ️ Note:** Input images are assumed to have pixel values in the range [0,1]. Use any normalization method you prefer, or apply `add_normalization` from `transferbench.models.utils` for standardization.

