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

