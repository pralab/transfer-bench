r"""Utility functions for the attacks."""

import torch
from torch import Tensor


def lp_constraint(inputs: Tensor, adv: Tensor, eps: float, p: float | str) -> bool:
    r"""Check if the adversarial example satisfies the constraints.

    The constraints are:
    - Box constraints: 0 <= adv <= 1
    - Lp norm constraints: ||inputs - adv||_p <= eps

    Parameters
    ----------
    - inputs (torch.Tensor): The original inputs.
    - adv (torch.Tensor): The adversarial examples.
    - eps (float): The maximum perturbation allowed.
    - p (float | str): The norm to be used.

    """
    box_constraints_inputs = torch.all(inputs >= 0) and torch.all(inputs <= 1)
    box_constraints_adv = torch.all(adv >= 0) and torch.all(adv <= 1)
    lp_norms = torch.all(
        torch.linalg.vector_norm(inputs - adv, float(p), dim=(1, 2, 3)) <= eps + 1e-7
    )
    return box_constraints_inputs and box_constraints_adv and lp_norms
