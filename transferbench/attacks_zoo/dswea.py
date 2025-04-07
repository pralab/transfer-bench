r"""DSWEA attack implementation"""  # TODO(add paper citation)

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# from DSWEA.metrics import MarginLoss
from tqdm import tqdm


def compute_weights(gradients, sigma=2.5):
    grad_norms = torch.Tensor([grad.norm() for grad in gradients])
    weights_unnorm = torch.exp(-grad_norms / (sigma**2))
    return weights_unnorm / torch.sum(weights_unnorm)


def dswea(
    x: Tensor,
    y_target: Tensor,
    victim_model: nn.Module,
    surrogate_models: list[nn.Module],
    alpha: float = 0.01,
    T: int = 100,
    M: int = 20,
    epsilon: float = 16.0 / 255,
    Q: int = 100,
    targeted: bool = True,
):
    """
    Implement the dswea attack from the original paper.

    Args:
        x: input example (benign)
        y_target: target class
        victim_model
        surrogate_models: list of surrogate models
        alpha: step size
        T: external iterations
        M: internal iterations
        epsilon: maximum perturbation
        Q: query limit
    """
    x_orig = x.clone()
    num_surrogates = len(surrogate_models)

    # Initialize loss function
    loss_fn = MarginLoss()

    # Initialize gradients
    grads_ens = [None] * num_surrogates
    weights = torch.ones(num_surrogates) / num_surrogates
    # moved outside of the loop
    x_star = x_orig.clone()  # starting from the original image
    for q in tqdm(range(Q), desc="Query Loop"):
        if q > 0:
            weights = compute_weights(grads_ens)
        # Initialize for external loop
        # G = torch.zeros_like(x_orig)  # not used
        # x_star = x_orig.clone()  # not starting from the last_adv
        # External iteration
        for _ in range(T):
            # Compute ensemble gradient
            xs_ens = [x_star.clone().requires_grad_() for _ in range(num_surrogates)]
            loss_ens = [
                loss_fn(model(x_loc), y_target)
                for model, x_loc in zip(surrogate_models, xs_ens, strict=True)
            ]
            grads_ens = torch.autograd.grad(loss_ens, xs_ens)

            # Ensemble gradient
            g_ens = sum(w * grad for (w, grad) in zip(weights, grads_ens, strict=True))

            # Initialize for internal loop
            G_bar = torch.zeros_like(x_orig)
            x_bar = x_star.clone()

            # Sort surrogates by loss
            sorted_idx = sorted(
                range(num_surrogates), key=loss_ens.__getitem__, reverse=True
            )

            # Internal iteration
            for m in range(M):
                k = sorted_idx[m % num_surrogates]

                # Compute unbiased gradient using only surrogate model k
                x_bar.requires_grad = True
                x_bar.grad = None

                pred_x_bar = surrogate_models[k](x_bar)
                loss_x_bar = loss_fn(pred_x_bar, y_target)
                grad_x_bar = torch.autograd.grad(loss_x_bar, x_bar)[0]
                g_bar = weights[k] * (grad_x_bar - grads_ens[k]) + g_ens
                with torch.no_grad():
                    G_bar = G_bar + g_bar
                    x_bar = x_bar - alpha * torch.sign(G_bar)
                    x_bar = (x_bar - x_orig).clamp(-epsilon, epsilon) + x_orig
                    x_bar = x_bar.clamp(0, 1)

            # G = G+G_bar  # not used
            x_star = x_bar
            # x_star = x_bar - alpha * torch.sign(G)  # Dont exploit the x_bar anymore
            # x_star = (x_star - x_orig).clamp(-epsilon, epsilon) + x_orig
            # x_star = x_star.clamp(0, 1)

        # Query victim model to check if attack is successful
        with torch.no_grad():
            victim_logits = victim_model(x_star)
            victim_pred = victim_logits.argmax(dim=1)
            q += 1  # Increment query count
            print(
                f"Query {q}: Victim model predicting class {victim_pred.item()}, Target is {y_target.item()}"
            )

            if victim_pred == y_target:
                print("Attack successful, Query:", q)
                return x_star, q  # Attack successful

    # If we've exceeded query limit, return last adversarial example
    return x_star, -1
