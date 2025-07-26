"""Hybrid Attack implementation.

Hybrid attack combines transfer attacks (using surrogate models) with
query-based optimization attacks for black-box adversarial example generation.

`Hybrid Batch Attacks: Finding Black-box Adversarial Examples with Limited Queries`
Fnu Suya, Jianfeng Chi, David Evans, Yuan Tian (USENIX Security 2020)
"""

import gc
import random
from dataclasses import asdict, dataclass
from functools import partial

import torch
from torch import Tensor

from transferbench.types import CallableModel, TransferAttack

from .nes import _nes_attack_single_with_trajectory, nes
from .utils import (
    AggregatedEnsemble,
    grad_projection,
    lp_projection,
    projected_gradient_descent,
)


def _select_seeds_random(
    batch_size: int,
    success_mask: Tensor,
    seeds_per_iteration: int = 1,
) -> list[int]:
    unsuccessful_indices = (~success_mask).nonzero(as_tuple=True)[0].tolist()
    if not unsuccessful_indices:
        return []
    num_to_select = min(seeds_per_iteration, len(unsuccessful_indices))
    return random.sample(unsuccessful_indices, num_to_select)

def _transfer_attack_phase(
    surrogate_models: list[CallableModel],
    inputs: Tensor,
    labels: Tensor,
    eps: float,
    p: str,
    alpha: float,
    inner_iterations: int,
    targets: Tensor | None = None,
) -> Tensor:
    ball_projection = partial(lp_projection, eps=eps, p=p)
    dot_projection = partial(grad_projection, p=p)
    loss_fn = AggregatedEnsemble(surrogate_models)
    return projected_gradient_descent(
            loss_fn=loss_fn,
            inputs=inputs,
            x_init=inputs,
            labels=labels,
            targets=targets,
            ball_projection=ball_projection,
            dot_projection=dot_projection,
            alpha=alpha,
            inner_iterations=inner_iterations,
        )

def _evaluate_transfer_success(
    victim_model: CallableModel,
    transfer_candidates: Tensor,
    labels: Tensor,
    original_batch_size: int,
    selected_indices: list[int],
    targets: Tensor | None = None,
) -> tuple[Tensor, int]:
    device = transfer_candidates.device
    forward_mask = torch.zeros(original_batch_size, dtype=torch.bool, device=device)
    forward_mask[selected_indices] = True
    with torch.no_grad():
        logits = victim_model(transfer_candidates, forward_mask)
        predictions = logits.argmax(dim=1)
        if targets is not None:
            success_mask = predictions == targets
        else:
            success_mask = predictions != labels
    return success_mask, len(selected_indices)


def _optimization_attack_phase(
    victim_model: CallableModel,
    transfer_candidates: Tensor,
    inputs: Tensor,
    labels: Tensor,
    eps: float,
    p: str,
    remaining_queries: int,
    nes_hyperparams: dict,
    targets: Tensor | None = None,

) -> tuple[Tensor, int]:
    if remaining_queries <= 0:
        return transfer_candidates, 0
    optimized_advs = nes(
        victim_model=victim_model,
        surrogate_models=[],
        inputs=inputs,
        labels=labels,
        targets=targets,
        eps=eps,
        p=p,
        maximum_queries=remaining_queries,
        x_init=transfer_candidates,
        **nes_hyperparams
    )
    return optimized_advs, remaining_queries


def _fine_tune_surrogate_models(
    surrogate_models: list[CallableModel],
    training_samples: Tensor,
    victim_logits_list: list[Tensor],
    fine_tune_epochs: int = 3,
    fine_tune_lr: float = 1e-3,
) -> None:
    if len(training_samples) == 0 or len(victim_logits_list) == 0:
        return
    victim_logits = torch.stack(victim_logits_list)
    victim_labels = victim_logits.argmax(dim=1)
    for surrogate in surrogate_models:
        actual_model = surrogate.model if hasattr(surrogate, "model") else surrogate
        if not any(param.requires_grad for param in actual_model.parameters()):
            continue
        optimizer = torch.optim.Adam(actual_model.parameters(), lr=fine_tune_lr)
        if hasattr(actual_model, "train"):
            actual_model.train()
        for _ in range(fine_tune_epochs):
            optimizer.zero_grad()
            if callable(surrogate):
                logits = surrogate(training_samples, None)
            else:
                logits = actual_model(training_samples)
            loss = torch.nn.functional.cross_entropy(logits, victim_labels)
            loss.backward()
            optimizer.step()
        if hasattr(actual_model, "eval"):
            actual_model.eval()


def hybrid_attack(
    victim_model: CallableModel,
    surrogate_models: list[CallableModel],
    inputs: Tensor,
    labels: Tensor,
    targets: Tensor | None = None,
    eps: float = 16 / 255,
    p: float | str = "inf",
    maximum_queries: int = 1000,
    fine_tune_frequency: int = 10,
    fine_tune_epochs: int = 3,
    fine_tune_lr: float = 1e-3,
    transfer_alpha: float = 2.0 / 255.0,
    transfer_iterations: int = 10,
    nes_sigma: float = 0.1,
    nes_lr: float = 0.02,
    nes_num_samples: int = 20,
    nes_momentum: float = 0.9,
) -> Tensor:
    """Hybrid attack combining transfer attacks and optimization-based attacks."""
    if p != "inf":
        msg = f"Unsupported norm p={p} for hybrid attack. Only L-infinity is supported."
        raise NotImplementedError(msg)
    device = inputs.device
    batch_size = inputs.size(0)
    best_adversarials = inputs.clone()
    attacked_flag = torch.zeros(batch_size, dtype=torch.bool, device=device)
    success_vec = torch.zeros(batch_size, dtype=torch.bool, device=device)
    query_num_vec = torch.zeros(batch_size, dtype=torch.int, device=device)
    training_samples = inputs.clone()
    victim_logits_list = []

    if surrogate_models:
        transfer_candidates = _transfer_attack_phase(
            surrogate_models=surrogate_models,
            inputs=inputs,
            labels=labels,
            targets=targets,
            eps=eps,
            p=p,
            alpha=transfer_alpha,
            inner_iterations=transfer_iterations,
        )
        with torch.no_grad():
            transfer_logits = victim_model(transfer_candidates, None)
            transfer_predictions = transfer_logits.argmax(dim=1)
            if targets is not None:
                transfer_success = transfer_predictions == targets
            else:
                transfer_success = transfer_predictions != labels
        success_vec[transfer_success] = True
        attacked_flag[transfer_success] = True
        best_adversarials[transfer_success] = transfer_candidates[transfer_success]
        query_num_vec += 1
        start_points = transfer_candidates.clone()
    else:
        start_points = inputs.clone()
    iteration = 0
    total_queries_used = query_num_vec.sum().item()
    max_queries_per_sample = maximum_queries // batch_size
    while (total_queries_used < maximum_queries and
           not attacked_flag.all() and
           iteration < batch_size * 2):
        candidate_idx = _select_next_seed(
            start_points, attacked_flag, labels, targets, victim_model,
            query_num_vec, max_queries_per_sample
        )
        if candidate_idx is None:
            break
        sample_queries_used = query_num_vec[candidate_idx].item()
        if sample_queries_used >= max_queries_per_sample:
            attacked_flag[candidate_idx] = True
            iteration += 1
            continue
        input_img = start_points[candidate_idx:candidate_idx+1]
        label = labels[candidate_idx:candidate_idx+1]
        target = targets[candidate_idx:candidate_idx+1] if targets is not None else None
        orig_img = inputs[candidate_idx:candidate_idx+1]

        remaining_sample_budget = max_queries_per_sample - sample_queries_used
        remaining_total_budget = maximum_queries - total_queries_used
        max_allowed_for_sample = maximum_queries - sample_queries_used
        allocated_queries = min(remaining_sample_budget, remaining_total_budget, max_allowed_for_sample)

        x_s, query_num, ae = _nes_attack_with_trajectory(
            victim_model=victim_model,
            attack_seed=input_img,
            initial_img=orig_img,
            label=label,
            target=target,
            eps=eps,
            sigma=nes_sigma,
            lr=nes_lr,
            num_samples=nes_num_samples,
            momentum=nes_momentum,
            maximum_queries=allocated_queries,
        )

        attacked_flag[candidate_idx] = True
        query_num_vec[candidate_idx] = min(query_num_vec[candidate_idx] + query_num, maximum_queries)
        total_queries_used += query_num
        best_adversarials[candidate_idx:candidate_idx+1] = ae

        with torch.no_grad():
            final_logits = victim_model(ae, None)
            final_prediction = final_logits.argmax(dim=1)

            if target is not None:
                success = final_prediction == target
            else:
                success = final_prediction != label
            success_vec[candidate_idx] = success.item()


        if len(x_s) > 0:
            trajectory_batch = torch.cat(x_s, dim=0)
            training_samples = torch.cat([training_samples, trajectory_batch], dim=0)

            with torch.no_grad():
                traj_logits = victim_model(trajectory_batch, None)
                victim_logits_list.extend(traj_logits)

        if ((iteration % fine_tune_frequency == 0)
            and (iteration > 0)
            and (victim_logits_list)):
            _fine_tune_surrogate_models(
                surrogate_models=surrogate_models,
                training_samples=training_samples[batch_size:],
                victim_logits_list=victim_logits_list,
                fine_tune_epochs=fine_tune_epochs,
                fine_tune_lr=fine_tune_lr,
            )

        iteration += 1
        if device.type == "mps":
            torch.mps.empty_cache()
            gc.collect()
    return best_adversarials


def _select_next_seed(
    start_points: Tensor,
    attacked_flag: Tensor,
    labels: Tensor,
    targets: Tensor | None,
    victim_model: CallableModel,
    query_num_vec: Tensor | None = None,
    max_queries_per_sample: int | None = None
 ) -> int | None:
    unattacked_mask = ~attacked_flag

    if query_num_vec is not None and max_queries_per_sample is not None:
        budget_mask = query_num_vec < max_queries_per_sample
        unattacked_mask = unattacked_mask & budget_mask

    unattacked_indices = unattacked_mask.nonzero(as_tuple=True)[0]
    if len(unattacked_indices) == 0:
        return None

    unattacked_samples = start_points[unattacked_indices]
    unattacked_labels = labels[unattacked_indices]
    unattacked_targets = targets[unattacked_indices] if targets is not None else None

    with torch.no_grad():
        logits = victim_model(unattacked_samples, None)
        if targets is not None:
            losses = torch.nn.functional.cross_entropy(
                logits, unattacked_targets, reduction="none"
            )
        else:
            losses = torch.nn.functional.cross_entropy(
                logits, unattacked_labels, reduction="none"
            )

    min_idx = losses.argmin().item()
    return unattacked_indices[min_idx].item()

def _nes_attack_with_trajectory(
    victim_model: CallableModel,
    attack_seed: Tensor,
    initial_img: Tensor,
    label: Tensor,
    target: Tensor | None,
    eps: float,
    sigma: float,
    lr: float,
    num_samples: int,
    momentum: float,
    maximum_queries: int,
) -> tuple[list[Tensor], int, Tensor]:
    x_s, queries_used, final_adv = _nes_attack_single_with_trajectory(
        victim_model=victim_model,
        attack_seed=attack_seed,
        initial_img=initial_img,
        label=label,
        target=target,
        eps=eps,
        sigma=sigma,
        lr=lr,
        num_samples=num_samples,
        momentum=momentum,
        plateau_drop=2.0,
        plateau_length=5,
        min_lr_ratio=200.0,
        maximum_queries=maximum_queries,
    )
    return x_s, queries_used, final_adv


@dataclass
class HybridHyperparameters:
    r""""Hyperparameters for the Hybrid Attack."""

    eps: float = 16 / 255
    p: str = "inf"
    maximum_queries: int = 1000
    fine_tune_frequency: int = 10
    fine_tune_epochs: int = 3
    fine_tune_lr: float = 1e-3
    transfer_alpha: float = 2.0 / 255.0
    transfer_iterations: int = 10
    nes_sigma: float = 0.1
    nes_lr: float = 0.02
    nes_num_samples: int = 20
    nes_momentum: float = 0.9


Hybrid: TransferAttack = partial(hybrid_attack, **asdict(HybridHyperparameters()))
