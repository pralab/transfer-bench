"""Natural Evolution Strategy (NES) attack implementation.

NES is a query-based optimization attack that estimates gradients using
random sampling and evolutionary strategies.

This implementation is based on the PyTorch version adapted from the original
hybrid meta-attack creators' code.
"""

import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from transferbench.types import CallableModel, TransferAttack


def nes(
    victim_model: CallableModel,
    surrogate_models: list[CallableModel],
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor] = None,
    eps: float = 16 / 255,
    p: float | str = "inf",
    maximum_queries: int = 1000,
    sigma: float = 0.1,
    max_lr: float = 0.02,
    samples_per_draw: int = 20,
    nes_batch_size: int = 10,
    momentum: float = 0.9,
    plateau_length: int = 5,
    plateau_drop: float = 2.0,
    min_lr: float = 0.0001,
    print_every: int = 100,
    lower: float = 0.0,
    upper: float = 1.0,
) -> Tensor:
    if p != "inf":
        raise NotImplementedError("Only L-infinity norm supported currently")

    device = inputs.device
    batch_size = inputs.size(0)

    is_targeted = targets is not None
    target_labels = targets if is_targeted else labels
    attack_sign = 1.0 if is_targeted else -1.0

    max_iters = int(torch.ceil(torch.tensor(maximum_queries / samples_per_draw)).item())

    lower_bound = torch.clamp(inputs - eps, lower, upper)
    upper_bound = torch.clamp(inputs + eps, lower, upper)

    adv_examples = inputs.clone()

    with torch.no_grad():
        original_logits = victim_model(inputs, None)
        original_preds = original_logits.argmax(dim=1)

    print(f'Original predictions: {original_preds.cpu().numpy()}')

    num_queries = torch.zeros(batch_size, dtype=torch.long, device=device)
    g = torch.zeros_like(adv_examples)
    last_losses = [[] for _ in range(batch_size)]
    current_max_lr = torch.full((batch_size,), max_lr, device=device)

    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    max_iters_per_sample = maximum_queries // samples_per_draw
    actual_max_iters = min(max_iters, max_iters_per_sample)

    x_s = []

    for iteration in range(actual_max_iters):
        if not active_mask.any():
            print(f"All samples converged at iteration {iteration}")
            break

        x_s.append(adv_examples.clone())
        start_time = time.time()

        prev_g = g.clone()

        losses, estimated_grad = _get_gradient_estimation(
            victim_model=victim_model,
            adv_batch=adv_examples,
            target_batch=target_labels,
            active_indices=active_mask,
            samples_per_draw=samples_per_draw,
            nes_batch_size=nes_batch_size,
            sigma=sigma,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            is_targeted=is_targeted,
        )

        g = momentum * prev_g + (1.0 - momentum) * estimated_grad

        with torch.no_grad():
            current_logits = victim_model(adv_examples, None)
            current_preds = current_logits.argmax(dim=1)

            if is_targeted:
                success_mask = current_preds == target_labels
            else:
                success_mask = current_preds != original_preds

            newly_successful = success_mask & active_mask
            if newly_successful.any():
                success_indices = torch.where(newly_successful)[0]
                for idx in success_indices:
                    print(f'[Sample {idx}] Success at iteration {iteration} with {num_queries[idx].item()} queries')
                    print(f'  Predicted: {current_preds[idx].item()}, Target: {target_labels[idx].item()}')
                active_mask &= ~success_mask

        for idx in range(batch_size):
            if not active_mask[idx]:
                continue

            last_losses[idx].append(losses[idx].item())
            last_losses[idx] = last_losses[idx][-plateau_length:]

            if len(last_losses[idx]) == plateau_length:
                if last_losses[idx][-1] > last_losses[idx][0]:
                    if current_max_lr[idx] > min_lr:
                        current_max_lr[idx] = max(current_max_lr[idx] / plateau_drop, min_lr)
                        if iteration % print_every == 0:
                            print(f"[Sample {idx}] Annealing learning rate to {current_max_lr[idx]:.4f}")
                    last_losses[idx] = []

        with torch.no_grad():
            lr_expanded = current_max_lr.view(-1, 1, 1, 1)
            update = attack_sign * lr_expanded * torch.sign(g)
            adv_examples = torch.where(
                active_mask.view(-1, 1, 1, 1),
                torch.clamp(adv_examples - update, lower_bound, upper_bound),
                adv_examples
            )

        num_queries[active_mask] += samples_per_draw

        if (iteration + 1) % print_every == 0:
            active_count = active_mask.sum().item()
            avg_loss = losses[active_mask].mean().item() if active_count > 0 else 0.0
            avg_lr = current_max_lr[active_mask].mean().item() if active_count > 0 else 0.0
            avg_queries = num_queries[active_mask].float().mean().item() if active_count > 0 else 0.0

            print(f'Step {iteration:05d}: active {active_count}/{batch_size}, '
                  f'avg queries {avg_queries:.1f}, avg loss {avg_loss:.6f}, '
                  f'avg lr {avg_lr:.4f} (time {time.time() - start_time:.2f}s)')

    print(f"\nAttack completed:")
    for idx in range(batch_size):
        with torch.no_grad():
            final_logits = victim_model(adv_examples[idx:idx+1], None)
            final_pred = final_logits.argmax(dim=1).item()

        status = "SUCCESS" if (is_targeted and final_pred == target_labels[idx].item()) or \
                            (not is_targeted and final_pred != original_preds[idx].item()) else "FAILED"
        print(f"  Sample {idx}: {status} - Original: {original_preds[idx].item()}, "
              f"Final: {final_pred}, Target: {target_labels[idx].item()}, "
              f"Queries: {num_queries[idx].item()}")

    return adv_examples


def _get_gradient_estimation(
    victim_model: CallableModel,
    adv_batch: Tensor,
    target_batch: Tensor,
    active_indices: Tensor,
    samples_per_draw: int,
    nes_batch_size: int,
    sigma: float,
    upper_bound: Tensor,
    lower_bound: Tensor,
    is_targeted: bool,
) -> tuple[Tensor, Tensor]:
    device = adv_batch.device
    B, C, H, W = adv_batch.shape

    num_batches = samples_per_draw // nes_batch_size

    all_grads = []
    all_losses = []

    active_count = active_indices.sum().item()
    if active_count == 0:
        return torch.zeros(B, device=device), torch.zeros_like(adv_batch)

    for _ in range(num_batches):
        noise_pos = torch.randn(B, nes_batch_size // 2, C, H, W, device=device)
        noise = torch.cat([noise_pos, -noise_pos], dim=1)

        scale = (upper_bound - lower_bound).unsqueeze(1)
        scaled_noise = sigma * noise * scale

        adv_expanded = adv_batch.unsqueeze(1).expand(-1, nes_batch_size, -1, -1, -1)
        eval_points = adv_expanded + scaled_noise

        eval_points_flat = eval_points.view(-1, C, H, W)

        with torch.no_grad():
            logits = victim_model(eval_points_flat, None)

        targets_expanded = target_batch.unsqueeze(1).expand(-1, nes_batch_size).reshape(-1)

        loss_flat = F.cross_entropy(logits, targets_expanded, reduction='none')

        loss_batch = loss_flat.view(B, nes_batch_size)

        loss_expanded = loss_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        grad_batch = (loss_expanded * noise).mean(dim=1) / sigma

        all_losses.append(loss_batch.mean(dim=1))
        all_grads.append(grad_batch)

    avg_loss = torch.stack(all_losses).mean(dim=0)
    avg_grad = torch.stack(all_grads).mean(dim=0)

    avg_grad = avg_grad * active_indices.view(-1, 1, 1, 1)
    avg_loss = avg_loss * active_indices.float()

    return avg_loss, avg_grad


@dataclass
class NESHyperparameters:
    eps: float = 16 / 255
    p: str = "inf"
    maximum_queries: int = 4000
    sigma: float = 0.1
    max_lr: float = 0.02
    samples_per_draw: int = 50
    nes_batch_size: int = 50
    momentum: float = 0.9
    plateau_length: int = 5
    plateau_drop: float = 2.0
    min_lr: float = 0.0001
    print_every: int = 100
    lower: float = 0.0
    upper: float = 1.0


def _nes_attack_single_with_trajectory(
    victim_model: CallableModel,
    attack_seed: Tensor,
    initial_img: Tensor,
    label: Tensor,
    target: Optional[Tensor],
    eps: float,
    sigma: float,
    lr: float,
    num_samples: int,
    momentum: float,
    plateau_drop: float,
    plateau_length: int,
    min_lr_ratio: float,
    maximum_queries: int,
) -> tuple[list[Tensor], int, Tensor]:
    if maximum_queries <= 0:
        return [], 0, attack_seed
    
    device = attack_seed.device
    is_targeted = target is not None
    target_label = target if is_targeted else label
    attack_sign = 1.0 if is_targeted else -1.0
    
    lower_bound = torch.clamp(initial_img - eps, 0.0, 1.0)
    upper_bound = torch.clamp(initial_img + eps, 0.0, 1.0)
    
    adv_example = attack_seed.clone()
    g = torch.zeros_like(adv_example)
    last_losses = []
    current_lr = lr
    min_lr = lr / min_lr_ratio
    
    with torch.no_grad():
        original_logits = victim_model(initial_img, None)
        original_pred = original_logits.argmax(dim=1)
    
    trajectory = []
    queries_used = 0
    
    samples_per_iteration = num_samples
    max_iterations = maximum_queries // samples_per_iteration
    
    for iteration in range(max_iterations):
        if queries_used >= maximum_queries:
            break
            
        trajectory.append(adv_example.clone())
        
        with torch.no_grad():
            current_logits = victim_model(adv_example, None)
            current_pred = current_logits.argmax(dim=1)
            
            if is_targeted:
                success = current_pred == target_label
            else:
                success = current_pred != original_pred
                
            if success.item():
                queries_used += 1
                break
        
        remaining_queries = maximum_queries - queries_used
        if remaining_queries < samples_per_iteration:
            samples_per_iteration = remaining_queries
            
        if samples_per_iteration <= 0:
            break
            
        noise = torch.randn(samples_per_iteration, *adv_example.shape[1:], device=device)
        
        eval_points = adv_example + sigma * noise.unsqueeze(0)
        eval_points = torch.clamp(eval_points.squeeze(0), lower_bound, upper_bound)
        
        with torch.no_grad():
            logits = victim_model(eval_points, None)
            
        target_expanded = target_label.expand(samples_per_iteration)
        losses = F.cross_entropy(logits, target_expanded, reduction='none')
        
        loss_expanded = losses.view(-1, 1, 1, 1)
        estimated_grad = (loss_expanded * noise).mean(dim=0, keepdim=True) / sigma
        
        g = momentum * g + (1.0 - momentum) * estimated_grad
        
        avg_loss = losses.mean().item()
        last_losses.append(avg_loss)
        last_losses = last_losses[-plateau_length:]
        
        if len(last_losses) == plateau_length:
            if last_losses[-1] > last_losses[0]:
                if current_lr > min_lr:
                    current_lr = max(current_lr / plateau_drop, min_lr)
                last_losses = []
        
        with torch.no_grad():
            update = attack_sign * current_lr * torch.sign(g)
            adv_example = torch.clamp(adv_example - update, lower_bound, upper_bound)
        
        queries_used += samples_per_iteration
    
    return trajectory, queries_used, adv_example


NES: TransferAttack = partial(nes, **asdict(NESHyperparameters()))