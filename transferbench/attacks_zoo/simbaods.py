r"""SimBA attack implementation with ODS.

Simba code is based on the original implementation from `https://github.com/cg563/simple-blackbox-attack/blob/master/simba.py``
while ODS mechamism has been added to the original code, following the paper
Diversity Can Be Transferred: Output Diversification for White- and Black-box Attacks `https://arxiv.org/pdf/2003.06878`.

"""

import torch
import torch.nn.functional as F
import utils

from transferbench.types import CallableModel
from torch import Tensor
from typing import Optional, Tuple


def get_probs(
    model: CallableModel, x: Tensor, y: Tensor, remaining: Optional[Tensor] = None
) -> Tensor:
    output = model(x, remaining).cpu()
    probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
    return torch.diag(probs)


def get_preds(model: CallableModel, x: Tensor, remaining: Tensor) -> Tensor:
    """Get the predicted labels for the input images."""
    output = model(x, remaining).cpu()
    _, preds = output.data.max(1)
    return preds


# runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
# (for targeted attack) <labels_batch>
def simba_batch(
    victim_model: CallableModel,
    surrogate_model: list[CallableModel],
    images_batch,
    labels_batch,
    targets_batch,
    max_iters,
    freq_dims,
    stride,
    epsilon,
    linf_bound=0.0,
    order="rand",
    targeted=False,
    pixel_attack=False,
    log_every=1,
):
    batch_size = images_batch.size(0)
    image_size = images_batch.size(2)
    # sample a random ordering for coordinates independently per batch element
    indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]

    expand_dims = freq_dims if order == "rand" else image_size
    n_dims = 3 * expand_dims * expand_dims
    x = torch.zeros(batch_size, n_dims)
    # logging tensors
    probs = torch.zeros(batch_size, max_iters)
    succs = torch.zeros(batch_size, max_iters)
    queries = torch.zeros(batch_size, max_iters)
    l2_norms = torch.zeros(batch_size, max_iters)
    linf_norms = torch.zeros(batch_size, max_iters)
    prev_probs = get_probs(victim_model, images_batch, labels_batch)
    labels = labels_batch if not targeted else targets_batch
    preds = get_preds(victim_model, images_batch, labels)
    if pixel_attack:
        trans = lambda z: z
    else:
        trans = lambda z: utils.block_idct(
            z, block_size=image_size, linf_bound=linf_bound
        )
    remaining_indices = torch.arange(0, batch_size).long()
    for k in range(max_iters):
        dim = indices[k]
        expanded = (
            images_batch[remaining_indices]
            + trans(self.expand_vector(x[remaining_indices], expand_dims))
        ).clamp(0, 1)
        perturbation = trans(self.expand_vector(x, expand_dims))
        l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
        linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
        preds_next = self.get_preds(expanded)
        preds[remaining_indices] = preds_next
        if targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        # check if all images are misclassified and stop early
        if remaining.sum() == 0:
            adv = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
            probs_k = self.get_probs(adv, labels_batch)
            probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
            succs[:, k:] = torch.ones(batch_size, max_iters - k)
            queries[:, k:] = torch.zeros(batch_size, max_iters - k)
            break
        remaining_indices = torch.arange(0, batch_size)[remaining].long()
        if k > 0:
            succs[:, k] = ~remaining
        diff = torch.zeros(remaining.sum(), n_dims)
        diff[:, dim] = epsilon
        left_vec = x[remaining_indices] - diff
        right_vec = x[remaining_indices] + diff
        # trying negative direction
        adv = (
            images_batch[remaining_indices]
            + trans(self.expand_vector(left_vec, expand_dims))
        ).clamp(0, 1)
        left_probs = self.get_probs(adv, labels_batch[remaining_indices])
        queries_k = torch.zeros(batch_size)
        # increase query count for all images
        queries_k[remaining_indices] += 1
        if targeted:
            improved = left_probs.gt(prev_probs[remaining_indices])
        else:
            improved = left_probs.lt(prev_probs[remaining_indices])
        # only increase query count further by 1 for images that did not improve in adversarial loss
        if improved.sum() < remaining_indices.size(0):
            queries_k[remaining_indices[~improved]] += 1
        # try positive directions
        adv = (
            images_batch[remaining_indices]
            + trans(self.expand_vector(right_vec, expand_dims))
        ).clamp(0, 1)
        right_probs = self.get_probs(adv, labels_batch[remaining_indices])
        if targeted:
            right_improved = right_probs.gt(
                torch.max(prev_probs[remaining_indices], left_probs)
            )
        else:
            right_improved = right_probs.lt(
                torch.min(prev_probs[remaining_indices], left_probs)
            )
        probs_k = prev_probs.clone()
        # update x depending on which direction improved
        if improved.sum() > 0:
            left_indices = remaining_indices[improved]
            left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
            x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
            probs_k[left_indices] = left_probs[improved]
        if right_improved.sum() > 0:
            right_indices = remaining_indices[right_improved]
            right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
            x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
            probs_k[right_indices] = right_probs[right_improved]
        probs[:, k] = probs_k
        queries[:, k] = queries_k
        prev_probs = probs[:, k]
        if (k + 1) % log_every == 0 or k == max_iters - 1:
            print(
                "Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f"
                % (
                    k + 1,
                    queries.sum(1).mean(),
                    probs[:, k].mean(),
                    remaining.float().mean(),
                )
            )
    expanded = (images_batch + trans(self.expand_vector(x, expand_dims))).clamp(0, 1)
    preds = self.get_preds(expanded)
    if targeted:
        remaining = preds.ne(labels_batch)
    else:
        remaining = preds.eq(labels_batch)
    succs[:, max_iters - 1] = ~remaining
    return expanded, probs, succs, queries, l2_norms, linf_norms
