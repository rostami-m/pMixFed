"""Recent adaptive layer-wise baselines requested by Reviewer jm3q (J4).

These are the closest prior art to pMixFed: both reject uniform treatment of depth.
Neither official repository carries a license file, so these are clean-room
reimplementations from the papers and from reading the public reference code for the
formulae only -- no code is copied.

FedLWS (Shi et al., ICLR 2025, arXiv 2503.15111)
    Server-side *layer-wise weight shrinking*. After ordinary FedAvg aggregation, each
    layer of the new global model is scaled by

        gamma_l = ||w_l^{t}|| / (||w_l^{t}|| + tau_l * ||w_l^{t+1} - w_l^{t}||)
        tau_l   = clamp(beta * mean_k || d_l^k - mean_k d_l^k ||, min_tau, max_tau)

    where d_l^k is client k's layer-l update. tau_l is large when clients disagree, so
    layers with high client divergence are shrunk more aggressively. Formula transcribed
    from the reference implementation's server_funct.py.

    Relation to pMixFed: this is *also* a depth-varying convex-combination-like
    operation on the global model, but it shrinks toward the origin rather than
    interpolating toward the previous global model, and its per-layer coefficient is
    derived from gradient disagreement rather than from validation performance.

FLAYER (Chen et al., IJCAI 2025, arXiv 2412.07062)
    Per-layer personalization: each client keeps a per-layer mixing weight between the
    global and its own parameters, adapted from how much that layer changed during local
    training. Layers that move a lot locally are treated as more personal (kept local);
    layers that barely move are taken from the global model. We implement the mechanism
    described in the paper; the official repo ships only the CIFAR-100/20-client
    configuration and no license, so exact hyperparameters may differ from ours and we
    report ours explicitly.
"""

from __future__ import annotations

from typing import Dict, List

import torch

__all__ = ["fedlws_shrink", "flayer_weights"]


def fedlws_shrink(
    prev_global: Dict[str, torch.Tensor],
    new_global: Dict[str, torch.Tensor],
    client_states: List[Dict[str, torch.Tensor]],
    blocks: List[str],
    beta: float = 0.03,
    min_tau: float = 0.01,
    max_tau: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """Apply FedLWS layer-wise shrinking to a freshly aggregated global model.

    Defaults are the reference implementation's (``beta=0.03``, ``min_tau=0.01``,
    ``max_tau=0.2``). These matter a great deal: our first attempt used ``beta=1.0`` and
    ``max_tau=10.0``, which makes ``tau`` roughly 30x larger, so ``gamma`` falls to ~0.95
    or below and the global weights are multiplicatively shrunk toward zero every round
    (~1e-3 of their initial scale after 100 rounds). That produced the erratic 6-51%
    accuracies we initially measured; it was a hyperparameter error on our part, not a
    property of the method.
    """
    out = {k: v.clone() for k, v in new_global.items()}

    for block in blocks:
        prefix = f"{block}."
        keys = [k for k in new_global if k.startswith(prefix)
                and torch.is_floating_point(new_global[k])]
        if not keys:
            continue

        # Flatten the whole block so gamma is one scalar per layer block.
        def flat(d):
            return torch.cat([d[k].reshape(-1).to(torch.float32) for k in keys])

        w_prev, w_new = flat(prev_global), flat(new_global)
        grad_norm = torch.norm(w_new - w_prev, p=2)

        client_updates = [flat(cs) - w_prev for cs in client_states]
        if client_updates:
            stacked = torch.stack(client_updates)
            mean_update = stacked.mean(dim=0)
            divergence = torch.stack(
                [torch.norm(u - mean_update, p=2) for u in client_updates]
            ).mean()
        else:
            divergence = torch.tensor(0.0)

        tau = torch.clamp(beta * divergence, min=min_tau, max=max_tau)
        denom = torch.norm(w_prev, p=2) + tau * grad_norm
        gamma = (torch.norm(w_prev, p=2) / denom) if denom > 0 else torch.tensor(1.0)
        g = float(gamma)

        for k in keys:
            out[k] = (new_global[k].to(torch.float32) * g).to(new_global[k].dtype)
    return out


def flayer_weights(
    before: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
    blocks: List[str],
    temperature: float = 1.0,
) -> Dict[str, float]:
    """FLAYER per-layer personalization weights from local-update magnitude.

    Returns, per block, the weight to place on the client's OWN parameters. A layer that
    moved a lot during local training is more client-specific, so it keeps more of
    itself; a layer that barely moved defers to the global model.

    Magnitudes are normalized by the layer's own parameter norm so that layers of very
    different scale are comparable, then mapped through a bounded function to [0, 1].
    """
    rel = {}
    for block in blocks:
        prefix = f"{block}."
        keys = [k for k in after if k.startswith(prefix)
                and torch.is_floating_point(after[k])]
        if not keys:
            continue
        num = torch.sqrt(sum((after[k].to(torch.float32) - before[k].to(torch.float32))
                             .pow(2).sum() for k in keys))
        den = torch.sqrt(sum(before[k].to(torch.float32).pow(2).sum() for k in keys))
        rel[block] = float(num / den) if float(den) > 0 else 0.0

    if not rel:
        return {}
    vals = list(rel.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    # Scale within the round's observed range, so the split adapts to training phase
    # rather than to an absolute threshold that would need tuning per architecture.
    return {b: float(min(1.0, max(0.0, temperature * (v - lo) / span)))
            for b, v in rel.items()}
