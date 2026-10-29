"""Layer-wise interpolation between a global and a local model, in parameter space.

This module is a corrected reimplementation of the mixing logic from the original
pMixFed notebook. Four defects in that version are fixed here; each is covered by a
regression test in ``tests/test_mixing.py``.

D1  The original built its output dict from ``layer.state_dict()``, whose keys are
    *unprefixed* (``weight``) rather than module-qualified (``conv1.weight``).  The
    subsequent ``load_state_dict(..., strict=False)`` therefore silently dropped every
    mixed tensor as an unexpected key and the interpolation was a no-op.  We now build
    fully-qualified keys and *verify* that every intended key was consumed.

D2  The original aggregation loop did ``delta = weight_after``, which rebinds a Python
    local and mutates nothing, discarding the mixed result.  All writes here are either
    explicit dict assignments or in-place ``copy_`` calls.

D4  The original computed an adaptive mix factor and then passed a different, constant
    one to the aggregation call.  There is a single ``mu`` argument here.

D5  ``mu`` is computed by :func:`adaptive_mu`, which implements the paper's Eq. (6)
    including the round-dependence ``delta = t / T`` that the original code omitted.

Convention (stated once, used everywhere): ``lambda_i`` is always the weight on the
GLOBAL model at layer ``i``, and decreases monotonically with depth.  ``lambda_i = 1``
means the layer is taken entirely from the global model; ``lambda_i = 0`` means the
layer is left entirely local (personalized).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

__all__ = [
    "layer_blocks",
    "lambda_profile",
    "adaptive_mu",
    "interpolate_state",
    "mix_global_to_local",
    "mix_local_to_global",
]


def layer_blocks(model: nn.Module) -> List[str]:
    """Ordered names of top-level child modules that own trainable parameters.

    These are the units over which the interpolation profile is defined, so "layer i"
    in the paper means the i-th entry of this list. Modules without parameters
    (activations, pooling, flatten) are skipped: they carry no weights to mix.
    """
    names = []
    for name, child in model.named_children():
        if any(p.requires_grad for p in child.parameters(recurse=True)):
            names.append(name)
    return names


def lambda_profile(mu: float, n_layers: int, phase: str) -> List[float]:
    """Per-layer weight on the global model, as the closed form of Eq. (9).

    Broadcasting:  lambda_i = min(1, mu * (n - i))
    Aggregation:   lambda_i = max(0, 1 - i * mu)

    Both are stated in closed form deliberately. The original paper wrote these as
    piecewise definitions whose branch conditions referenced ``lambda_i`` itself,
    which is circular; two reviewers noted this independently.

    ``i`` is 1-indexed from the base layer, matching the paper's text.
    """
    if phase not in ("broadcast", "aggregation"):
        raise ValueError(f"phase must be 'broadcast' or 'aggregation', got {phase!r}")
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if not (0.0 <= mu <= 1.0):
        raise ValueError(f"mu must lie in [0, 1], got {mu}")

    out = []
    for i in range(1, n_layers + 1):
        if phase == "broadcast":
            lam = min(1.0, mu * (n_layers - i))
        else:
            lam = max(0.0, 1.0 - i * mu)
        out.append(float(lam))
    return out


def adaptive_mu(acc_ratio: float, t: int, T: int, b: float = 2.0) -> float:
    """The adaptive mix factor of Eq. (6).

        mu = 1 - sigmoid(delta * (acc_ratio ** b - 1)),   delta = t / T

    ``acc_ratio`` is the local model's accuracy divided by the global model's, both
    measured on the client's held-out *validation* split. A ratio above 1 (local model
    already better) drives ``mu`` down, so fewer layers are taken from the global
    model. A cold-start client has a ratio well below 1 and receives a large ``mu``.

    Note this includes the ``delta = t/T`` round-dependence that the original
    implementation omitted entirely (defect D5): early in training ``delta`` is near
    zero, so ``mu`` starts near 0.5 regardless of accuracy and the schedule sharpens
    as training proceeds.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if acc_ratio < 0:
        raise ValueError("acc_ratio must be non-negative")
    delta = t / T
    # Guard the exponentiation: acc_ratio can legitimately be 0 on a dead client.
    powed = acc_ratio ** b if acc_ratio > 0 else 0.0
    z = delta * (powed - 1.0)
    # Numerically stable logistic.
    if z >= 0:
        sig = 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        sig = e / (1.0 + e)
    return 1.0 - sig


def interpolate_state(
    global_model: nn.Module,
    local_model: nn.Module,
    lambdas: Sequence[float],
    blocks: Iterable[str] | None = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Return a full state dict interpolating global and local parameters per layer.

    ``result[p] = lambdas[i] * global[p] + (1 - lambdas[i]) * local[p]`` for every
    parameter ``p`` belonging to the i-th trainable block.

    Keys are module-qualified and taken from the *local* model's state dict, so the
    result can be loaded with ``strict=True``. This is the fix for D1: we assert that
    the keys we generate actually exist in the target, rather than relying on
    ``strict=False`` to quietly swallow mistakes.

    Parameters not owned by a trainable block (e.g. buffers such as BatchNorm running
    statistics) are passed through from the local model untouched.
    """
    g_state = global_model.state_dict()
    l_state = local_model.state_dict()

    if blocks is None:
        blocks = layer_blocks(local_model)
    blocks = list(blocks)

    if len(lambdas) != len(blocks):
        raise ValueError(
            f"got {len(lambdas)} lambdas for {len(blocks)} trainable blocks"
        )

    # Output is keyed strictly by the LOCAL model's state dict, so a client with fewer (or
    # more) blocks than the server receives something loadable with strict=True. Global-only
    # keys are simply never introduced.
    out = OrderedDict((k, v.detach().clone()) for k, v in l_state.items())

    mixed_keys = set()
    for lam, block in zip(lambdas, blocks):
        prefix = f"{block}."
        for key in l_state:
            if not key.startswith(prefix):
                continue
            if key not in g_state:
                # Layer exists locally but not globally (heterogeneous depth):
                # leave it purely local, i.e. lambda = 0 for this parameter.
                continue
            if g_state[key].shape != l_state[key].shape:
                continue
            g = g_state[key].detach().to(l_state[key].dtype)
            l = l_state[key].detach()
            if not torch.is_floating_point(l):
                # Integer buffers (e.g. num_batches_tracked) are not interpolated.
                continue
            out[key] = lam * g.to(l.device) + (1.0 - lam) * l
            mixed_keys.add(key)

    # D1 regression guard: if a caller passes blocks that match nothing, that is a
    # silent no-op of exactly the kind that invalidated the original results.
    if blocks and not mixed_keys:
        raise RuntimeError(
            "interpolation produced no mixed parameters; block names "
            f"{blocks!r} do not match any keys in the model state dict"
        )
    return out


def mix_global_to_local(
    global_model: nn.Module, local_model: nn.Module, mu: float
) -> "OrderedDict[str, torch.Tensor]":
    """Broadcasting stage: pull global knowledge into the local model."""
    blocks = layer_blocks(local_model)
    lambdas = lambda_profile(mu, len(blocks), "broadcast")
    return interpolate_state(global_model, local_model, lambdas, blocks)


def mix_local_to_global(
    global_model: nn.Module, local_model: nn.Module, mu: float
) -> "OrderedDict[str, torch.Tensor]":
    """Aggregation stage: fold one client's update into the global model.

    Returns a state dict in the *global* model's key space.
    """
    blocks = layer_blocks(global_model)
    lambdas = lambda_profile(mu, len(blocks), "aggregation")
    # Same interpolation, but the result lives in the global key space, so swap the
    # roles: we want lambda on the global model, written over global keys.
    g_state = global_model.state_dict()
    l_state = local_model.state_dict()
    out = OrderedDict((k, v.detach().clone()) for k, v in g_state.items())

    touched = set()
    for lam, block in zip(lambdas, blocks):
        prefix = f"{block}."
        for key in g_state:
            if not key.startswith(prefix) or key not in l_state:
                continue
            if g_state[key].shape != l_state[key].shape:
                continue
            if not torch.is_floating_point(g_state[key]):
                continue
            g = g_state[key].detach()
            l = l_state[key].detach().to(g.device)
            out[key] = lam * g + (1.0 - lam) * l
            touched.add(key)

    if blocks and not touched:
        raise RuntimeError("aggregation interpolation produced no mixed parameters")
    return out
