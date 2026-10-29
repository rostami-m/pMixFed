"""Per-algorithm client behaviour, dispatched explicitly.

Defect D3 in the original implementation was the idiom

    if algo == "FedAvg" or "pFedMix" or "pMixFed":

which Python parses as ``(algo == "FedAvg") or "pFedMix" or "pMixFed"`` and which is
therefore *always true*. It appeared in four places, so every baseline silently executed
the full-model Adam path instead of its own update rule, and the algorithm-specific code
below each guard was dead. That is the most likely cause of the FedAlt/FedSim "training
collapses" the previous version of the paper interpreted as a scientific finding.

Here dispatch is by registry lookup on an exact string, and an unknown name raises.
There is no fall-through path.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .mixing import adaptive_mu, layer_blocks, mix_global_to_local

__all__ = ["ALGORITHMS", "get_algorithm", "AlgoSpec"]


class AlgoSpec:
    """How one algorithm initializes a client and which blocks it aggregates.

    ``broadcast``  writes the incoming global state into the client's model.
    ``aggregate_blocks`` names the blocks whose parameters are sent back to the server.
    ``trainable_blocks`` names the blocks updated by local SGD (the rest are frozen).
    """

    def __init__(
        self,
        name: str,
        *,
        broadcast: str,
        aggregate: str,
        local_update: str,
        needs_mu: bool = False,
        notes: str = "",
    ):
        self.name = name
        self.broadcast = broadcast      # 'full' | 'body' | 'interpolate'
        self.aggregate = aggregate      # 'full' | 'body'
        self.local_update = local_update  # 'full' | 'body' | 'head' | 'alternating'
        self.needs_mu = needs_mu
        self.notes = notes

    def __repr__(self):
        return f"AlgoSpec({self.name})"


# Baseline definitions follow FedBABU's flag semantics
# (--local_upt_part / --aggr_part), so our runs are directly comparable to the
# numbers in that codebase and to the CSVs previously used in the paper.
ALGORITHMS: Dict[str, AlgoSpec] = {
    # Critical control baseline: no federation at all. Under strong label skew a
    # client sees only 3-5 of 10 classes, so a purely local model faces a much easier
    # problem than the global one and can score very highly on its own test split.
    # Any personalized method that does not clearly beat 'local' is not demonstrating
    # that federation helps -- it is demonstrating that personalization helps, which
    # local training already does for free.
    "local": AlgoSpec(
        "Local-only", broadcast="none", aggregate="none", local_update="full",
        notes="No communication; each client trains alone. Sanity floor for PFL claims.",
    ),
    "fedavg": AlgoSpec(
        "FedAvg", broadcast="full", aggregate="full", local_update="full",
        notes="McMahan et al. 2017; full model shared and aggregated.",
    ),
    "fedper": AlgoSpec(
        "FedPer", broadcast="body", aggregate="body", local_update="full",
        notes="Arivazhagan et al. 2019; personalized head kept local, whole model trained.",
    ),
    "fedbabu": AlgoSpec(
        "FedBABU", broadcast="body", aggregate="body", local_update="body",
        notes="Oh et al. 2022; head frozen at init during federated training.",
    ),
    "lgfedavg": AlgoSpec(
        "LG-FedAvg", broadcast="head", aggregate="head", local_update="full",
        notes="Liang et al. 2020; shares the head, keeps local feature extractors.",
    ),
    "fedalt": AlgoSpec(
        "FedAlt", broadcast="body", aggregate="body", local_update="alternating",
        notes="Pillutla et al. 2022; alternating updates of shared and personal parts.",
    ),
    "fedsim": AlgoSpec(
        "FedSim", broadcast="body", aggregate="body", local_update="full",
        notes="Pillutla et al. 2022; simultaneous updates of shared and personal parts.",
    ),
    "fedlws": AlgoSpec(
        "FedLWS", broadcast="full", aggregate="full", local_update="full",
        notes="Shi et al. ICLR 2025; server-side layer-wise weight shrinking.",
    ),
    "flayer": AlgoSpec(
        "FLAYER", broadcast="perlayer", aggregate="full", local_update="full",
        notes="Chen et al. IJCAI 2025; per-layer personalization from update magnitude.",
    ),
    "pmixfed": AlgoSpec(
        "pMixFed", broadcast="interpolate", aggregate="full", local_update="full",
        needs_mu=True,
        notes="Ours; continuous depth-varying interpolation instead of a hard split.",
    ),
}


def get_algorithm(name: str) -> AlgoSpec:
    key = name.lower().replace("-", "").replace("_", "")
    if key not in ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {name!r}; available: {sorted(ALGORITHMS)}"
        )
    return ALGORITHMS[key]


def shared_blocks(model: nn.Module, spec: AlgoSpec) -> List[str]:
    """Which blocks this algorithm treats as globally shared."""
    all_blocks = layer_blocks(model)
    head = set(getattr(model, "head_names", []))
    if spec.aggregate == "none":
        return []
    if spec.aggregate == "full":
        return all_blocks
    if spec.aggregate == "body":
        return [b for b in all_blocks if b not in head]
    if spec.aggregate == "head":
        return [b for b in all_blocks if b in head]
    raise ValueError(spec.aggregate)


def trainable_blocks(model: nn.Module, spec: AlgoSpec) -> List[str]:
    all_blocks = layer_blocks(model)
    head = set(getattr(model, "head_names", []))
    if spec.local_update in ("full", "alternating"):
        return all_blocks
    if spec.local_update == "body":
        return [b for b in all_blocks if b not in head]
    if spec.local_update == "head":
        return [b for b in all_blocks if b in head]
    raise ValueError(spec.local_update)
