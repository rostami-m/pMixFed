"""Federated training loop shared by every algorithm.

Protocol notes (these are the fixes for J1/J2/D6):

* The adaptive mix factor is computed from each client's **validation** split.
* Early stopping and model selection use client-averaged **validation** accuracy.
* The **test** split is evaluated but never trained on, and the reported number is the
  test accuracy at the selected round. We also record the best test accuracy so the
  two reporting conventions can be compared rather than conflated -- but ``acc_test``
  at the selected round is what the paper reports.

Every algorithm, including the baselines, runs through this one loop with the same
optimizer, schedule, seeding, and evaluation code, so differences between rows of the
results table are attributable to the algorithms rather than to the harness (D7).
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .algorithms import AlgoSpec, get_algorithm, shared_blocks, trainable_blocks
from .mixing import adaptive_mu, lambda_profile, layer_blocks, mix_global_to_local
from .recent_baselines import fedlws_shrink, flayer_weights

__all__ = ["RunConfig", "federated_run"]


@dataclass
class RunConfig:
    algo: str = "pmixfed"
    dataset: str = "cifar10"
    model: str = "cnn"
    n_clients: int = 100
    frac: float = 0.1
    rounds: int = 100
    local_epochs: int = 4
    batch_size: int = 32
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    optimizer: str = "sgd"          # sgd | adam  (sgd is the default for ALL methods)
    lr_decay: float = 1.0           # multiplicative per-round decay
    partition: str = "shard"
    shard_per_user: int = 5
    alpha: float = 0.5
    val_fraction: float = 0.15
    mu_b: float = 2.0
    mu_floor: float = 0.0           # lower bound on mu; prevents the sharing collapse
    mu_signal: str = "ratio"        # 'ratio' (Eq.7, as submitted) | 'progress' | 'fixed'
    mu_ema: float = 0.0             # EMA smoothing on mu in [0,1); 0 disables
    pmix_alternating: bool = False  # FedAlt-style two-pass local update
    pmix_keep_head: bool = True     # never share the head (FedPer-style guarantee)
    fedlws_beta: float = 1.0        # FedLWS tau scaling
    flayer_temp: float = 1.0        # FLAYER weight temperature
    seed: int = 42
    device: str = "auto"
    eval_every: int = 1
    patience: int = 0               # 0 disables early stopping
    n_cold_start: int = 0           # clients excluded from all training rounds
    finetune_epochs: int = 0        # eval-time personalization epochs, on the VAL split
    finetune_part: str = "head"     # which blocks the eval-time finetune updates
    finetune_on: str = "train"      # 'train' (published protocol) or 'val'. NEVER test.
    equal_compute: bool = False     # halve local epochs for 2-pass methods (FedAlt)
    out_dir: str = "results"
    tag: str = ""
    threads: int = 4                # cap CPU threads so the machine stays usable
    hetero_depth: bool = False      # clients get CNNs of differing depth (reviewer J3)


def _pick_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, criterion) -> tuple[float, float]:
    """Return (accuracy %, mean loss). Never takes a gradient step."""
    if len(loader.dataset) == 0:
        return float("nan"), float("nan")
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total, loss_sum / total


def _make_optimizer(params, cfg: RunConfig, lr: float):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def _local_train(model, loader, cfg, lr, device, criterion, spec, train_blocks):
    """Run local epochs, updating only ``train_blocks``."""
    model.train().to(device)
    allowed = set(train_blocks)
    params = [p for n, p in model.named_parameters()
              if n.split(".")[0] in allowed and p.requires_grad]
    if not params:
        return 0.0
    opt = _make_optimizer(params, cfg, lr)

    # Freeze everything outside train_blocks by zeroing grads it may accumulate.
    total, seen = 0.0, 0
    for _ in range(cfg.local_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
            seen += y.size(0)
    return total / max(1, seen)



def _apply_broadcast(work, global_sd, spec, cfg, rnd, num_classes, device, criterion,
                     val_loader, shared, blocks, prev_global_acc, mu_state, client_id,
                     flayer_w=None, update_state=True, server_arch=None):
    """Write the incoming global model into `work`, per the algorithm's broadcast rule.

    This is used by BOTH the training loop and the evaluation loop. Having one
    implementation is deliberate: an earlier version duplicated the logic and the two
    copies drifted, so pMixFed was trained with the 'progress' mu signal but evaluated
    with the 'ratio' one, and FLAYER was evaluated as plain FedAvg. Any future change to
    broadcasting now necessarily applies to both paths.

    Returns the mu actually used (or None for methods without one).
    """
    from .models import build_model

    if spec.broadcast == "none":
        return None

    if spec.broadcast == "interpolate":
        gtmp = build_model(server_arch or cfg.model, cfg.dataset, num_classes).to(device)
        gtmp.load_state_dict(global_sd)
        acc_global, _ = evaluate(gtmp, val_loader, device, criterion)

        if cfg.mu_signal == "fixed":
            mu_raw = cfg.mu_floor if cfg.mu_floor > 0 else 0.5
        elif cfg.mu_signal == "progress":
            ref = prev_global_acc.get(client_id)
            if ref is None or ref <= 0:
                mu_raw = 0.5
            else:
                prog = (acc_global + 1e-6) / (ref + 1e-6)
                mu_raw = 1.0 - adaptive_mu(prog, t=rnd, T=cfg.rounds, b=cfg.mu_b)
            if update_state:
                prev_global_acc[client_id] = acc_global
        else:  # 'ratio' -- the rule as submitted
            acc_local, _ = evaluate(work, val_loader, device, criterion)
            ratio = (acc_local + 1e-6) / (acc_global + 1e-6)
            mu_raw = adaptive_mu(ratio, t=rnd, T=cfg.rounds, b=cfg.mu_b)

        mu_k = max(cfg.mu_floor, mu_raw)
        if cfg.mu_ema > 0:
            prior = mu_state.get(client_id, mu_k)
            mu_k = cfg.mu_ema * prior + (1 - cfg.mu_ema) * mu_k
            if update_state:
                mu_state[client_id] = mu_k

        mixed = mix_global_to_local(gtmp, work, mu_k)
        if cfg.pmix_keep_head:
            st_local = work.state_dict()
            for key in list(mixed):
                if key.split(".")[0] in set(getattr(work, "head_names", [])):
                    mixed[key] = st_local[key].clone()
        work.load_state_dict(mixed, strict=True)
        del gtmp
        return mu_k

    if spec.broadcast == "perlayer":
        st = work.state_dict()
        for key in st:
            b = key.split(".")[0]
            if not torch.is_floating_point(st[key]):
                continue
            if key not in global_sd or global_sd[key].shape != st[key].shape:
                continue
            a = 1.0 - (flayer_w.get(b, 0.5) if flayer_w else 0.5)
            st[key] = (a * global_sd[key].to(torch.float32)
                       + (1 - a) * st[key].to(torch.float32)).to(st[key].dtype)
        work.load_state_dict(st, strict=True)
        return None

    # 'full' | 'body' | 'head': copy exactly the shared blocks, keep the rest local.
    st = work.state_dict()
    for key in st:
        if key.split(".")[0] in set(shared) and key in global_sd \
                and global_sd[key].shape == st[key].shape:
            st[key] = global_sd[key].clone()
    work.load_state_dict(st, strict=True)
    return None


def federated_run(cfg: RunConfig) -> Dict:
    from .data import build_federated_data
    from .models import build_model

    torch.set_num_threads(max(1, cfg.threads))
    device = _pick_device(cfg.device)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        # cuDNN autotuning picks different kernels run-to-run, and some reductions are
        # atomically ordered, so identical seeds can differ by ~0.05% accuracy on GPU.
        # That is far below our seed-to-seed spread (~1-2 points) and so does not affect
        # any conclusion, but we pin it anyway so runs are exactly reproducible and
        # margins cannot be an artifact of kernel scheduling.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    spec = get_algorithm(cfg.algo)
    loaders, num_classes = build_federated_data(
        cfg.dataset, cfg.n_clients, partition=cfg.partition,
        shard_per_user=cfg.shard_per_user, alpha=cfg.alpha,
        val_fraction=cfg.val_fraction, batch_size=cfg.batch_size, seed=cfg.seed,
    )

    # Cold-start clients are held out of every training round (P14).
    all_ids = list(range(cfg.n_clients))
    cold = all_ids[cfg.n_clients - cfg.n_cold_start:] if cfg.n_cold_start else []
    active = [i for i in all_ids if i not in set(cold)]

    # Heterogeneous-depth cohort: the server keeps the deepest architecture, clients are
    # assigned round-robin across the family. Blocks a client lacks simply never match, so
    # the lambda_i = 0 rule leaves them out of broadcasting and aggregation.
    client_arch = {}
    if cfg.hetero_depth:
        from .models import HETERO_FAMILY
        for i in range(cfg.n_clients):
            client_arch[i] = HETERO_FAMILY[i % len(HETERO_FAMILY)]
        server_arch = HETERO_FAMILY[-1]
    else:
        server_arch = cfg.model

    global_model = build_model(server_arch, cfg.dataset, num_classes).to(device)
    blocks = layer_blocks(global_model)
    shared = shared_blocks(global_model, spec)
    train_blks = trainable_blocks(global_model, spec)

    # Stateful personalized parameters, one copy per client.
    if cfg.hetero_depth:
        local_state = {}
        for i in all_ids:
            m = build_model(client_arch[i], cfg.dataset, num_classes)
            local_state[i] = copy.deepcopy(m.state_dict())
            del m
    else:
        local_state = {i: copy.deepcopy(global_model.state_dict()) for i in all_ids}
    criterion = nn.CrossEntropyLoss()
    work = build_model(server_arch, cfg.dataset, num_classes).to(device)
    _work_cache: Dict[str, nn.Module] = {server_arch: work}

    def get_work(cid):
        """Scratch model matching client cid's architecture."""
        a = client_arch.get(cid, server_arch)
        if a not in _work_cache:
            _work_cache[a] = build_model(a, cfg.dataset, num_classes).to(device)
        return _work_cache[a]

    history: List[dict] = []
    # For mu_signal='progress': the accuracy the *global* model achieved on each
    # client's validation split the last time that client participated. This lets us
    # ask "is the global model improving?" instead of "is it better than my local
    # model?", the latter being unwinnable under label skew (a 3-class local model
    # beats a 10-class global one by construction).
    prev_global_acc: Dict[int, float] = {}
    flayer_state: Dict[int, Dict[str, float]] = {}
    pre_local: Dict[str, torch.Tensor] = {}
    mu_state: Dict[int, float] = {}
    best_val, best_round, best_test, since = -1.0, -1, float("nan"), 0
    rng = np.random.default_rng(cfg.seed)
    t_start = time.time()

    for rnd in range(cfg.rounds):
        lr = cfg.lr * (cfg.lr_decay ** rnd)
        m = max(1, int(round(cfg.frac * len(active))))
        selected = rng.choice(active, size=m, replace=False)

        # --- aggregation-stage mu: one shared value, from the PREVIOUS global model
        # measured on each participating client's validation split (Eq. 8).
        mu_agg = None
        prev_global_val = float("nan")
        if spec.needs_mu:
            accs, weights = [], []
            # Evaluate the PREVIOUS global model. Use a scratch model of the server
            # architecture: with heterogeneous-depth cohorts `work` may be a client-shaped
            # model, which cannot hold the server state dict.
            gscratch = _work_cache[server_arch]
            for k in selected:
                gscratch.load_state_dict(global_model.state_dict())
                a, _ = evaluate(gscratch, loaders[k]["val"], device, criterion)
                if not math.isnan(a):
                    accs.append(a)
                    weights.append(len(loaders[k]["train"].dataset))
            if accs:
                w = np.asarray(weights, dtype=float)
                # Omega_k-weighted mean accuracy of the previous global model (Eq. 8).
                prev_global_val = float(np.average(accs, weights=w))
                # Express it relative to the running best so the sigmoid argument is a
                # ratio in the same sense as the broadcasting stage: > 1 means the
                # previous global model is doing well, so retain more of it.
                reference = best_val if best_val > 0 else prev_global_val
                ratio = (prev_global_val + 1e-6) / (reference + 1e-6)
                mu_agg = adaptive_mu(ratio, t=rnd, T=cfg.rounds, b=cfg.mu_b)

        deltas, sizes, mus = [], [], []
        train_losses = []

        for k in selected:
            work = get_work(k)
            work.load_state_dict(local_state[k])

            # ---------------- broadcasting ----------------
            # Shared with the evaluation path via _apply_broadcast (see its docstring).
            mu_k = _apply_broadcast(
                work, global_model.state_dict(), spec, cfg, rnd, num_classes, device,
                criterion, loaders[k]["val"], shared, blocks, prev_global_acc,
                mu_state, k, flayer_w=flayer_state.get(k), update_state=True,
                server_arch=server_arch)
            if mu_k is not None:
                mus.append(mu_k)
            if spec.broadcast == "perlayer":
                pre_local = {kk: vv.detach().clone()
                             for kk, vv in work.state_dict().items()}

            # ---------------- local training ----------------
            if spec.local_update == "alternating" or (
                    spec.needs_mu and cfg.pmix_alternating):
                head = [b for b in blocks if b in set(getattr(work, "head_names", []))]
                body = [b for b in blocks if b not in set(head)]
                # FedAlt makes two passes over the local data (personal params, then
                # shared params), so at equal `local_epochs` it consumes 2x the
                # gradient steps of every other method. With equal_compute we halve
                # its per-pass budget so the comparison is compute-matched.
                sub = cfg
                if cfg.equal_compute and cfg.local_epochs > 1:
                    import dataclasses as _dc
                    sub = _dc.replace(cfg, local_epochs=max(1, cfg.local_epochs // 2))
                l1 = _local_train(work, loaders[k]["train"], sub, lr, device,
                                  criterion, spec, head)
                l2 = _local_train(work, loaders[k]["train"], sub, lr, device,
                                  criterion, spec, body)
                loss = 0.5 * (l1 + l2)
            else:
                loss = _local_train(work, loaders[k]["train"], cfg, lr, device,
                                    criterion, spec, train_blks)
            train_losses.append(loss)

            if spec.broadcast == "perlayer" and pre_local:
                flayer_state[k] = flayer_weights(pre_local, work.state_dict(),
                                                 blocks, cfg.flayer_temp)
            local_state[k] = copy.deepcopy(work.state_dict())
            deltas.append({key: v.detach().clone()
                           for key, v in work.state_dict().items()})
            sizes.append(len(loaders[k]["train"].dataset))

        # ---------------- aggregation ----------------
        w = np.asarray(sizes, dtype=float)
        w = w / w.sum()
        gst = global_model.state_dict()
        new_gst = {}
        for key in gst:
            if key.split(".")[0] not in set(shared):
                new_gst[key] = gst[key].clone()
                continue
            if not torch.is_floating_point(gst[key]):
                new_gst[key] = gst[key].clone()
                continue
            # Heterogeneous depth: aggregate over only the clients that possess this
            # layer, renormalizing their weights over that subset. A layer nobody has is
            # carried over unchanged.
            contrib = [(wi, d) for wi, d in zip(w, deltas)
                       if key in d and d[key].shape == gst[key].shape]
            if not contrib:
                new_gst[key] = gst[key].clone()
                continue
            tot = sum(wi for wi, _ in contrib)
            acc = torch.zeros_like(gst[key], dtype=torch.float32)
            for wi, d in contrib:
                acc += float(wi / tot) * d[key].to(torch.float32)
            new_gst[key] = acc.to(gst[key].dtype)

        if spec.needs_mu and mu_agg is not None:
            # Interpolate the aggregate with the PREVIOUS global model, layer-wise.
            # Because mu is shared here, this term is client-independent and can be
            # formed once at the server -- the simplification Reviewer YVGJ noted.
            lam = lambda_profile(mu_agg, len(blocks), "aggregation")
            lam_by_block = dict(zip(blocks, lam))
            for key in new_gst:
                b = key.split(".")[0]
                if b in lam_by_block and torch.is_floating_point(new_gst[key]):
                    a = lam_by_block[b]
                    new_gst[key] = a * gst[key].to(new_gst[key].dtype) + (1 - a) * new_gst[key]

        if spec.name == "FedLWS":
            new_gst = fedlws_shrink(gst, new_gst, deltas, blocks, beta=cfg.fedlws_beta)

        global_model.load_state_dict(new_gst, strict=True)

        # ---------------- evaluation ----------------
        if (rnd + 1) % cfg.eval_every == 0 or rnd == cfg.rounds - 1:
            val_accs, test_accs, test_losses = [], [], []
            for k in active:
                work = get_work(k)
                # Personalized evaluation: start from this client's stored local model
                # with shared blocks refreshed from the new global model.
                work.load_state_dict(local_state[k], strict=True)
                # Same broadcast rule as training. update_state=False so that scoring
                # every client each eval round does not perturb the mu/accuracy history
                # that the training loop depends on.
                _apply_broadcast(
                    work, global_model.state_dict(), spec, cfg, rnd, num_classes,
                    device, criterion, loaders[k]["val"], shared, blocks,
                    prev_global_acc, mu_state, k,
                    flayer_w=flayer_state.get(k), update_state=False,
                    server_arch=server_arch)

                # Evaluation-time personalization. FedBABU in particular is DEFINED
                # by fine-tuning the (otherwise frozen) head before scoring; omitting
                # it understates the method by ~30 points. Crucially this happens on
                # the client's VALIDATION split, never on test.
                if cfg.finetune_epochs > 0:
                    import dataclasses as _dc
                    ftcfg = _dc.replace(cfg, local_epochs=cfg.finetune_epochs)
                    ft_blocks = (
                        [b for b in blocks if b in set(getattr(work, "head_names", []))]
                        if cfg.finetune_part == "head" else blocks
                    )
                    # The published FedBABU/FedRep protocol fine-tunes on the client's
                    # own TRAINING data, which is legitimate (it is data the client
                    # already owns and trained on). We default to that rather than the
                    # small validation split, which is too small and overfits. Under no
                    # setting do we fine-tune on test -- that was defect D6.
                    ft_loader = loaders[k]["train" if cfg.finetune_on == "train" else "val"]
                    _local_train(work, ft_loader, ftcfg, cfg.lr, device,
                                 criterion, spec, ft_blocks)

                va, _ = evaluate(work, loaders[k]["val"], device, criterion)
                ta, tl = evaluate(work, loaders[k]["test"], device, criterion)
                if not math.isnan(va):
                    val_accs.append(va)
                if not math.isnan(ta):
                    test_accs.append(ta)
                    test_losses.append(tl)

            val_acc = float(np.mean(val_accs)) if val_accs else float("nan")
            test_acc = float(np.mean(test_accs)) if test_accs else float("nan")
            row = {
                "epoch": rnd,
                "loss_avg": float(np.mean(train_losses)) if train_losses else float("nan"),
                "loss_test": float(np.mean(test_losses)) if test_losses else float("nan"),
                "acc_val": val_acc,
                "acc_test": test_acc,
                "acc_test_std": float(np.std(test_accs)) if test_accs else float("nan"),
                "mu_mean": float(np.mean(mus)) if mus else None,
                "mu_agg": mu_agg,
                "lr": lr,
            }
            history.append(row)

            # Model selection on VALIDATION accuracy only.
            if val_acc == val_acc and val_acc > best_val:
                best_val, best_round, best_test, since = val_acc, rnd, test_acc, 0
            else:
                since += 1

            print(f"[{spec.name}] round {rnd + 1}/{cfg.rounds} "
                  f"loss={row['loss_avg']:.4f} val={val_acc:.2f} test={test_acc:.2f}"
                  + (f" mu={row['mu_mean']:.3f}" if row["mu_mean"] is not None else ""),
                  flush=True)

            if cfg.patience and since >= cfg.patience:
                print(f"early stopping at round {rnd + 1} (no val gain for {since} evals)")
                break

    # Cold-start clients: never trained, adapted briefly at eval time on val data only.
    cold_acc = None
    if cold:
        accs = []
        for k in cold:
            work.load_state_dict(global_model.state_dict())
            _local_train(work, loaders[k]["val"], cfg, cfg.lr, device,
                         criterion, spec, train_blks)
            a, _ = evaluate(work, loaders[k]["test"], device, criterion)
            if not math.isnan(a):
                accs.append(a)
        cold_acc = float(np.mean(accs)) if accs else None

    elapsed = time.time() - t_start
    result = {
        "config": asdict(cfg),
        "algorithm": spec.name,
        "history": history,
        "selected_round": best_round,
        "val_acc_at_selected": best_val,
        # This is the headline number: test accuracy at the round chosen by validation.
        "test_acc_at_selected": best_test,
        "test_acc_final": history[-1]["acc_test"] if history else float("nan"),
        # Reported only for transparency, never as the headline (J2).
        "test_acc_best_over_rounds": max((h["acc_test"] for h in history
                                          if h["acc_test"] == h["acc_test"]), default=float("nan")),
        "cold_start_acc": cold_acc,
        "n_cold_start": len(cold),
        "wall_time_s": elapsed,
        "device": str(device),
    }

    os.makedirs(cfg.out_dir, exist_ok=True)
    tag = cfg.tag or (f"{spec.name}-{cfg.dataset}-{cfg.model}-N{cfg.n_clients}"
                      f"-C{cfg.frac}-{cfg.partition}-s{cfg.seed}")
    with open(os.path.join(cfg.out_dir, f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
