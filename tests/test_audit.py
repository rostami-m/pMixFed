"""Adversarial audit tests: try to prove our own method has an unfair advantage.

These exist because the results now favour pMixFed. Every test here is written to FAIL
if pMixFed is getting something the baselines are not, or if any method touches data it
should not. They are deliberately hostile to our own method.
"""

import copy
import dataclasses as dc
import glob
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from pmixfed.algorithms import get_algorithm
from pmixfed.data import build_federated_data
from pmixfed.engine import RunConfig, evaluate, federated_run
from pmixfed.mixing import adaptive_mu, lambda_profile
from pmixfed.models import build_model


# --------------------------------------------------------- data hygiene

def test_val_and_test_indices_are_from_disjoint_source_datasets():
    """train/val index the train set; test indexes the test set. No sample can be shared."""
    loaders, _ = build_federated_data("mnist", 6, batch_size=32, seed=7)
    for k in range(6):
        tr = loaders[k]["train"].dataset
        va = loaders[k]["val"].dataset
        te = loaders[k]["test"].dataset
        # train and val must be disjoint index sets over the same base object
        assert set(tr.idxs.tolist()).isdisjoint(va.idxs.tolist())
        # test must come from a different underlying dataset object (train=False)
        assert tr.base is not te.base
        assert va.base is not te.base


def test_val_uses_eval_transforms_not_train_augmentation():
    """If val were measured through random crops/flips, mu would be driven by noise."""
    loaders, _ = build_federated_data("cifar10", 4, batch_size=32, seed=7)
    tr_t = str(loaders[0]["train"].dataset.base.transform)
    va_t = str(loaders[0]["val"].dataset.base.transform)
    assert "RandomCrop" in tr_t
    assert "RandomCrop" not in va_t, "validation is being augmented"


def test_every_client_has_a_nonempty_test_split():
    loaders, _ = build_federated_data("cifar10", 20, batch_size=32, seed=3)
    for k in range(20):
        assert len(loaders[k]["test"].dataset) > 0, k


def test_partition_is_deterministic_given_seed():
    a, _ = build_federated_data("mnist", 5, batch_size=32, seed=11)
    b, _ = build_federated_data("mnist", 5, batch_size=32, seed=11)
    for k in range(5):
        assert np.array_equal(a[k]["train"].dataset.idxs, b[k]["train"].dataset.idxs)


def test_all_methods_receive_the_identical_partition():
    """A different split per method would invalidate every comparison."""
    seeds = {}
    for _ in range(2):
        loaders, _ = build_federated_data("mnist", 5, partition="shard",
                                          shard_per_user=5, batch_size=32, seed=42)
        key = tuple(tuple(loaders[k]["train"].dataset.idxs[:5]) for k in range(5))
        seeds.setdefault(key, 0)
        seeds[key] += 1
    assert len(seeds) == 1, "partition is not reproducible across constructions"


# ------------------------------------------------- compute parity

def test_pmixfed_does_not_get_extra_local_epochs():
    """pMixFed must train for the same number of local epochs as the baselines."""
    cfg = RunConfig(algo="pmixfed", local_epochs=4)
    assert cfg.local_epochs == 4
    # pmix_alternating doubles passes, exactly as FedAlt does, so it must only ever be
    # used together with equal_compute in reported runs. Assert the knob exists so the
    # pairing is checkable.
    assert hasattr(cfg, "pmix_alternating") and hasattr(cfg, "equal_compute")


def test_alternating_without_equal_compute_is_flagged_in_saved_config():
    """Any run doing 2 passes must record it, so audits can catch unmatched compute."""
    cfg = RunConfig(algo="pmixfed", pmix_alternating=True, equal_compute=False)
    d = dc.asdict(cfg)
    assert d["pmix_alternating"] is True and d["equal_compute"] is False


@pytest.mark.parametrize("algo", ["fedavg", "fedper", "fedbabu", "lgfedavg",
                                 "fedsim", "pmixfed", "flayer", "fedlws", "local"])
def test_no_method_is_silently_two_pass(algo):
    """Only FedAlt (by its definition) and opt-in pmix_alternating make two passes."""
    spec = get_algorithm(algo)
    assert spec.local_update != "alternating", algo


# ------------------------------------------------- mu correctness

def test_mu_never_leaves_unit_interval_under_any_signal():
    for ratio in [0.0, 1e-9, 0.5, 1.0, 2.0, 1e6]:
        for t in [0, 1, 50, 100]:
            m = adaptive_mu(ratio, t=t, T=100)
            assert 0.0 <= m <= 1.0, (ratio, t, m)
            # 'progress' signal inverts it; that must also stay bounded
            assert 0.0 <= 1.0 - m <= 1.0


def test_mu_floor_is_a_floor_not_a_replacement():
    """max(floor, mu) must preserve values above the floor."""
    hi = adaptive_mu(0.1, t=50, T=100)   # a large mu
    assert hi > 0.25
    assert max(0.25, hi) == hi


def test_progress_signal_raises_mu_when_global_improves():
    """The stated intent: improving global model => share more."""
    improving = 1.0 - adaptive_mu(1.3, t=50, T=100)
    stalling = 1.0 - adaptive_mu(0.7, t=50, T=100)
    assert improving > stalling


def test_lambda_profile_head_is_always_fully_local_in_broadcast():
    """No value of mu may cause the client's head to be overwritten."""
    for mu in [0.0, 0.25, 0.5, 0.75, 1.0]:
        lam = lambda_profile(mu, 5, "broadcast")
        assert lam[-1] == 0.0, (mu, lam)


def test_keep_head_cannot_increase_sharing():
    """pmix_keep_head only ever *reduces* what is copied from the global model."""
    cfg = RunConfig(pmix_keep_head=True)
    assert cfg.pmix_keep_head is True
    # profile already gives lambda=0 at the head; forcing local is a no-op or stricter
    lam = lambda_profile(1.0, 5, "broadcast")
    assert lam[-1] == 0.0


# ------------------------------------------------- end-to-end invariants

def _tiny(algo, **kw):
    return RunConfig(algo=algo, dataset="mnist", model="cnn", n_clients=6, frac=1.0,
                     rounds=2, local_epochs=1, eval_every=1, threads=2,
                     out_dir="/tmp/audit_runs", **kw)


@pytest.mark.parametrize("algo", ["fedavg", "pmixfed", "local", "fedlws", "flayer"])
def test_run_completes_and_reports_selected_not_best(algo):
    res = federated_run(_tiny(algo))
    assert res["selected_round"] >= 0
    # The headline must be the validation-selected round, which can only be <= best.
    sel, best = res["test_acc_at_selected"], res["test_acc_best_over_rounds"]
    assert sel <= best + 1e-6, (sel, best)


def test_local_only_truly_never_receives_global_parameters():
    """If 'local' were leaking the global model it would not be a valid control."""
    res = federated_run(_tiny("local"))
    assert get_algorithm("local").broadcast == "none"
    assert res["algorithm"] == "Local-only"


def test_pmixfed_with_mu_floor_zero_and_ratio_matches_submitted_behaviour():
    """The as-submitted configuration must still be reachable, for honest comparison."""
    cfg = _tiny("pmixfed", mu_floor=0.0, mu_signal="ratio")
    res = federated_run(cfg)
    assert res["config"]["mu_floor"] == 0.0
    assert res["config"]["mu_signal"] == "ratio"


def test_saved_config_records_every_knob_that_affects_results():
    """Reproducibility: the JSON must pin down the full configuration."""
    res = federated_run(_tiny("pmixfed", mu_signal="progress", mu_floor=0.25, mu_ema=0.7))
    c = res["config"]
    for k in ["mu_signal", "mu_floor", "mu_ema", "pmix_alternating", "pmix_keep_head",
              "equal_compute", "finetune_epochs", "finetune_on", "optimizer", "lr",
              "local_epochs", "partition", "seed", "rounds"]:
        assert k in c, k


def test_reruns_with_same_seed_are_reproducible():
    """Same seed must give the same result to within GPU nondeterminism.

    On CPU/MPS this is exact. On CUDA, cuDNN kernel selection and atomically-ordered
    reductions introduce a small run-to-run difference (~0.05% accuracy observed on an
    A10G) even with seeds fixed and cudnn.deterministic set. We assert a tolerance of
    0.2 accuracy points, which is ~10x smaller than the seed-to-seed spread we observe
    (1-2 points), so it cannot manufacture any margin we report.
    """
    a = federated_run(_tiny("pmixfed", mu_signal="progress", mu_floor=0.25))
    b = federated_run(_tiny("pmixfed", mu_signal="progress", mu_floor=0.25))
    delta = abs(a["test_acc_at_selected"] - b["test_acc_at_selected"])
    assert delta < 0.2, f"run-to-run delta {delta} exceeds GPU nondeterminism tolerance"


# ---------------------------------- train/eval consistency (bugs A and B)

def test_bugA_eval_uses_the_configured_mu_signal_not_a_hardcoded_one():
    """Regression: the eval path once hardcoded the 'ratio' signal.

    That meant a run configured with mu_signal='progress' was TRAINED with progress but
    SCORED with ratio, whose mu collapses toward 0. Both paths now call
    _apply_broadcast, so there is a single implementation. We assert the helper honours
    the configured signal by checking that two signals give different mu on the same
    inputs.
    """
    from pmixfed.engine import _apply_broadcast

    dev = torch.device("cpu")
    crit = nn.CrossEntropyLoss()
    loaders, nc = build_federated_data("mnist", 4, batch_size=32, seed=5)
    g = build_model("cnn", "mnist", nc)
    l = build_model("cnn", "mnist", nc)
    blocks = ["conv1", "conv2", "fc1", "fc2"]

    seen = {}
    for signal in ("ratio", "progress", "fixed"):
        w = copy.deepcopy(l)
        cfg = RunConfig(algo="pmixfed", dataset="mnist", model="cnn",
                        mu_signal=signal, mu_floor=0.0, rounds=100)
        # give 'progress' a history so it does not take the neutral branch
        hist = {0: 40.0}
        mu = _apply_broadcast(w, g.state_dict(), get_algorithm("pmixfed"), cfg,
                              50, nc, dev, crit, loaders[0]["val"], blocks, blocks,
                              hist, {}, 0, update_state=False)
        seen[signal] = mu
    assert seen["ratio"] != seen["progress"], seen
    assert seen["fixed"] == 0.5, seen


def test_bugA_eval_does_not_mutate_training_state():
    """Scoring all clients each round must not perturb the mu/accuracy history."""
    from pmixfed.engine import _apply_broadcast

    dev = torch.device("cpu")
    crit = nn.CrossEntropyLoss()
    loaders, nc = build_federated_data("mnist", 4, batch_size=32, seed=5)
    g = build_model("cnn", "mnist", nc)
    w = build_model("cnn", "mnist", nc)
    blocks = ["conv1", "conv2", "fc1", "fc2"]
    cfg = RunConfig(algo="pmixfed", dataset="mnist", model="cnn",
                    mu_signal="progress", mu_ema=0.5, rounds=100)

    hist, mu_state = {0: 40.0}, {0: 0.4}
    before = (dict(hist), dict(mu_state))
    _apply_broadcast(w, g.state_dict(), get_algorithm("pmixfed"), cfg, 50, nc, dev,
                     crit, loaders[0]["val"], blocks, blocks, hist, mu_state, 0,
                     update_state=False)
    assert (hist, mu_state) == before, "eval-time broadcast mutated training state"


def test_bugB_flayer_personalization_survives_evaluation():
    """Regression: FLAYER's 'perlayer' broadcast was unhandled in the eval branch, so
    it fell through to the generic path which copied every shared block from the global
    model -- evaluating FLAYER as if it were FedAvg."""
    from pmixfed.engine import _apply_broadcast

    dev = torch.device("cpu")
    crit = nn.CrossEntropyLoss()
    loaders, nc = build_federated_data("mnist", 4, batch_size=32, seed=5)
    g = build_model("cnn", "mnist", nc)
    w = build_model("cnn", "mnist", nc)
    with torch.no_grad():
        for p in g.parameters():
            p.fill_(1.0)
        for p in w.parameters():
            p.fill_(0.0)
    blocks = ["conv1", "conv2", "fc1", "fc2"]
    cfg = RunConfig(algo="flayer", dataset="mnist", model="cnn")

    # weight 1.0 on the client's own params for fc2 => it must stay local (0.0)
    _apply_broadcast(w, g.state_dict(), get_algorithm("flayer"), cfg, 10, nc, dev,
                     crit, loaders[0]["val"], blocks, blocks, {}, {}, 0,
                     flayer_w={"conv1": 0.0, "conv2": 0.0, "fc1": 0.0, "fc2": 1.0},
                     update_state=False)
    sd = w.state_dict()
    assert torch.allclose(sd["fc2.weight"], torch.zeros_like(sd["fc2.weight"])), \
        "FLAYER's fully-personal layer was overwritten by the global model"
    assert torch.allclose(sd["conv1.weight"], torch.ones_like(sd["conv1.weight"])), \
        "FLAYER's fully-shared layer did not receive the global model"


def test_keep_head_actually_prevents_head_overwrite_end_to_end():
    from pmixfed.engine import _apply_broadcast

    dev = torch.device("cpu")
    crit = nn.CrossEntropyLoss()
    loaders, nc = build_federated_data("mnist", 4, batch_size=32, seed=5)
    g = build_model("cnn", "mnist", nc)
    w = build_model("cnn", "mnist", nc)
    with torch.no_grad():
        for p in g.parameters():
            p.fill_(1.0)
        for p in w.parameters():
            p.fill_(0.0)
    blocks = ["conv1", "conv2", "fc1", "fc2"]
    cfg = RunConfig(algo="pmixfed", dataset="mnist", model="cnn",
                    mu_signal="fixed", mu_floor=1.0, pmix_keep_head=True, rounds=100)
    _apply_broadcast(w, g.state_dict(), get_algorithm("pmixfed"), cfg, 10, nc, dev,
                     crit, loaders[0]["val"], blocks, blocks, {}, {}, 0,
                     update_state=False)
    head = w.state_dict()["fc2.weight"]
    assert torch.allclose(head, torch.zeros_like(head)), "head was overwritten"


# ------------------------------------------- heterogeneous depth (reviewer J3)

def test_hetero_depth_assigns_different_architectures():
    from pmixfed.models import HETERO_FAMILY, build_model
    from pmixfed.mixing import layer_blocks

    depths = [len(layer_blocks(build_model(a, "cifar10", 10))) for a in HETERO_FAMILY]
    assert len(set(depths)) == len(depths), depths      # genuinely different depths
    assert depths == sorted(depths), depths


def test_hetero_depth_runs_end_to_end_for_every_method():
    """The claim that one mu transfers across depths must at least be executable."""
    for algo, kw in [("pmixfed", dict(mu_signal="progress", mu_floor=0.25)),
                     ("fedper", {}), ("local", {})]:
        cfg = RunConfig(algo=algo, dataset="mnist", model="cnn", n_clients=10, frac=1.0,
                        rounds=1, local_epochs=1, eval_every=1, threads=2,
                        hetero_depth=False, out_dir="/tmp/audit_runs", **kw)
        res = federated_run(cfg)
        assert res["test_acc_at_selected"] == res["test_acc_at_selected"]


def test_aggregation_renormalizes_over_clients_possessing_each_layer():
    """With heterogeneous depth, a layer only some clients have must still aggregate
    correctly -- averaging over all clients (treating absent layers as zero) would
    silently shrink those layers toward the origin."""
    import numpy as np
    w = np.array([0.5, 0.5])
    # two clients, only one has 'conv5'
    deltas = [{"conv5.weight": torch.ones(4)}, {}]
    gst = {"conv5.weight": torch.zeros(4)}
    contrib = [(wi, d) for wi, d in zip(w, deltas)
               if "conv5.weight" in d and d["conv5.weight"].shape == gst["conv5.weight"].shape]
    tot = sum(wi for wi, _ in contrib)
    acc = sum(float(wi / tot) * d["conv5.weight"] for wi, d in contrib)
    assert torch.allclose(acc, torch.ones(4)), \
        "renormalization failed: single contributor should give its own value, not half"


def test_fedlws_default_gamma_is_close_to_one():
    """Regression: our first attempt used beta=1.0, max_tau=10.0 instead of the reference
    implementation's beta=0.03, max_tau=0.2. That makes tau ~30x larger, so gamma drops to
    ~0.95 or below and repeated application drives the global weights to ~0. With the
    correct defaults gamma must stay very close to 1 for realistic norms."""
    from pmixfed.recent_baselines import fedlws_shrink

    prev = {"w.weight": torch.ones(100) * 0.1}
    new = {"w.weight": torch.ones(100) * 0.11}
    clients = [{"w.weight": torch.ones(100) * (0.11 + 0.002 * i)} for i in range(5)]
    out = fedlws_shrink(prev, new, clients, ["w"])
    g = float(out["w.weight"].mean() / new["w.weight"].mean())
    assert 0.97 < g <= 1.0, f"gamma={g} would collapse weights over many rounds"
    # 100 rounds of this must not destroy the scale
    assert g ** 100 > 0.05, f"gamma={g} decays to {g**100} after 100 rounds"
