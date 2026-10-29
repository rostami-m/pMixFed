"""Regression tests for algorithm dispatch (defect D3) and the data protocol (D6/J1)."""

import numpy as np
import pytest
import torch

from pmixfed.algorithms import ALGORITHMS, get_algorithm, shared_blocks, trainable_blocks
from pmixfed.models import CNNCifar, CNNMnist, build_model


# ------------------------------------------------------------------------- D3

def test_d3_chained_or_truthiness_is_always_true():
    """Documents the original bug so nobody reintroduces the idiom.

    `algo == "FedAvg" or "pFedMix" or "pMixFed"` parses as
    `(algo == "FedAvg") or "pFedMix" or "pMixFed"`, whose value is a non-empty
    string -- truthy -- for every possible `algo`.
    """
    for algo in ["FedAlt", "FedSim", "FedBABU", "Ditto", "literally anything"]:
        assert bool(algo == "FedAvg" or "pFedMix" or "pMixFed") is True


def test_d3_dispatch_is_exact_and_unknown_names_raise():
    with pytest.raises(ValueError, match="unknown algorithm"):
        get_algorithm("NotAnAlgorithm")


def test_d3_each_algorithm_has_a_distinct_behaviour_signature():
    """If two baselines collapse to the same config, they are not really being compared."""
    sigs = {}
    for key, spec in ALGORITHMS.items():
        sigs.setdefault(
            (spec.broadcast, spec.aggregate, spec.local_update, spec.needs_mu), []
        ).append(key)
    # Two collisions are legitimate in this coarse signature:
    #   fedper/fedsim  - both share the body and train the whole model
    #   fedavg/fedlws  - FedLWS is FedAvg plus a server-side per-layer shrink, which
    #                    this signature does not capture (it is applied post-aggregation)
    allowed = ({"fedper", "fedsim"}, {"fedavg", "fedlws"})
    collisions = {k: v for k, v in sigs.items() if len(v) > 1}
    assert all(set(v) in allowed for v in collisions.values()), collisions


def test_baselines_do_not_all_aggregate_the_full_model():
    """The D3 bug made every method behave like full-model FedAvg."""
    aggr = {k: s.aggregate for k, s in ALGORITHMS.items()}
    assert len(set(aggr.values())) > 1, aggr
    assert aggr["fedavg"] == "full"
    assert aggr["fedbabu"] == "body"
    assert aggr["lgfedavg"] == "head"


def test_fedbabu_freezes_the_head_but_fedper_does_not():
    """The defining difference between the two, per FedBABU's own flags."""
    m = CNNCifar(10)
    babu = trainable_blocks(m, get_algorithm("fedbabu"))
    per = trainable_blocks(m, get_algorithm("fedper"))
    assert "fc" not in babu, babu
    assert "fc" in per, per


def test_lgfedavg_shares_the_head_only():
    m = CNNCifar(10)
    assert shared_blocks(m, get_algorithm("lgfedavg")) == ["fc"]


def test_pmixfed_is_the_only_method_needing_mu():
    needs = [k for k, s in ALGORITHMS.items() if s.needs_mu]
    assert needs == ["pmixfed"]


# ---------------------------------------------------------------- data / D6

def test_d6_val_split_is_disjoint_from_train():
    """The mix factor and early stopping must not see training data."""
    from pmixfed.data import ClientSplit

    with pytest.raises(ValueError, match="train/val index overlap"):
        ClientSplit(train=np.array([1, 2, 3]), val=np.array([3, 4]), test=np.array([0]))


def test_d6_clientsplit_accepts_disjoint_sets():
    from pmixfed.data import ClientSplit

    s = ClientSplit(train=np.array([1, 2]), val=np.array([3]), test=np.array([1, 2]))
    assert len(s.train) == 2 and len(s.val) == 1


# ------------------------------------------------------------------- models

@pytest.mark.parametrize(
    "name,dataset,shape",
    [("cnn", "mnist", (2, 1, 28, 28)), ("cnn", "cifar10", (2, 3, 32, 32))],
)
def test_models_forward_and_split(name, dataset, shape):
    m = build_model(name, dataset, 10)
    out = m(torch.randn(*shape))
    assert out.shape == (2, 10)
    assert m.head_blocks() and m.body_blocks()
    assert not set(m.head_blocks()) & set(m.body_blocks())


# ------------------------------------------------- recent baselines (J4)

def test_fedlws_shrinks_toward_origin_and_is_bounded():
    """gamma = ||w|| / (||w|| + tau*||dw||) is in (0,1], so it can only shrink."""
    from pmixfed.recent_baselines import fedlws_shrink

    prev = {"conv1.weight": torch.ones(4, 3, 3, 3)}
    new = {"conv1.weight": torch.ones(4, 3, 3, 3) * 2.0}
    clients = [{"conv1.weight": torch.ones(4, 3, 3, 3) * (2.0 + 0.1 * i)}
               for i in range(3)]
    out = fedlws_shrink(prev, new, clients, ["conv1"], beta=1.0)
    r = out["conv1.weight"].abs().mean() / new["conv1.weight"].abs().mean()
    assert 0.0 < float(r) <= 1.0, r


def test_fedlws_shrinks_more_when_clients_disagree():
    from pmixfed.recent_baselines import fedlws_shrink

    prev = {"w.weight": torch.ones(8)}
    new = {"w.weight": torch.ones(8) * 1.5}
    agree = [{"w.weight": torch.ones(8) * 1.5} for _ in range(3)]
    disagree = [{"w.weight": torch.ones(8) * (1.5 + 3.0 * i)} for i in range(3)]
    a = fedlws_shrink(prev, new, agree, ["w"])["w.weight"].mean()
    d = fedlws_shrink(prev, new, disagree, ["w"])["w.weight"].mean()
    assert float(d) < float(a), (float(d), float(a))


def test_flayer_gives_more_personalization_to_layers_that_moved_more():
    from pmixfed.recent_baselines import flayer_weights

    before = {"a.weight": torch.ones(10), "b.weight": torch.ones(10)}
    after = {"a.weight": torch.ones(10) * 1.01,   # barely moved
             "b.weight": torch.ones(10) * 2.00}   # moved a lot
    w = flayer_weights(before, after, ["a", "b"])
    assert w["b"] > w["a"], w
    assert all(0.0 <= v <= 1.0 for v in w.values()), w
