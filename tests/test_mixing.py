"""Regression tests for the layer-wise interpolation.

Each test named ``test_dN_*`` pins down one of the defects found in the original
notebook implementation. D1 and D2 would each have been caught by a three-line test,
which is the point of this file.

Run with:  PYTHONPATH=code python3 -m pytest code/tests -q
"""

import math
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from pmixfed.mixing import (
    adaptive_mu,
    interpolate_state,
    lambda_profile,
    layer_blocks,
    mix_global_to_local,
    mix_local_to_global,
)


class TinyNet(nn.Module):
    """Three trainable blocks, plus parameterless modules that must be skipped."""

    def __init__(self, n_out=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4, n_out)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.fc(self.flatten(self.pool(x)))


def constant_net(value, n_out=4):
    net = TinyNet(n_out)
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(value)
    return net


# --------------------------------------------------------------------------- D1

def test_d1_interpolation_actually_changes_parameters():
    """The original code produced unprefixed keys that load_state_dict discarded."""
    g, l = constant_net(1.0), constant_net(0.0)
    before = {k: v.clone() for k, v in l.state_dict().items()}

    mixed = mix_global_to_local(g, l, mu=0.5)
    missing, unexpected = l.load_state_dict(mixed, strict=True), None

    after = l.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed, "no parameter changed: the interpolation was a no-op (D1)"


def test_d1_keys_are_module_qualified_and_load_strictly():
    g, l = constant_net(1.0), constant_net(0.0)
    mixed = mix_global_to_local(g, l, mu=0.5)
    # Every key must be qualified, i.e. contain a dot, and match the model exactly.
    assert all("." in k for k in mixed), "found unprefixed keys (the D1 bug)"
    # strict=True would raise if any key were unexpected or missing.
    l.load_state_dict(mixed, strict=True)


def test_d1_nonmatching_block_names_raise_instead_of_silently_doing_nothing():
    g, l = constant_net(1.0), constant_net(0.0)
    with pytest.raises(RuntimeError, match="no mixed parameters"):
        interpolate_state(g, l, [0.5], blocks=["does_not_exist"])


# --------------------------------------------------------------------------- D2

def test_d2_aggregation_result_is_not_discarded():
    """The original `delta = weight_after` rebound a local and mutated nothing."""
    g, l = constant_net(1.0), constant_net(0.0)
    before = {k: v.clone() for k, v in g.state_dict().items()}
    mixed = mix_local_to_global(g, l, mu=0.5)
    g.load_state_dict(mixed, strict=True)
    after = g.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed, "aggregation interpolation had no effect (D2)"


# ------------------------------------------------------- exactness / limits

def test_lambda_one_recovers_global_exactly():
    """mu large enough that every broadcast lambda saturates at 1 -> pure global."""
    g, l = constant_net(1.0), constant_net(0.0)
    n = len(layer_blocks(l))
    mixed = interpolate_state(g, l, [1.0] * n)
    for k, v in mixed.items():
        if torch.is_floating_point(v):
            assert torch.allclose(v, g.state_dict()[k]), k


def test_lambda_zero_leaves_local_untouched():
    g, l = constant_net(1.0), constant_net(0.0)
    n = len(layer_blocks(l))
    mixed = interpolate_state(g, l, [0.0] * n)
    for k, v in mixed.items():
        if torch.is_floating_point(v):
            assert torch.allclose(v, l.state_dict()[k]), k


def test_half_lambda_is_the_midpoint():
    g, l = constant_net(1.0), constant_net(0.0)
    n = len(layer_blocks(l))
    mixed = interpolate_state(g, l, [0.5] * n)
    for k, v in mixed.items():
        if torch.is_floating_point(v):
            assert torch.allclose(v, torch.full_like(v, 0.5)), k


# ------------------------------------------------------------ lambda profile

def test_profile_matches_closed_form_broadcast():
    n, mu = 5, 0.3
    got = lambda_profile(mu, n, "broadcast")
    want = [min(1.0, mu * (n - i)) for i in range(1, n + 1)]
    assert got == pytest.approx(want)


def test_profile_matches_closed_form_aggregation():
    n, mu = 5, 0.3
    got = lambda_profile(mu, n, "aggregation")
    want = [max(0.0, 1.0 - i * mu) for i in range(1, n + 1)]
    assert got == pytest.approx(want)


@pytest.mark.parametrize("phase", ["broadcast", "aggregation"])
@pytest.mark.parametrize("mu", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_profile_is_monotone_nonincreasing_and_bounded(phase, mu):
    """The stated convention: lambda weights the global model and decays with depth."""
    lam = lambda_profile(mu, 8, phase)
    assert all(0.0 <= x <= 1.0 for x in lam), lam
    assert all(lam[i] >= lam[i + 1] for i in range(len(lam) - 1)), lam


def test_profile_rejects_out_of_range_mu():
    with pytest.raises(ValueError):
        lambda_profile(1.5, 4, "broadcast")


def test_mu_zero_shares_nothing_in_broadcast_except_saturation():
    """mu = 0 -> every lambda is 0 -> the client keeps its own model entirely."""
    assert lambda_profile(0.0, 6, "broadcast") == [0.0] * 6


# --------------------------------------------------------------- adaptive mu

def test_d5_mu_depends_on_round_index():
    """The original get_mu() had no t/T term at all; this pins the dependence."""
    early = adaptive_mu(acc_ratio=2.0, t=1, T=50)
    late = adaptive_mu(acc_ratio=2.0, t=49, T=50)
    assert early != late, "mu must depend on the round (D5)"


def test_mu_at_round_zero_is_one_half():
    """delta = 0 => sigmoid(0) = 0.5 regardless of accuracy."""
    for ratio in [0.1, 1.0, 5.0]:
        assert adaptive_mu(ratio, t=0, T=50) == pytest.approx(0.5)


def test_mu_decreases_when_local_model_outperforms_global():
    """Strong local model => share less of the global model => smaller mu."""
    weak = adaptive_mu(acc_ratio=0.5, t=25, T=50)
    equal = adaptive_mu(acc_ratio=1.0, t=25, T=50)
    strong = adaptive_mu(acc_ratio=3.0, t=25, T=50)
    assert strong < equal < weak


def test_mu_is_half_when_models_are_equally_good():
    assert adaptive_mu(1.0, t=30, T=50) == pytest.approx(0.5)


def test_mu_stays_in_unit_interval():
    for ratio in [0.0, 0.01, 1.0, 10.0, 100.0]:
        for t in [0, 1, 25, 50]:
            mu = adaptive_mu(ratio, t=t, T=50)
            assert 0.0 <= mu <= 1.0, (ratio, t, mu)


def test_cold_start_client_absorbs_more_global_than_expert_client():
    cold = adaptive_mu(acc_ratio=0.2, t=40, T=50)
    expert = adaptive_mu(acc_ratio=2.5, t=40, T=50)
    n = 6
    cold_share = sum(lambda_profile(cold, n, "broadcast"))
    expert_share = sum(lambda_profile(expert, n, "broadcast"))
    assert cold_share > expert_share


# ------------------------------------------------- heterogeneous model depth

def test_heterogeneous_depth_leaves_unmatched_layers_local():
    """A local layer with no global counterpart must stay purely local (lambda=0)."""
    g = constant_net(1.0, n_out=4)
    l = constant_net(0.0, n_out=7)  # fc shape differs -> cannot be mixed
    mixed = mix_global_to_local(g, l, mu=1.0)
    # fc must be untouched despite mu=1, because shapes disagree.
    assert torch.allclose(mixed["fc.weight"], l.state_dict()["fc.weight"])
    # conv layers, which do match, must have been mixed.
    assert not torch.allclose(mixed["conv1.weight"], l.state_dict()["conv1.weight"])


def test_layer_blocks_skips_parameterless_modules():
    blocks = layer_blocks(TinyNet())
    assert blocks == ["conv1", "conv2", "fc"]
    assert "act" not in blocks and "pool" not in blocks


def test_mu_one_recovers_fedper_profile_exactly():
    """The family's headline special case: at mu >= 1 the broadcast profile saturates to
    (1,...,1,0), which is FedPer's update rule -- body wholly global, head wholly private.

    Note the threshold is mu >= 1, not mu >= 1/M: lambda_i = min(1, mu*(M-i)) reaches 1 for
    the last body layer (i = M-1) only when mu >= 1. At mu = 1/M the profile is still
    strictly decreasing, e.g. (0.8, 0.6, 0.4, 0.2, 0) for M = 5.
    """
    for M in (3, 5, 7):
        for mu in (1.0, 1.0 + 1e-9, 5.0):
            lam = lambda_profile(min(mu, 1.0), M, "broadcast")
            assert lam[:-1] == [1.0] * (M - 1), (M, mu, lam)
            assert lam[-1] == 0.0
        # and 1/M is emphatically NOT saturated, for M > 2
        if M > 2:
            lam = lambda_profile(1.0 / M, M, "broadcast")
            assert lam[0] < 1.0, (M, lam)
