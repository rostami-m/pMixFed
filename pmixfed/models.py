"""Model architectures, matching those used by FedBABU so results are comparable.

Each model exposes ``body_blocks()`` / ``head_blocks()`` so that partial-personalization
baselines (FedPer, FedBABU, FedAlt, FedSim) and pMixFed can all be expressed in terms
of the same layer-block decomposition used by ``pmixfed.mixing.layer_blocks``.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CNNCifar", "CNNMnist", "MobileNetCifar", "build_model"]


class _Partitionable(nn.Module):
    """Mixin giving a body/head split defined over top-level trainable blocks."""

    head_names: List[str] = []

    def head_blocks(self) -> List[str]:
        return list(self.head_names)

    def body_blocks(self) -> List[str]:
        from .mixing import layer_blocks

        head = set(self.head_names)
        return [b for b in layer_blocks(self) if b not in head]


class CNNMnist(_Partitionable):
    """2 conv + 2 fc, as described for MNIST in the paper."""

    head_names = ["fc2"]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNNCifar(_Partitionable):
    """4-conv CNN + 1 fc, matching the paper's CIFAR description."""

    head_names = ["fc"]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 2 * 2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.adaptive_avg_pool2d(x, 2).flatten(1)
        return self.fc(x)


class _DWBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv2 = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)))


class MobileNetCifar(_Partitionable):
    """MobileNet-style network for CIFAR, as in FedBABU."""

    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    head_names = ["linear"]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        layers, cin = [], 32
        for x in self.cfg:
            cout, stride = (x, 1) if isinstance(x, int) else x
            layers.append(_DWBlock(cin, cout, stride))
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.linear(x)


class CNNCifarShallow(_Partitionable):
    """A 2-conv variant of CNNCifar, for heterogeneous-depth cohorts.

    Deliberately shares block *names* with CNNCifar so that matching layers interoperate
    and non-matching ones are skipped by the lambda_i = 0 rule.
    """

    head_names = ["fc"]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 2 * 2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.adaptive_avg_pool2d(x, 2).flatten(1)
        return self.fc(x)


class CNNCifarDeep(_Partitionable):
    """A 6-conv variant, the deep end of the heterogeneous cohort."""

    head_names = ["fc"]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 2 * 2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.adaptive_avg_pool2d(x, 2).flatten(1)
        return self.fc(x)


HETERO_FAMILY = ["cnn_shallow", "cnn", "cnn_deep"]


def build_model(name: str, dataset: str, num_classes: int) -> nn.Module:
    if name == "cnn_shallow":
        return CNNCifarShallow(num_classes)
    if name == "cnn_deep":
        return CNNCifarDeep(num_classes)
    name, dataset = name.lower(), dataset.lower()
    if name == "cnn":
        return CNNMnist(num_classes) if dataset == "mnist" else CNNCifar(num_classes)
    if name in ("mobile", "mobilenet"):
        return MobileNetCifar(num_classes)
    raise ValueError(f"unknown model {name!r}")
