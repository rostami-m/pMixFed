"""Federated data partitioning with a leak-free three-way split.

Two protocols are supported:

``shard``      the shard-based scheme of FedBABU / LG-FedAvg, kept for comparability
               with the numbers previously reported in the literature.
``dirichlet``  label-Dirichlet partitioning with concentration ``alpha``, which the
               reviewers asked for because the shard scheme with S=5 of 10 classes is
               a fairly mild form of heterogeneity.

The important structural change relative to the original implementation is the split
itself. Previously each client had only train and test loaders, the mix factor was
computed from *test* accuracy, and evaluation fine-tuned on the *test* split before
scoring on it. Here every client gets three disjoint index sets:

    train  -  local SGD
    val    -  computing the adaptive mix factor, early stopping, and any
              personalization performed at evaluation time
    test   -  touched exactly once, to produce the final reported number

``val`` is carved out of the client's share of the training set, so the test set is
never used for anything but final measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

__all__ = ["ClientSplit", "build_federated_data", "NORMALIZE"]


NORMALIZE = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "cifar100": ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
}
NUM_CLASSES = {"mnist": 10, "cifar10": 10, "cifar100": 100}


@dataclass
class ClientSplit:
    """Index sets for one client. All three are disjoint by construction."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self):
        # train and val index the *training* dataset; test indexes the separate test
        # dataset, so only train/val can meaningfully overlap. Checking train against
        # test would compare indices into different arrays and false-positive.
        if np.intersect1d(self.train, self.val).size:
            raise ValueError(
                "train/val index overlap in client split: the validation split must be "
                "disjoint from the training data, otherwise the mix factor and early "
                "stopping are computed on data the model has trained on"
            )


class IndexedSubset(Dataset):
    def __init__(self, base: Dataset, idxs):
        self.base = base
        self.idxs = np.asarray(idxs, dtype=np.int64)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.base[int(self.idxs[i])]


def _transforms(name: str, train: bool):
    mean, std = NORMALIZE[name]
    if name == "mnist":
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def _load_raw(name: str, root: str):
    name = name.lower()
    ctor = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[name]
    tr = ctor(root, train=True, download=True, transform=_transforms(name, True))
    # A second view of the training set with *eval* transforms, so validation is not
    # measured through random crops and flips.
    tr_eval = ctor(root, train=True, download=True, transform=_transforms(name, False))
    te = ctor(root, train=False, download=True, transform=_transforms(name, False))
    return tr, tr_eval, te


def _targets(ds) -> np.ndarray:
    t = ds.targets
    if isinstance(t, torch.Tensor):
        return t.numpy()
    return np.asarray(t)


def _shard_partition(targets, n_clients, shard_per_user, rng):
    """Shard scheme: each client draws `shard_per_user` shards, each of one class."""
    n_classes = len(np.unique(targets))
    by_class = {c: rng.permutation(np.where(targets == c)[0]) for c in range(n_classes)}

    shards_per_class = int(shard_per_user * n_clients / n_classes)
    if shards_per_class == 0:
        raise ValueError("shard_per_user * n_clients is too small for this dataset")

    pool = {}
    for c, idx in by_class.items():
        cut = len(idx) - (len(idx) % shards_per_class) if len(idx) % shards_per_class else len(idx)
        chunks = np.split(idx[:cut], shards_per_class)
        # Scatter the remainder so no data is silently dropped.
        for j, extra in enumerate(idx[cut:]):
            chunks[j % len(chunks)] = np.append(chunks[j % len(chunks)], extra)
        pool[c] = list(chunks)

    assign = np.array(list(range(n_classes)) * shards_per_class)
    rng.shuffle(assign)
    assign = assign.reshape((n_clients, -1))

    out = []
    for i in range(n_clients):
        picked = []
        for c in assign[i]:
            picked.append(pool[c].pop(rng.integers(len(pool[c]))))
        out.append(np.concatenate(picked))
    return out


def _dirichlet_partition(targets, n_clients, alpha, rng, min_size=10):
    """Label-Dirichlet partitioning. Smaller alpha => more skew."""
    n_classes = len(np.unique(targets))
    for _ in range(100):
        parts = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx = rng.permutation(np.where(targets == c)[0])
            p = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for j, chunk in enumerate(np.split(idx, cuts)):
                parts[j].extend(chunk.tolist())
        sizes = [len(p) for p in parts]
        if min(sizes) >= min_size:
            return [np.array(sorted(p)) for p in parts]
    raise RuntimeError(
        f"could not build a Dirichlet partition with >= {min_size} samples per client "
        f"after 100 attempts (alpha={alpha}, n_clients={n_clients})"
    )


def build_federated_data(
    dataset: str,
    n_clients: int,
    *,
    partition: str = "shard",
    shard_per_user: int = 5,
    alpha: float = 0.5,
    val_fraction: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    root: str = "data",
    num_workers: int = 0,
) -> Tuple[Dict[int, Dict[str, DataLoader]], int]:
    """Return per-client {train,val,test} loaders and the number of classes.

    The test partition is built with the *same* label assignment as the training
    partition, so a client's test distribution matches its training distribution --
    this is what makes "personalized accuracy" meaningful.
    """
    dataset = dataset.lower()
    if partition not in ("shard", "dirichlet"):
        raise ValueError("partition must be 'shard' or 'dirichlet'")

    rng = np.random.default_rng(seed)
    tr, tr_eval, te = _load_raw(dataset, root)
    y_tr, y_te = _targets(tr), _targets(te)

    if partition == "shard":
        train_parts = _shard_partition(y_tr, n_clients, shard_per_user, rng)
        # Mirror each client's observed class set onto the test pool.
        test_pool = {c: list(rng.permutation(np.where(y_te == c)[0])) for c in np.unique(y_te)}
        test_parts = []
        for part in train_parts:
            classes, counts = np.unique(y_tr[part], return_counts=True)
            want = {c: max(1, int(round(len(test_pool[c]) / n_clients))) for c in classes}
            picked = []
            for c in classes:
                take = min(want[c], len(test_pool[c]))
                picked.extend(test_pool[c][:take])
                test_pool[c] = test_pool[c][take:]
            test_parts.append(np.array(picked, dtype=np.int64))
    else:
        train_parts = _dirichlet_partition(y_tr, n_clients, alpha, rng)
        # Build the test split by matching each client's empirical label histogram.
        test_pool = {c: list(rng.permutation(np.where(y_te == c)[0])) for c in np.unique(y_te)}
        test_parts = []
        for part in train_parts:
            hist = np.bincount(y_tr[part], minlength=len(test_pool))
            frac = hist / max(1, hist.sum())
            picked = []
            for c, f in enumerate(frac):
                take = min(int(round(f * len(y_te) / n_clients)), len(test_pool[c]))
                picked.extend(test_pool[c][:take])
                test_pool[c] = test_pool[c][take:]
            if not picked:  # degenerate client: give it one sample so eval is defined
                c = int(np.argmax(hist))
                if test_pool[c]:
                    picked = [test_pool[c].pop()]
            test_parts.append(np.array(picked, dtype=np.int64))

    loaders: Dict[int, Dict[str, DataLoader]] = {}
    for k in range(n_clients):
        part = rng.permutation(train_parts[k])
        n_val = max(1, int(round(val_fraction * len(part))))
        val_idx, train_idx = part[:n_val], part[n_val:]
        split = ClientSplit(train=train_idx, val=val_idx, test=test_parts[k])

        loaders[k] = {
            "train": DataLoader(
                IndexedSubset(tr, split.train),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
            ),
            # Validation uses the eval-transform view of the training data.
            "val": DataLoader(
                IndexedSubset(tr_eval, split.val),
                batch_size=256,
                shuffle=False,
                num_workers=num_workers,
            ),
            "test": DataLoader(
                IndexedSubset(te, split.test),
                batch_size=256,
                shuffle=False,
                num_workers=num_workers,
            ),
        }
    return loaders, NUM_CLASSES[dataset]
