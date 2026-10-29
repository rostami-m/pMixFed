#!/usr/bin/env python3
"""Reproduce the experiments in the paper.

Infrastructure-agnostic: this is a plain Python driver with no dependency on any cloud
provider, scheduler, or shared filesystem. It runs configurations sequentially by default,
or `--workers N` in parallel processes on one machine. Every run is idempotent -- a
configuration whose output JSON already exists is skipped -- so the script can be
interrupted and re-invoked, or split across machines by running it with a different
`--shard i/n` on each and collecting the JSONs afterwards.

Examples
--------
    # the table used to develop the method (Table 2), 3 seeds, ~10 min per run on one GPU
    python3 experiments.py --group main

    # everything in the paper, 4 concurrent workers
    python3 experiments.py --group all --workers 4

    # split across two machines, then merge the two results/ directories
    python3 experiments.py --group all --shard 0/2      # machine A
    python3 experiments.py --group all --shard 1/2      # machine B

    # see what would run without running it
    python3 experiments.py --group all --dry-run

Groups
------
    main      Table 2: selection setting, all methods, with and without head fine-tuning
    ablation  Table 3: the mix-factor rule, isolating adaptivity
    heldout   Table 6: eight settings not used to develop the method
    hetero    Table 7: heterogeneous-depth cohort
    mobile    MobileNet on CIFAR-100 at a 300-round budget
    all       all of the above
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)

# Shared training configuration. Identical for every method, which is the point:
# differences between rows of a results table should be attributable to the algorithms.
COMMON = [
    "--local-epochs", "4", "--batch-size", "32",
    "--lr", "0.01", "--optimizer", "sgd", "--momentum", "0.9", "--lr-decay", "0.99",
    "--eval-every", "25",
]

# Methods, and the per-method flags their published definitions require.
#   fedbabu/fedper/fedavg: evaluation-time head fine-tuning is part of the method
#   fedalt: two local passes per round, so its per-pass budget is halved to match compute
#   fedlws: published hyperparameters; the result is sensitive to them
METHODS = {
    "pmixfed":  ["--mu-signal", "progress", "--mu-floor", "0.25", "--mu-ema", "0.7"],
    "pmixfed-fixed": ["--algo", "pmixfed", "--mu-signal", "fixed", "--mu-floor", "0.5"],
    "fedper":   [],
    "fedavg":   [],
    "fedbabu":  [],
    "fedalt":   ["--equal-compute"],
    "fedsim":   [],
    "lgfedavg": [],
    "flayer":   [],
    "fedlws":   [],
    "local":    [],
}

SETTINGS = {
    "c10-N100-C0.1": ["--dataset", "cifar10", "--model", "cnn",
                      "--n-clients", "100", "--frac", "0.1",
                      "--partition", "shard", "--shard-per-user", "5"],
    "c10-N100-C1.0": ["--dataset", "cifar10", "--model", "cnn",
                      "--n-clients", "100", "--frac", "1.0",
                      "--partition", "shard", "--shard-per-user", "5"],
    "c10-N10-C0.1":  ["--dataset", "cifar10", "--model", "cnn",
                      "--n-clients", "10", "--frac", "0.1",
                      "--partition", "shard", "--shard-per-user", "5"],
    "c10-N10-C1.0":  ["--dataset", "cifar10", "--model", "cnn",
                      "--n-clients", "10", "--frac", "1.0",
                      "--partition", "shard", "--shard-per-user", "5"],
    "c100":          ["--dataset", "cifar100", "--model", "cnn",
                      "--n-clients", "100", "--frac", "0.1",
                      "--partition", "shard", "--shard-per-user", "50"],
    "dir0.1":        ["--dataset", "cifar10", "--model", "cnn",
                      "--n-clients", "100", "--frac", "0.1",
                      "--partition", "dirichlet", "--alpha", "0.1"],
    "dir0.5":        ["--dataset", "cifar10", "--model", "cnn",
                      "--n-clients", "100", "--frac", "0.1",
                      "--partition", "dirichlet", "--alpha", "0.5"],
    "dir1.0":        ["--dataset", "cifar10", "--model", "cnn",
                      "--n-clients", "100", "--frac", "0.1",
                      "--partition", "dirichlet", "--alpha", "1.0"],
    "coldstart":     ["--dataset", "cifar10", "--model", "cnn",
                      "--n-clients", "100", "--frac", "0.1",
                      "--partition", "shard", "--shard-per-user", "5",
                      "--n-cold-start", "10"],
    "mobile-c100":   ["--dataset", "cifar100", "--model", "mobile",
                      "--n-clients", "100", "--frac", "0.1",
                      "--partition", "shard", "--shard-per-user", "50"],
}

FINETUNE = ["--finetune-epochs", "5"]


def _job(tag, algo, extra, rounds):
    flags = ["--algo", algo] if not any(f == "--algo" for f in extra) else []
    return {"tag": tag, "argv": flags + extra + COMMON + ["--rounds", str(rounds)]}


def build_jobs(group):
    """Return the list of runs for a group, as (tag, argv) records."""
    jobs = []

    def add(tag, method, setting, seed, rounds=100, finetune=False):
        extra = list(METHODS[method]) + list(SETTINGS[setting]) + ["--seed", str(seed)]
        if finetune:
            extra += FINETUNE
        algo = method.split("-")[0]
        jobs.append(_job(f"{tag}-s{seed}", algo, extra, rounds))

    if group in ("main", "all"):
        # Table 2: every method, with and without head fine-tuning
        for m in METHODS:
            for ft in (False, True):
                for s in SEEDS:
                    add(f"main-{m}{'-ft' if ft else ''}", m, "c10-N100-C0.1", s,
                        finetune=ft)

    if group in ("ablation", "all"):
        # Table 3: isolate the contribution of adaptivity in the mix-factor rule
        variants = {
            "ratio-nofloor": ["--algo", "pmixfed", "--mu-signal", "ratio", "--mu-floor", "0.0"],
            "ratio-floor":   ["--algo", "pmixfed", "--mu-signal", "ratio", "--mu-floor", "0.25"],
            "progress":      ["--algo", "pmixfed", "--mu-signal", "progress", "--mu-floor", "0.25"],
            "progress-ema":  ["--algo", "pmixfed", "--mu-signal", "progress",
                              "--mu-floor", "0.25", "--mu-ema", "0.7"],
            "fixed":         ["--algo", "pmixfed", "--mu-signal", "fixed", "--mu-floor", "0.5"],
        }
        for name, flags in variants.items():
            for s in SEEDS:
                extra = flags + SETTINGS["c10-N100-C0.1"] + ["--seed", str(s)]
                jobs.append(_job(f"ablation-{name}-s{s}", "pmixfed", extra, 100))

    if group in ("heldout", "all"):
        # Table 6: settings that play no part in choosing the method's parameters
        held = ["c10-N100-C1.0", "c10-N10-C0.1", "c10-N10-C1.0", "c100",
                "dir0.1", "dir0.5", "dir1.0", "coldstart"]
        for setting, m in itertools.product(held, METHODS):
            for s in SEEDS:
                add(f"heldout-{setting}-{m}", m, setting, s, finetune=True)

    if group in ("hetero", "all"):
        # Table 7: clients hold CNNs of differing depth
        for m in ("pmixfed", "pmixfed-fixed", "fedper", "local"):
            for s in SEEDS:
                extra = (list(METHODS[m]) + SETTINGS["c10-N100-C0.1"]
                         + ["--hetero-depth", "--seed", str(s)] + FINETUNE)
                jobs.append(_job(f"hetero-{m}-s{s}", m.split("-")[0], extra, 100))

    if group in ("mobile", "all"):
        # MobileNet needs a longer budget: at 100 rounds several methods are still
        # improving steeply, which makes a 100-round comparison uninformative.
        for m in ("pmixfed", "pmixfed-fixed", "fedper", "fedavg", "fedalt", "local"):
            for s in SEEDS:
                add(f"mobile-{m}", m, "mobile-c100", s, rounds=300, finetune=True)

    # De-duplicate while preserving order (groups overlap under --group all).
    seen, out = set(), []
    for j in jobs:
        if j["tag"] not in seen:
            seen.add(j["tag"])
            out.append(j)
    return out


def run_one(job, out_dir, threads, python):
    dest = Path(out_dir) / f"{job['tag']}.json"
    if dest.exists():
        return job["tag"], "skipped"
    cmd = [python, str(HERE / "run.py"), *job["argv"],
           "--threads", str(threads), "--out-dir", str(out_dir), "--tag", job["tag"]]
    log = Path(out_dir) / "logs"
    log.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(HERE))
    # Keep BLAS from oversubscribing when several workers share a machine.
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = str(threads)
    with open(log / f"{job['tag']}.log", "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return job["tag"], "ok" if rc == 0 else f"FAILED(rc={rc})"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="main",
                    choices=["main", "ablation", "heldout", "hetero", "mobile", "all"])
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent runs on this machine")
    ap.add_argument("--threads", type=int, default=4,
                    help="BLAS/torch threads per run")
    ap.add_argument("--shard", default=None, metavar="i/n",
                    help="run only shard i of n, for splitting across machines")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    jobs = build_jobs(args.group)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        jobs = [j for k, j in enumerate(jobs) if k % n == i]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not (Path(args.out_dir) / f"{j['tag']}.json").exists()]
    print(f"group={args.group}: {len(jobs)} configurations, {len(pending)} to run, "
          f"{len(jobs) - len(pending)} already present")

    if args.dry_run:
        for j in jobs:
            print("  ", j["tag"])
        return

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, j, args.out_dir, args.threads, args.python)
                    for j in pending]
            for f in futs:
                tag, status = f.result()
                print(f"  {status:12} {tag}", flush=True)
    else:
        for j in pending:
            tag, status = run_one(j, args.out_dir, args.threads, args.python)
            print(f"  {status:12} {tag}", flush=True)

    print(f"\nDone. Aggregate with:  python3 {HERE.name}/collect.py {args.out_dir}")


if __name__ == "__main__":
    main()
