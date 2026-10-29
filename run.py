#!/usr/bin/env python3
"""Entry point for a single federated run.

Example:
    PYTHONPATH=code python3 code/run.py --algo pmixfed --dataset cifar10 \
        --model cnn --n-clients 100 --frac 0.1 --rounds 100 --threads 4
"""
from __future__ import annotations

import argparse
import dataclasses

from pmixfed.algorithms import ALGORITHMS
from pmixfed.engine import RunConfig, federated_run


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    defaults = RunConfig()
    for f in dataclasses.fields(RunConfig):
        flag = "--" + f.name.replace("_", "-")
        val = getattr(defaults, f.name)
        if f.type is bool or isinstance(val, bool):
            p.add_argument(flag, action="store_true", default=val)
        else:
            p.add_argument(flag, type=type(val) if val is not None else str, default=val)
    p.add_argument("--list-algorithms", action="store_true")
    args = p.parse_args()

    if args.list_algorithms:
        for k, v in sorted(ALGORITHMS.items()):
            print(f"{k:12} {v.name:12} {v.notes}")
        return

    kwargs = {f.name: getattr(args, f.name) for f in dataclasses.fields(RunConfig)}
    cfg = RunConfig(**kwargs)
    res = federated_run(cfg)

    print("\n" + "=" * 66)
    print(f"algorithm                  : {res['algorithm']}")
    print(f"selected round (by val)    : {res['selected_round'] + 1}")
    print(f"val  acc @ selected        : {res['val_acc_at_selected']:.2f}")
    print(f"TEST acc @ selected  <-- reported : {res['test_acc_at_selected']:.2f}")
    print(f"test acc final round       : {res['test_acc_final']:.2f}")
    print(f"test acc best over rounds  : {res['test_acc_best_over_rounds']:.2f}  "
          f"(transparency only, not reported as headline)")
    if res["cold_start_acc"] is not None:
        print(f"cold-start acc ({res['n_cold_start']} clients) : {res['cold_start_acc']:.2f}")
    print(f"wall time                  : {res['wall_time_s']:.1f}s on {res['device']}")
    print("=" * 66)


if __name__ == "__main__":
    main()
