#!/usr/bin/env python3
"""Aggregate run JSONs into per-setting tables with mean +/- std over seeds.

Reports the validation-selected test accuracy as the headline (never best-over-rounds),
and prints seed counts so partial results are never mistaken for complete ones.

Usage:
    python3 code/collect.py results/final
    python3 code/collect.py results/final --group heldout
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tag = d.get("config", {}).get("tag") or os.path.basename(f)[:-5]
        out.append((tag, d))
    return out


def strip_seed(tag):
    return re.sub(r"-s\d+$", "", tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--group", default=None, help="only tags containing this substring")
    args = ap.parse_args()

    runs = []
    for d in args.dirs:
        runs += load(os.path.join(d, "*.json"))

    if args.group:
        runs = [(t, d) for t, d in runs if args.group in t]
    if not runs:
        print("no runs found")
        return

    # Group by setting prefix (everything before the method-ish token is unreliable,
    # so we bucket on the tag minus its seed suffix and let the caller read the names).
    buckets = defaultdict(list)
    for tag, d in runs:
        buckets[strip_seed(tag)].append(d)

    # Split into logical sections by the leading token of the tag.
    sections = defaultdict(list)
    for base, ds in buckets.items():
        sec = base.split("-")[0]
        sections[sec].append((base, ds))

    for sec in sorted(sections):
        print(f"\n{'=' * 78}\n{sec}\n{'=' * 78}")
        print(f"{'run':38} {'n':>2} {'val':>13} {'TEST (reported)':>17} {'best':>7}")
        rows = []
        for base, ds in sections[sec]:
            vals = [x["val_acc_at_selected"] for x in ds
                    if x["val_acc_at_selected"] == x["val_acc_at_selected"]]
            tests = [x["test_acc_at_selected"] for x in ds
                     if x["test_acc_at_selected"] == x["test_acc_at_selected"]]
            bests = [x["test_acc_best_over_rounds"] for x in ds]
            if not tests:
                continue
            vm = st.mean(vals) if vals else float("nan")
            tm = st.mean(tests)
            vs = st.stdev(vals) if len(vals) > 1 else 0.0
            ts = st.stdev(tests) if len(tests) > 1 else 0.0
            rows.append((tm, base, len(ds), vm, vs, tm, ts,
                         st.mean(bests) if bests else float("nan")))
        for _, base, n, vm, vs, tm, ts, bm in sorted(rows, reverse=True):
            flag = "" if n >= 2 else "  (single seed)"
            print(f"{base:38} {n:2d} {vm:6.2f}+/-{vs:4.2f} {tm:9.2f}+/-{ts:4.2f} "
                  f"{bm:7.2f}{flag}")

    n_seeds = defaultdict(set)
    for tag, d in runs:
        n_seeds[strip_seed(tag)].add(d["config"]["seed"])
    incomplete = {k: sorted(v) for k, v in n_seeds.items() if len(v) < 3}
    print(f"\n{len(runs)} runs, {len(buckets)} distinct configurations.")
    if incomplete:
        print(f"{len(incomplete)} configurations have <3 seeds "
              f"-- treat their margins as provisional.")


if __name__ == "__main__":
    main()
