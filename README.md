# pMixFed: layer-wise interpolation for partial personalized federated learning

Reference implementation and experiment driver for the paper. Everything here runs with
PyTorch on a single machine, CPU or GPU; there is no dependency on a cloud provider,
cluster scheduler, or shared filesystem.

## Contents

| path | purpose |
|---|---|
| `pmixfed/mixing.py` | the sharing profile $\lambda_i$, the mix factor $\mu$, parameter-space interpolation |
| `pmixfed/algorithms.py` | method registry: which blocks each algorithm shares, aggregates, and trains |
| `pmixfed/engine.py` | one federated training loop used by every method |
| `pmixfed/data.py` | shard and label-Dirichlet partitioning, three-way per-client split |
| `pmixfed/models.py` | CNN backbones (MNIST, CIFAR), MobileNet, and a mixed-depth family |
| `pmixfed/recent_baselines.py` | FLAYER and FedLWS |
| `run.py` | run one configuration |
| `experiments.py` | reproduce the paper's tables |
| `collect.py` | aggregate result JSONs into tables |
| `tests/` | 81 tests, including invariants that guard the protocol |

## Install

```bash
pip install torch torchvision numpy pytest
```

Datasets download automatically on first use into `data/` relative to the working
directory. No other setup is required.

## Verify the installation

```bash
PYTHONPATH=. python3 -m pytest tests -q          # ~3 min on CPU
```

The suite includes correctness tests for the interpolation and, deliberately, a set of
adversarial tests that fail if any method receives an advantage the others do not: unequal
local compute, unequal fine-tuning, a training/evaluation mismatch, or any path that lets
test data influence training. If you modify the engine, these are the tests to watch.

## Run a single configuration

```bash
# our method on the setting of Table 2
PYTHONPATH=. python3 run.py --algo pmixfed \
    --mu-signal progress --mu-floor 0.25 --mu-ema 0.7 \
    --dataset cifar10 --model cnn --n-clients 100 --frac 0.1 \
    --partition shard --shard-per-user 5 \
    --rounds 100 --local-epochs 4 --lr 0.01 --optimizer sgd --momentum 0.9 \
    --lr-decay 0.99 --seed 42 --finetune-epochs 5

PYTHONPATH=. python3 run.py --list-algorithms      # all available methods
```

Each run writes one JSON containing the full configuration, the per-round history, and the
reported metric, so a results directory is self-describing.

### The mix factor

`--mu-signal` selects how the sharing profile's slope is set each round:

| value | behaviour |
|---|---|
| `fixed` | constant $\mu$, taken from `--mu-floor`; no adaptation |
| `progress` | $\mu$ from the incoming global model's improvement on the client's validation split |
| `ratio` | $\mu$ from the ratio of local to global validation accuracy |

`--mu-floor` lower-bounds $\mu$. At $\mu \ge 1$ the profile saturates to
$(1,\dots,1,0)$ and the method reduces exactly to FedPer, which is a useful sanity check:
`--mu-signal fixed --mu-floor 1.0` shares the whole body and keeps the head private, matching
`--algo fedper`. The configurations reported in the paper use $\mu = 0.5$, in the region
where layers are only partially shared.

## Reproduce the paper

```bash
PYTHONPATH=. python3 experiments.py --group main       # Table 2
PYTHONPATH=. python3 experiments.py --group ablation   # Table 3
PYTHONPATH=. python3 experiments.py --group heldout    # Table 6
PYTHONPATH=. python3 experiments.py --group hetero     # Table 7
PYTHONPATH=. python3 experiments.py --group mobile     # MobileNet, 300 rounds
PYTHONPATH=. python3 experiments.py --group all        # 375 configurations

PYTHONPATH=. python3 collect.py results
```

Runs are idempotent: a configuration whose JSON already exists is skipped, so the driver
can be interrupted and re-invoked freely.

**Scaling.** `--workers N` runs N configurations concurrently on one machine. To use
several machines without any shared infrastructure, give each a different shard and merge
the output directories afterwards:

```bash
python3 experiments.py --group all --shard 0/3        # machine A
python3 experiments.py --group all --shard 1/3        # machine B
python3 experiments.py --group all --shard 2/3        # machine C
```

`--dry-run` lists what would run. On one modern GPU, a CNN configuration takes roughly ten
minutes; the MobileNet group is substantially slower.

## The evaluation protocol, and why it is enforced in code

The paper's central empirical claim is that protocol choices dominate method rankings on
this benchmark, so the protocol is enforced by the implementation rather than left to
convention:

- **Three-way per-client split.** `data.py` gives each client disjoint train / validation /
  test index sets, with validation carved out of that client's own training shard.
  `ClientSplit.__post_init__` raises if train and validation overlap.
- **Test data is used once per reported number.** The test loader is reachable only from the
  no-grad `evaluate()`; it is never passed to `_local_train`. `--finetune-on` accepts only
  `train` or `val`.
- **Model selection on validation.** The reported figure is `test_acc_at_selected`: test
  accuracy at the round with the best client-averaged validation accuracy.
  `test_acc_best_over_rounds` is also recorded and is by construction an upper bound.
- **Matched compute.** FedAlt makes two local passes per round, which at equal
  `--local-epochs` is twice the gradient steps of every other method; `--equal-compute`
  halves its per-pass budget.
- **Matched fine-tuning.** `--finetune-epochs` applies evaluation-time head fine-tuning on
  the client's own training data. Several baselines are defined to include it, so the paper
  reports every method both with and without it.
- **One optimizer for all methods.** SGD with momentum throughout; Adam is available but
  not used in reported results.

## Notes on the baselines

FLAYER and FedLWS are our own implementations, as neither reference repository carries a
license. FedLWS is sensitive to two hyperparameters: with $\beta = 0.03$ and
$\tau_{\max} = 0.2$ as published it trains normally, while a $\tau$ an order of magnitude
larger shrinks the aggregated weights multiplicatively every round and degrades the method
severely. The defaults here are the published values.

## Reproducibility

Runs are deterministic given `--seed` on CPU and MPS. On CUDA, cuDNN kernel selection
introduces a run-to-run difference of roughly 0.05 accuracy points even with
`deterministic=True` — about an order of magnitude below the seed-to-seed spread, and well
below any margin discussed in the paper.
