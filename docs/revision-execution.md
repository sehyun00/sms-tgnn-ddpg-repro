# Major-Revision Experiment Contract

This document describes the reportable experiment path. The legacy
`main.py`/`run_multiseed.py` path remains a small compatibility smoke workflow
and must not produce manuscript numbers.

## Data contract

- The paper-scale membership source is the archived Wikipedia revision
  `997494787` dated 2020-12-31. The downloaded HTML and its SHA-256 digest are
  stored only in the ignored prepared-data directory.
- The exact primary test symbols are `MMM, ABT, ACN, AFL, APD, ARE, GOOGL, MO,
  AMZN, APA`.
- Five reserve symbols with distinct sectors are selected by a fixed SHA-256
  ordering. After excluding test and reserve symbols, five eligible symbols per
  GICS sector are selected for the 55-stock training universe.
- Training eligibility means at least 98% adjusted-close coverage from the
  2004 warm-up start through 2020 against the SPY trading calendar, so 252-day
  features and the 60-day input window are available when the 2006 fold opens.
  Selection reasons and coverage are recorded in
  `universe_manifest.json`.
- Yahoo Finance prices are adjusted using the Adj-Close/Close ratio. Predictive
  Fama-French factors are lagged by one trading observation; contemporaneous RF
  is used only for ex-post metric calculation. Both series are stored separately. Raw downloads,
  received manuscripts, review letters, and personal correspondence remain
  outside Git.

## Reportable commands

All commands are restart-safe with `--resume`. The paper configuration refuses
to train or backtest without CUDA.

```bash
python scripts/run_revision_experiments.py --config config/paper_revision.yaml --stage prepare
python scripts/run_revision_experiments.py --config config/paper_revision.yaml --stage train --fold primary --model all --seed all --frequency all --graph primary --resume
python scripts/run_revision_experiments.py --config config/paper_revision.yaml --stage backtest --fold primary --model all --seed all --frequency all --graph primary --universe n10 --resume
```

Graph ablation reuses the frequency-specific DDPG checkpoints:

```bash
python scripts/run_revision_experiments.py --config config/paper_revision.yaml --stage train --fold primary --model tgnn,hybrid --seed all --frequency all --graph sector,correlation --resume
python scripts/run_revision_experiments.py --config config/paper_revision.yaml --stage backtest --fold primary --model tgnn,hybrid,hybrid_fixed --seed all --frequency quarterly --graph all --universe n10 --resume
```

Secondary and variable-N checks:

```bash
python scripts/run_revision_experiments.py --config config/paper_revision.yaml --stage train --fold secondary --model all --seed all --frequency all --graph primary --resume
python scripts/run_revision_experiments.py --config config/paper_revision.yaml --stage backtest --fold secondary --model all --seed all --frequency all --graph primary --universe n10 --resume
python scripts/run_revision_experiments.py --config config/paper_revision.yaml --stage backtest --fold primary --model all --seed 42 --frequency quarterly --graph primary --universe n5 --resume
python scripts/run_revision_experiments.py --config config/paper_revision.yaml --stage backtest --fold primary --model all --seed 42 --frequency quarterly --graph primary --universe n15 --resume
python scripts/run_revision_experiments.py --config config/paper_revision.yaml --stage aggregate
```

## Artifact contract

Each run has a deterministic ID and a `run.json` state of `RUNNING`, `SUCCESS`,
or `FAILED`. A successful training run includes a checkpoint, training history,
and checkpoint hash. A successful backtest includes target and branch weights,
per-cost daily returns, decisions, metrics, and model-source hashes.
Paper-scale aggregation excludes runs produced from a dirty Git worktree or a
different code/config/data fingerprint.

`aggregate/` contains:

- `metrics_all.csv` and `summary_by_seed.csv`;
- `primary_table.csv` and `primary_comparisons.csv`;
- `alpha_diagnostics.csv`, XAI artifacts, and the primary comparison figure;
- `reviewer_response_evidence.md`, generated solely from successful ledger entries.

The primary inference family is quarterly, 10 bps, combined graph, N=10. It
uses 20-day paired moving blocks, 10,000 bootstrap/randomization replicates,
and Holm adjustment. Seed intervals and time-series intervals are reported
separately.

## Integrity and interpretation

- Training and validation stop before each test window. Features, covariance,
  and graph edges use information available at the decision close only.
- TGNN targets are future 1/5/21/63-trading-day returns. RL reward is the next
  rebalance interval's net log return at 10 bps.
- DDPG and TD3 collect their complete frequency-specific trajectories and then
  receive the same configured number of gradient steps per epoch; TD3 differs
  only by twin critics, delayed actor updates, and clipped target noise.
- The learned Alpha MLP sees state summaries only. TGNN and DDPG are frozen,
  their source hashes are embedded in the alpha checkpoint, and any mismatch
  aborts evaluation.
- Cost scenarios reuse the identical gross target-weight path. Initial turnover
  is 1.0; later turnover is half the absolute difference from drifted pre-trade
  weights.
- An unfavorable or non-significant result remains in the evidence set. XAI is
  called faithfulness-supported only if top-attention masking exceeds matched
  random masking on both weight and forward-return effects for at least 75% of
  sampled dates and both one-sided binomial tests reject a 50% success rate at
  5%; otherwise it is diagnostic. DSS and variable-N performance
  claims are narrowed as documented.
