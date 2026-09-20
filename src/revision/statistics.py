from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .ledger import build_ledger, config_hash
from .utils import git_commit, git_worktree_fingerprint


def annualized_sharpe(returns: np.ndarray, risk_free: np.ndarray) -> float:
    excess = np.asarray(returns, dtype=float) - np.asarray(risk_free, dtype=float)
    deviation = excess.std(ddof=1)
    return float(excess.mean() / deviation * np.sqrt(252.0)) if deviation > 0 else float("nan")


def _moving_block_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = int(np.ceil(length / block_length))
    starts = rng.integers(0, length, size=blocks)
    indices = np.concatenate([(start + np.arange(block_length)) % length for start in starts])
    return indices[:length]


def paired_moving_block_inference(
    first: np.ndarray,
    second: np.ndarray,
    risk_free: np.ndarray,
    iterations: int,
    block_length: int,
    confidence_level: float,
    seed: int,
) -> Dict[str, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    risk_free = np.asarray(risk_free, dtype=float)
    if not (len(first) == len(second) == len(risk_free)) or len(first) < 2:
        raise ValueError("Paired block inference requires equal non-trivial series")
    observed = annualized_sharpe(first, risk_free) - annualized_sharpe(second, risk_free)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=float)
    randomized = np.empty(iterations, dtype=float)
    block_ids = np.arange(len(first)) // max(1, block_length)
    unique_blocks = np.unique(block_ids)
    for iteration in range(iterations):
        indices = _moving_block_indices(len(first), block_length, rng)
        bootstrap[iteration] = annualized_sharpe(first[indices], risk_free[indices]) - annualized_sharpe(
            second[indices], risk_free[indices]
        )
        swap_flags = rng.integers(0, 2, size=len(unique_blocks)).astype(bool)
        swap = swap_flags[block_ids]
        permuted_first = np.where(swap, second, first)
        permuted_second = np.where(swap, first, second)
        randomized[iteration] = annualized_sharpe(permuted_first, risk_free) - annualized_sharpe(
            permuted_second, risk_free
        )
    alpha = 1.0 - confidence_level
    finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
    finite_randomized = randomized[np.isfinite(randomized)]
    if len(finite_bootstrap) == 0 or len(finite_randomized) == 0:
        raise RuntimeError("Bootstrap inference produced no finite replicates")
    p_value = float((1 + np.sum(np.abs(finite_randomized) >= abs(observed))) / (1 + len(finite_randomized)))
    return {
        "sharpe_difference": float(observed),
        "time_ci_low": float(np.quantile(finite_bootstrap, alpha / 2)),
        "time_ci_high": float(np.quantile(finite_bootstrap, 1.0 - alpha / 2)),
        "p_value": p_value,
    }


def bootstrap_seed_ci(
    differences: Iterable[float], iterations: int, confidence_level: float, seed: int
) -> tuple[float, float]:
    values = np.asarray(list(differences), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    alpha = 1.0 - confidence_level
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1.0 - alpha / 2))


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, position in enumerate(order):
        candidate = min(1.0, (count - rank) * values[position])
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted.tolist()


def _successful_backtest_runs(
    ledger: pd.DataFrame, config: Dict[str, Any], data_hash: str
) -> list[tuple[Dict[str, Any], Path]]:
    runs: list[tuple[Dict[str, Any], Path]] = []
    expected_config = config_hash(config)
    expected_commit = git_commit(config["_meta"]["project_root"])
    expected_fingerprint = git_worktree_fingerprint(config["_meta"]["project_root"])
    eligible = ledger[
        (ledger["stage"] == "backtest")
        & (ledger["status"] == "SUCCESS")
        & (ledger["config_hash"] == expected_config)
        & (ledger["data_hash"] == data_hash)
    ].drop_duplicates("run_id")
    for row in eligible.itertuples(index=False):
        manifest_path = Path(row.run_path) / "run.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            (expected_commit == "unknown" or payload.get("code_commit") == expected_commit)
            and (expected_fingerprint == "unknown" or payload.get("code_fingerprint") == expected_fingerprint)
            and (config["data"].get("provider") != "yfinance" or not payload.get("code_dirty", True))
        ):
            runs.append((payload, manifest_path.parent))
    return runs


def _metrics_rows(runs: list[tuple[Dict[str, Any], Path]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload, directory in runs:
        metrics_path = directory / "metrics.csv"
        if not metrics_path.exists():
            raise RuntimeError(f"Successful run lacks metrics.csv: {directory}")
        metrics = pd.read_csv(metrics_path)
        for row in metrics.to_dict("records"):
            rows.append(
                {
                    "run_id": payload["run_id"],
                    "fold": payload["fold"],
                    "universe": payload["universe"],
                    "model": payload["model"],
                    "seed": int(payload["seed"]),
                    "frequency": payload["frequency"],
                    "graph": payload["graph"],
                    **row,
                    "run_path": str(directory),
                }
            )
    return pd.DataFrame(rows)


def _primary_filter(config: Dict[str, Any], metrics: pd.DataFrame) -> pd.DataFrame:
    primary_frequency = config["backtest"]["primary_frequency"]
    primary_graph = config["backtest"]["primary_graph"]
    primary_cost = float(config["backtest"]["primary_cost_bps"])
    base = metrics[
        (metrics["fold"] == "primary")
        & (metrics["universe"] == "n10")
        & (metrics["frequency"] == primary_frequency)
        & np.isclose(metrics["cost_bps"], primary_cost)
    ].copy()
    graph_models = base["model"].isin(["tgnn", "hybrid", "hybrid_fixed"])
    return base[(~graph_models) | (base["graph"] == primary_graph)]


def _daily_for_run(directory: Path, cost_bps: float) -> pd.Series:
    path = directory / f"daily_cost_{float(cost_bps):g}bps.csv.gz"
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return frame["net_return"].astype(float)


def _primary_inference(
    config: Dict[str, Any],
    primary: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    primary_model = config["statistics"]["primary_model"]
    primary_rows = primary[primary["model"] == primary_model]
    if primary_rows.empty:
        return pd.DataFrame()
    cost = float(config["backtest"]["primary_cost_bps"])
    rf = factors.set_index("Date")["RF"].astype(float)
    comparisons: list[dict[str, Any]] = []
    deterministic = {"equal_weight", "min_variance", "risk_parity"}

    for comparison_index, comparator in enumerate(config["statistics"]["primary_comparators"]):
        comparator_rows = primary[primary["model"] == comparator]
        pairs: list[tuple[int, pd.Series, pd.Series, float]] = []
        for _, first_row in primary_rows.iterrows():
            if comparator in deterministic:
                if comparator_rows.empty:
                    continue
                second_row = comparator_rows.iloc[0]
            else:
                matches = comparator_rows[comparator_rows["seed"] == first_row["seed"]]
                if matches.empty:
                    continue
                second_row = matches.iloc[0]
            first_daily = _daily_for_run(Path(first_row["run_path"]), cost)
            second_daily = _daily_for_run(Path(second_row["run_path"]), cost)
            common = first_daily.index.intersection(second_daily.index).intersection(rf.index)
            if len(common) < 2:
                continue
            seed_difference = float(first_row["sharpe"] - second_row["sharpe"])
            pairs.append((int(first_row["seed"]), first_daily.loc[common], second_daily.loc[common], seed_difference))
        if not pairs:
            continue
        common_dates = pairs[0][1].index
        for _, first_daily, second_daily, _ in pairs[1:]:
            common_dates = common_dates.intersection(first_daily.index).intersection(second_daily.index)
        first_mean = np.vstack([pair[1].loc[common_dates].to_numpy() for pair in pairs]).mean(axis=0)
        second_mean = np.vstack([pair[2].loc[common_dates].to_numpy() for pair in pairs]).mean(axis=0)
        inference = paired_moving_block_inference(
            first_mean,
            second_mean,
            rf.loc[common_dates].to_numpy(),
            iterations=int(config["statistics"]["bootstrap_iterations"]),
            block_length=int(config["statistics"]["block_length"]),
            confidence_level=float(config["statistics"]["confidence_level"]),
            seed=20260907 + comparison_index,
        )
        seed_low, seed_high = bootstrap_seed_ci(
            [pair[3] for pair in pairs],
            int(config["statistics"]["bootstrap_iterations"]),
            float(config["statistics"]["confidence_level"]),
            20261000 + comparison_index,
        )
        comparisons.append(
            {
                "model": primary_model,
                "comparator": comparator,
                "paired_seeds": len(pairs),
                "seed_difference_mean": float(np.mean([pair[3] for pair in pairs])),
                "seed_ci_low": seed_low,
                "seed_ci_high": seed_high,
                **inference,
            }
        )
    output = pd.DataFrame(comparisons)
    if not output.empty:
        output["p_value_holm"] = holm_adjust(output["p_value"])
    return output


def _aggregate_alpha(runs: list[tuple[Dict[str, Any], Path]], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload, directory in runs:
        if payload["model"] != "hybrid":
            continue
        weights_path = directory / "weights.csv.gz"
        branch_path = directory / "branch_daily_returns.csv"
        weights = pd.read_csv(weights_path)
        alpha = weights[["decision_date", "alpha"]].drop_duplicates()
        branch_correlation = float("nan")
        if branch_path.exists():
            branch = pd.read_csv(branch_path)
            branch_correlation = float(branch["tgnn_gross_return"].corr(branch["ddpg_gross_return"]))
        rows.append(
            {
                "run_id": payload["run_id"],
                "fold": payload["fold"],
                "seed": payload["seed"],
                "frequency": payload["frequency"],
                "graph": payload["graph"],
                "alpha_mean": float(alpha["alpha"].mean()),
                "alpha_std": float(alpha["alpha"].std(ddof=1)),
                "alpha_min": float(alpha["alpha"].min()),
                "alpha_max": float(alpha["alpha"].max()),
                "branch_return_correlation": branch_correlation,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "alpha_diagnostics.csv", index=False)
    return frame


def _aggregate_xai(
    config: Dict[str, Any], runs: list[tuple[Dict[str, Any], Path]], output_dir: Path
) -> Dict[str, Any]:
    frames = []
    for payload, directory in runs:
        path = directory / "xai_faithfulness.csv"
        if path.exists():
            one = pd.read_csv(path)
            one["run_id"] = payload["run_id"]
            one["seed"] = payload["seed"]
            frames.append(one)
    if not frames:
        summary = {"status": "not-run", "interpretation": "diagnostic-only"}
    else:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output_dir / "xai_faithfulness_all.csv", index=False)
        weight_ratio = float(combined["top_exceeds_random_weight"].mean())
        return_ratio = float(combined["top_exceeds_random_return"].mean())
        threshold = float(config["xai"]["faithfulness_fraction_threshold"])
        weight_successes = int(combined["top_exceeds_random_weight"].sum())
        return_successes = int(combined["top_exceeds_random_return"].sum())
        weight_p = float(binomtest(weight_successes, len(combined), 0.5, alternative="greater").pvalue)
        return_p = float(binomtest(return_successes, len(combined), 0.5, alternative="greater").pvalue)
        faithful = (
            weight_ratio >= threshold
            and return_ratio >= threshold
            and weight_p < 0.05
            and return_p < 0.05
        )
        summary = {
            "status": "complete",
            "dates": len(combined),
            "top_exceeds_random_weight_fraction": weight_ratio,
            "top_exceeds_random_return_fraction": return_ratio,
            "top_exceeds_random_weight_p_value": weight_p,
            "top_exceeds_random_return_p_value": return_p,
            "faithfulness_fraction_threshold": threshold,
            "interpretation": "faithfulness-supported" if faithful else "diagnostic-only",
        }
    (output_dir / "xai_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_figures(
    config: Dict[str, Any], primary: pd.DataFrame, all_metrics: pd.DataFrame, output_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    if primary.empty:
        return
    main = primary.groupby("model", as_index=False)[["cagr", "sharpe", "mdd"]].mean(numeric_only=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(main["model"], main["cagr"] * 100.0)
    axes[0].set_ylabel("CAGR (%)")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(main["model"], main["sharpe"])
    axes[1].set_ylabel("Sharpe ratio")
    axes[1].tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_dir / "primary_model_comparison.png", dpi=200)
    plt.close(figure)

    sensitivity = all_metrics[
        (all_metrics["fold"] == "primary")
        & (all_metrics["universe"] == "n10")
        & (all_metrics["frequency"] == config["backtest"]["primary_frequency"])
    ].copy()
    graph_models = sensitivity["model"].isin(["tgnn", "hybrid", "hybrid_fixed"])
    sensitivity = sensitivity[
        (~graph_models) | (sensitivity["graph"] == config["backtest"]["primary_graph"])
    ]
    sensitivity = sensitivity.groupby(["model", "cost_bps"], as_index=False)["sharpe"].mean()
    figure, axis = plt.subplots(figsize=(8, 5))
    for model, group in sensitivity.groupby("model"):
        axis.plot(group["cost_bps"], group["sharpe"], marker="o", label=model)
    axis.set_xlabel("Transaction cost (bps)")
    axis.set_ylabel("Mean Sharpe ratio")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "transaction_cost_sensitivity.png", dpi=200)
    plt.close(figure)

    graph = all_metrics[
        (all_metrics["fold"] == "primary")
        & (all_metrics["universe"] == "n10")
        & (all_metrics["frequency"] == config["backtest"]["primary_frequency"])
        & np.isclose(all_metrics["cost_bps"], float(config["backtest"]["primary_cost_bps"]))
        & all_metrics["model"].isin(["tgnn", "hybrid"])
    ]
    if not graph.empty:
        graph_summary = graph.groupby(["model", "graph"], as_index=False)["sharpe"].mean()
        pivot = graph_summary.pivot(index="graph", columns="model", values="sharpe")
        figure, axis = plt.subplots(figsize=(7, 4.5))
        pivot.plot(kind="bar", ax=axis)
        axis.set_ylabel("Mean Sharpe ratio")
        axis.tick_params(axis="x", rotation=0)
        figure.tight_layout()
        figure.savefig(output_dir / "graph_ablation.png", dpi=200)
        plt.close(figure)


def _write_reviewer_evidence(
    output_dir: Path,
    primary: pd.DataFrame,
    inference: pd.DataFrame,
    alpha: pd.DataFrame,
    xai: Dict[str, Any],
    readiness: Dict[str, Any],
) -> None:
    complete_models = sorted(primary["model"].unique()) if not primary.empty else []
    lines = [
        "# Reviewer-response evidence map",
        "",
        "> Generated only from successful run manifests. Replace manuscript numbers only after the evidence freeze.",
        "",
        f"Evidence readiness: **{readiness['status']}**.",
        "",
        "| Reviewer issue | Disposition | Evidence |",
        "|---|---|---|",
        "| Transaction costs | Experiment | `metrics_all.csv`, four post-hoc cost scenarios from identical gross weights |",
        "| Weak baselines | Experiment | `primary_table.csv` with Equal Weight, Minimum Variance, Risk Parity, and TD3 |",
        "| DDPG underperformance | Experiment/defense | run-level training histories plus paired comparison table; report negative outcomes unchanged |",
        "| Statistical significance | Experiment | `primary_comparisons.csv`, moving-block confidence intervals and Holm-adjusted p-values |",
        "| Static graph contradiction | Experiment | sector/correlation/combined graph runs identified in the ledger |",
        "| Alpha training unclear | Experiment/method correction | frozen branch hashes, `alpha_diagnostics.csv`, and fixed-alpha ablation |",
        f"| XAI faithfulness | {'Experiment' if xai.get('status') == 'complete' else 'Claim narrowing'} | `xai_summary.json`; interpretation is `{xai.get('interpretation')}` |",
        "| Variable N | Claim narrowing | N=5/10/15 functional runs only; remove performance-generalization wording |",
        "| DSS validation | Claim narrowing | describe as proposed deployment architecture, not an implemented real-time DSS |",
        "| Horizon claims | Claim narrowing | characterize frequency results as sensitivity evidence, not universal superiority |",
        "",
        f"Completed primary models: {', '.join(complete_models) if complete_models else 'none'}.",
        f"Primary comparisons available: {len(inference)}.",
        f"Alpha diagnostic runs available: {len(alpha)}.",
        "",
        "## Manuscript-edit boundary",
        "",
        "This file supplies evidence and wording direction; it does not edit the full manuscript. Every retained quantitative claim must cite a run ID from `ledger.csv`.",
    ]
    (output_dir / "reviewer_response_evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evidence_readiness(
    config: Dict[str, Any], metrics: pd.DataFrame, primary: pd.DataFrame, inference: pd.DataFrame, xai: Dict[str, Any]
) -> Dict[str, Any]:
    neural = {"tgnn", "ddpg", "td3", "hybrid", "hybrid_fixed"}
    baselines = {"equal_weight", "min_variance", "risk_parity"}
    required_models = neural | baselines
    expected_primary_seeds = len(config["folds"]["primary"]["seeds"])
    seed_counts = primary.groupby("model")["seed"].nunique().to_dict() if not primary.empty else {}
    model_seed_gate = {
        model: int(seed_counts.get(model, 0)) >= (expected_primary_seeds if model in neural else 1)
        for model in sorted(required_models)
    }
    expected_costs = {float(value) for value in config["backtest"]["cost_bps"]}
    cost_gate: dict[str, bool] = {}
    for model in sorted(required_models):
        model_rows = metrics[
            (metrics["fold"] == "primary")
            & (metrics["universe"] == "n10")
            & (metrics["model"] == model)
        ]
        if model in {"tgnn", "hybrid", "hybrid_fixed"}:
            model_rows = model_rows[model_rows["graph"] == config["backtest"]["primary_graph"]]
        cost_gate[model] = all(
            expected_costs.issubset(
                set(model_rows.loc[model_rows["frequency"] == frequency, "cost_bps"].astype(float))
            )
            for frequency in config["backtest"]["frequencies"]
        )
    graph_runs = metrics[
        (metrics["fold"] == "primary")
        & (metrics["universe"] == "n10")
        & (metrics["frequency"] == config["backtest"]["primary_frequency"])
        & np.isclose(metrics["cost_bps"], float(config["backtest"]["primary_cost_bps"]))
    ]
    graph_gate = {
        model: set(config["backtest"]["graph_modes"]).issubset(
            set(graph_runs.loc[graph_runs["model"] == model, "graph"])
        )
        for model in ("tgnn", "hybrid", "hybrid_fixed")
    }
    expected_secondary_seeds = len(config["folds"]["secondary"]["seeds"])
    secondary = metrics[
        (metrics["fold"] == "secondary")
        & (metrics["universe"] == "n10")
        & np.isclose(metrics["cost_bps"], float(config["backtest"]["primary_cost_bps"]))
    ]
    secondary_counts = secondary.groupby("model")["seed"].nunique().to_dict() if not secondary.empty else {}
    secondary_gate = {
        model: int(secondary_counts.get(model, 0)) >= (expected_secondary_seeds if model in neural else 1)
        for model in sorted(required_models)
    }
    universe_gate = {
        universe: required_models.issubset(
            set(metrics.loc[(metrics["fold"] == "primary") & (metrics["universe"] == universe), "model"])
        )
        for universe in ("n5", "n10", "n15")
    }
    gates = {
        "primary_model_seeds": model_seed_gate,
        "transaction_costs": cost_gate,
        "graph_ablation": graph_gate,
        "secondary_fold": secondary_gate,
        "functional_universes": universe_gate,
        "primary_comparisons": len(inference) == len(config["statistics"]["primary_comparators"]),
        "xai_completed": xai.get("status") == "complete",
    }

    def all_true(value: Any) -> bool:
        if isinstance(value, dict):
            return all(all_true(item) for item in value.values())
        return bool(value)

    return {"status": "READY" if all_true(gates) else "INCOMPLETE", "gates": gates}


def aggregate_results(config: Dict[str, Any], factors: pd.DataFrame, data_hash: str) -> Path:
    results_dir = Path(config["paths"]["results_dir"])
    output_dir = results_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = build_ledger(results_dir)
    runs = _successful_backtest_runs(ledger, config, data_hash)
    if not runs:
        raise RuntimeError("No successful backtest runs are available for aggregation")
    metrics = _metrics_rows(runs)
    metrics.to_csv(output_dir / "metrics_all.csv", index=False)
    summary = (
        metrics.groupby(["fold", "universe", "model", "frequency", "graph", "cost_bps"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            cagr_mean=("cagr", "mean"),
            cagr_std=("cagr", "std"),
            sharpe_mean=("sharpe", "mean"),
            sharpe_std=("sharpe", "std"),
            mdd_mean=("mdd", "mean"),
            mdd_std=("mdd", "std"),
            turnover_mean=("mean_turnover", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "summary_by_seed.csv", index=False)
    primary = _primary_filter(config, metrics)
    primary.to_csv(output_dir / "primary_runs.csv", index=False)
    primary_table = (
        primary.groupby(["model", "graph"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            cagr_mean=("cagr", "mean"),
            cagr_std=("cagr", "std"),
            sharpe_mean=("sharpe", "mean"),
            sharpe_std=("sharpe", "std"),
            mdd_mean=("mdd", "mean"),
            mdd_std=("mdd", "std"),
            turnover_mean=("mean_turnover", "mean"),
        )
        .reset_index()
    )
    primary_table.to_csv(output_dir / "primary_table.csv", index=False)
    inference = _primary_inference(config, primary, factors)
    inference.to_csv(output_dir / "primary_comparisons.csv", index=False)
    alpha = _aggregate_alpha(runs, output_dir)
    xai = _aggregate_xai(config, runs, output_dir)
    readiness = _evidence_readiness(config, metrics, primary, inference, xai)
    (output_dir / "evidence_readiness.json").write_text(
        json.dumps(readiness, indent=2), encoding="utf-8"
    )
    _write_figures(config, primary, metrics, output_dir)
    _write_reviewer_evidence(output_dir, primary, inference, alpha, xai, readiness)
    return output_dir
