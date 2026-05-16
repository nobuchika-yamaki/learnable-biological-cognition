#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structural-metric stratification analysis for the v8 fair benchmark.

Purpose
-------
This script does not change any model definition. It takes the completed v8
benchmark outputs and asks:

    Under which structural regimes does each policy principle perform best?

The stratification is based on environment-level structural diagnostics, not on
model outcomes. This is therefore a post hoc interpretive analysis, not a
parameter search to make LVE win.

Required input files
--------------------
From a completed v8 run directory, typically:

    main_analysis/accepted_environment_diagnostics.csv
    main_analysis/per_environment_results.csv

Optional input files are ignored.

Main outputs
------------
    structural_metric_quantiles.csv
    stratum_env_counts.csv
    model_summary_by_stratum.csv
    model_ranks_by_stratum.csv
    best_model_by_stratum.csv
    lve_vs_models_by_stratum.csv
    all_model_pairwise_primary_by_stratum.csv
    pareto_core_by_stratum.csv
    pareto_efficiency_by_stratum.csv
    interpretation_summary.csv
    figures/*.png

Recommended use
---------------
python3 -u ~/Downloads/structural_stratification_v8.py \
  --indir ~/Desktop/lve_v8_main_progress/main_analysis \
  --outdir ~/Desktop/lve_v8_structural_stratification \
  --permutations 5000 \
  --bootstraps 3000

If the file names contain parentheses or are in another folder, provide explicit paths:

python3 -u ~/Downloads/structural_stratification_v8.py \
  --diagnostics ~/Downloads/accepted_environment_diagnostics.csv \
  --results ~/Downloads/per_environment_results.csv \
  --outdir ~/Desktop/lve_v8_structural_stratification
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_ORDER = [
    "learnable_viable_expansion",
    "viability_constrained_endpoint",
    "homeostasis_only",
    "novelty_seeking",
    "random_admissible",
    "model_free_reward",
    "uncertainty_minimizing",
    "unconstrained_controllability",
]

LVE = "learnable_viable_expansion"

PRIMARY = "survival_adjusted_certified_viable_future_expansion"

STRUCTURAL_METRICS = [
    "policy_viability_risk",
    "latent_true_viable_expansion",
    "initial_certification_gap",
    "max_learnable_certified_expansion",
]

OPTIONAL_STRUCTURAL_METRICS = [
    "viability_control_conflict",
    "true_policy_viability_pressure",
]

OUTCOME_HIGH = [
    PRIMARY,
    "survival_adjusted_true_viable_future_expansion",
    "survival_probability",
    "terminal_true_viable_controllability_gain",
    "terminal_certified_viable_controllability_gain",
    "certification_gap_reduction",
    "learning_efficiency",
]

OUTCOME_LOW = [
    "cumulative_viability_violations",
    "cumulative_movement_cost",
    "cumulative_uncertainty_exposure",
]

SELECTED_OUTCOMES = [
    PRIMARY,
    "survival_adjusted_true_viable_future_expansion",
    "survival_probability",
    "certification_gap_reduction",
    "cumulative_viability_violations",
    "cumulative_movement_cost",
]


def find_file(indir: Path, stem: str) -> Path:
    """Find a file robustly when names contain parentheses."""
    candidates = sorted(indir.glob(f"{stem}.csv"))
    if candidates:
        return candidates[-1]
    candidates = sorted(indir.glob(f"{stem}*.csv"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"Could not find {stem}.csv under {indir}")


def load_inputs(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.diagnostics:
        diag_path = Path(args.diagnostics).expanduser()
    else:
        diag_path = find_file(Path(args.indir).expanduser(), "accepted_environment_diagnostics")

    if args.results:
        res_path = Path(args.results).expanduser()
    else:
        res_path = find_file(Path(args.indir).expanduser(), "per_environment_results")

    diag = pd.read_csv(diag_path)
    res = pd.read_csv(res_path)

    required_diag = {"env_id"}
    required_res = {"env_id", "model", PRIMARY}

    miss_diag = required_diag - set(diag.columns)
    miss_res = required_res - set(res.columns)
    if miss_diag:
        raise ValueError(f"diagnostics file missing columns: {sorted(miss_diag)}")
    if miss_res:
        raise ValueError(f"results file missing columns: {sorted(miss_res)}")

    print(f"[load] diagnostics: {diag_path} shape={diag.shape}")
    print(f"[load] results:     {res_path} shape={res.shape}")

    return diag, res


def available_metrics(diag: pd.DataFrame, requested: Optional[str]) -> List[str]:
    if requested:
        metrics = [x.strip() for x in requested.split(",") if x.strip()]
    else:
        metrics = list(STRUCTURAL_METRICS)
        for m in OPTIONAL_STRUCTURAL_METRICS:
            if m in diag.columns and m not in metrics:
                metrics.append(m)

    missing = [m for m in metrics if m not in diag.columns]
    if missing:
        raise ValueError(f"Structural metric(s) not in diagnostics: {missing}")

    return metrics


def assign_tertiles(diag: pd.DataFrame, metric: str) -> pd.Series:
    """Assign low/middle/high bins with robust handling of tied values."""
    x = pd.to_numeric(diag[metric], errors="coerce")
    if x.isna().all():
        raise ValueError(f"Metric {metric} is all NaN")

    # Rank-based qcut avoids failures when many values are identical.
    rank = x.rank(method="first")
    labels = ["low", "middle", "high"]
    try:
        out = pd.qcut(rank, q=3, labels=labels)
    except ValueError:
        # Fallback: use median split if tertiles cannot be formed.
        med = x.median()
        out = pd.Series(np.where(x <= med, "low", "high"), index=diag.index)
    return out.astype(str)


def make_stratified_dataset(diag: pd.DataFrame, res: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    d = diag.copy()
    for metric in metrics:
        d[f"{metric}_tertile"] = assign_tertiles(d, metric)

    keep_cols = ["env_id"] + metrics + [f"{m}_tertile" for m in metrics]
    extra = [
        "source_seed",
        "start_row",
        "start_col",
        "accepted_critical_learning_regime",
    ]
    keep_cols += [c for c in extra if c in d.columns]

    merged = res.merge(d[keep_cols], on="env_id", how="left", validate="many_to_one")
    return merged


def sem(x: pd.Series) -> float:
    n = x.notna().sum()
    if n <= 1:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(n))


def summarize_by_stratum(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    outcomes = [c for c in OUTCOME_HIGH + OUTCOME_LOW if c in df.columns]

    for metric in metrics:
        bin_col = f"{metric}_tertile"
        for stratum in ["low", "middle", "high"]:
            g0 = df[df[bin_col] == stratum]
            if g0.empty:
                continue
            n_env = int(g0["env_id"].nunique())
            for model, g in g0.groupby("model"):
                row = {
                    "stratifying_metric": metric,
                    "stratum": stratum,
                    "n_env": n_env,
                    "model": model,
                    "n_rows": int(len(g)),
                }
                for outcome in outcomes:
                    vals = pd.to_numeric(g[outcome], errors="coerce")
                    row[f"{outcome}_mean"] = float(vals.mean())
                    row[f"{outcome}_sd"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
                    row[f"{outcome}_sem"] = sem(vals)
                    row[f"{outcome}_median"] = float(vals.median())
                    row[f"{outcome}_iqr"] = float(vals.quantile(0.75) - vals.quantile(0.25))
                rows.append(row)

    return pd.DataFrame(rows)


def rank_by_stratum(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, stratum), g in summary.groupby(["stratifying_metric", "stratum"]):
        h = g.copy()
        rank_cols = []
        for outcome in OUTCOME_HIGH:
            col = f"{outcome}_mean"
            if col in h.columns:
                rcol = f"{outcome}_rank"
                h[rcol] = h[col].rank(ascending=False, method="min").astype(int)
                rank_cols.append(rcol)
        for outcome in OUTCOME_LOW:
            col = f"{outcome}_mean"
            if col in h.columns:
                rcol = f"{outcome}_rank"
                h[rcol] = h[col].rank(ascending=True, method="min").astype(int)
                rank_cols.append(rcol)

        h["mean_rank"] = h[rank_cols].mean(axis=1) if rank_cols else np.nan
        h["median_rank"] = h[rank_cols].median(axis=1) if rank_cols else np.nan
        h = h.sort_values(["mean_rank", "median_rank", "model"])
        rows.append(h)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def best_model_by_stratum(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcome_cols = [f"{o}_mean" for o in SELECTED_OUTCOMES if f"{o}_mean" in summary.columns]

    for (metric, stratum), g in summary.groupby(["stratifying_metric", "stratum"]):
        n_env = int(g["n_env"].iloc[0])
        for col in outcome_cols:
            outcome = col.replace("_mean", "")
            high_is_good = outcome not in OUTCOME_LOW
            if high_is_good:
                row = g.loc[g[col].idxmax()]
            else:
                row = g.loc[g[col].idxmin()]
            rows.append({
                "stratifying_metric": metric,
                "stratum": stratum,
                "n_env": n_env,
                "outcome": outcome,
                "best_model": row["model"],
                "best_value": float(row[col]),
                "lve_value": float(g.loc[g["model"] == LVE, col].iloc[0]) if (g["model"] == LVE).any() else np.nan,
                "direction": "higher_is_better" if high_is_good else "lower_is_better",
            })

    return pd.DataFrame(rows)


def paired_perm(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> Tuple[float, float]:
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    obs = float(np.mean(d))
    if len(d) == 0:
        return np.nan, np.nan
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(d)), replace=True)
    null = np.mean(signs * d[None, :], axis=1)
    p = float((1 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1))
    return obs, p


def boot_ci(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if len(d) == 0:
        return np.nan, np.nan
    vals = np.empty(n_boot, dtype=float)
    n = len(d)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(np.mean(d[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(len(p), dtype=float)
    running = 1.0
    for i in range(len(p) - 1, -1, -1):
        rank = i + 1
        running = min(running, p[order[i]] * len(p) / rank)
        adj[order[i]] = min(running, 1.0)
    return adj


def lve_vs_models_by_stratum(df: pd.DataFrame, metrics: List[str], permutations: int, bootstraps: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)

    outcomes = [o for o in SELECTED_OUTCOMES if o in df.columns]
    for metric in metrics:
        bin_col = f"{metric}_tertile"
        for stratum in ["low", "middle", "high"]:
            g0 = df[df[bin_col] == stratum]
            if g0.empty:
                continue
            for outcome in outcomes:
                pivot = g0.pivot(index="env_id", columns="model", values=outcome)
                if LVE not in pivot.columns:
                    continue
                for model in [m for m in pivot.columns if m != LVE]:
                    pair = pivot[[LVE, model]].dropna()
                    if len(pair) < 2:
                        continue
                    x = pair[LVE].to_numpy(float)
                    y = pair[model].to_numpy(float)
                    diff, p = paired_perm(x, y, permutations, rng)
                    lo, hi = boot_ci(x, y, bootstraps, rng)
                    rows.append({
                        "stratifying_metric": metric,
                        "stratum": stratum,
                        "outcome": outcome,
                        "comparison": f"{LVE}_minus_{model}",
                        "model_b": model,
                        "n_env": int(len(pair)),
                        "paired_mean_difference": diff,
                        "bootstrap95_low": lo,
                        "bootstrap95_high": hi,
                        "p_two_sided": p,
                    })

    out = pd.DataFrame(rows)
    if len(out):
        out["p_two_sided_fdr"] = bh_fdr(out["p_two_sided"].to_numpy(float))
    return out


def all_model_pairwise_primary_by_stratum(df: pd.DataFrame, metrics: List[str], permutations: int, bootstraps: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed + 101)

    for metric in metrics:
        bin_col = f"{metric}_tertile"
        for stratum in ["low", "middle", "high"]:
            g0 = df[df[bin_col] == stratum]
            if g0.empty:
                continue
            pivot = g0.pivot(index="env_id", columns="model", values=PRIMARY)
            models = [m for m in MODEL_ORDER if m in pivot.columns] + [m for m in pivot.columns if m not in MODEL_ORDER]
            for a, b in itertools.combinations(models, 2):
                pair = pivot[[a, b]].dropna()
                if len(pair) < 2:
                    continue
                x = pair[a].to_numpy(float)
                y = pair[b].to_numpy(float)
                diff, p = paired_perm(x, y, permutations, rng)
                lo, hi = boot_ci(x, y, bootstraps, rng)
                rows.append({
                    "stratifying_metric": metric,
                    "stratum": stratum,
                    "outcome": PRIMARY,
                    "comparison": f"{a}_minus_{b}",
                    "model_a": a,
                    "model_b": b,
                    "n_env": int(len(pair)),
                    "paired_mean_difference": diff,
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "p_two_sided": p,
                })

    out = pd.DataFrame(rows)
    if len(out):
        out["p_two_sided_fdr"] = bh_fdr(out["p_two_sided"].to_numpy(float))
    return out


def pareto_table_for_group(g: pd.DataFrame, include_movement: bool) -> pd.DataFrame:
    agg = g.groupby("model").agg(
        certified_expansion=(PRIMARY, "mean"),
        true_expansion=("survival_adjusted_true_viable_future_expansion", "mean"),
        survival=("survival_probability", "mean"),
        violations=("cumulative_viability_violations", "mean"),
        gap_reduction=("certification_gap_reduction", "mean"),
        movement=("cumulative_movement_cost", "mean"),
        uncertainty=("cumulative_uncertainty_exposure", "mean"),
    ).reset_index()

    dominated_flags = []
    dominated_by = []

    for _, row in agg.iterrows():
        is_dom = False
        doms = []
        for _, other in agg.iterrows():
            if row["model"] == other["model"]:
                continue

            weak = (
                other["certified_expansion"] >= row["certified_expansion"]
                and other["true_expansion"] >= row["true_expansion"]
                and other["survival"] >= row["survival"]
                and other["violations"] <= row["violations"]
            )
            strict = (
                other["certified_expansion"] > row["certified_expansion"]
                or other["true_expansion"] > row["true_expansion"]
                or other["survival"] > row["survival"]
                or other["violations"] < row["violations"]
            )

            if include_movement:
                weak = weak and other["movement"] <= row["movement"]
                strict = strict or other["movement"] < row["movement"]

            if weak and strict:
                is_dom = True
                doms.append(str(other["model"]))

        dominated_flags.append(is_dom)
        dominated_by.append(";".join(doms))

    agg["pareto_dominated"] = dominated_flags
    agg["dominated_by"] = dominated_by
    return agg.sort_values(
        ["pareto_dominated", "certified_expansion", "true_expansion", "survival"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def pareto_by_stratum(df: pd.DataFrame, metrics: List[str], include_movement: bool) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        bin_col = f"{metric}_tertile"
        for stratum in ["low", "middle", "high"]:
            g0 = df[df[bin_col] == stratum]
            if g0.empty:
                continue
            ptab = pareto_table_for_group(g0, include_movement=include_movement)
            ptab.insert(0, "stratum", stratum)
            ptab.insert(0, "stratifying_metric", metric)
            ptab.insert(2, "n_env", int(g0["env_id"].nunique()))
            rows.append(ptab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def metric_quantiles(diag: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        x = pd.to_numeric(diag[metric], errors="coerce").dropna()
        rows.append({
            "metric": metric,
            "min": float(x.min()),
            "p10": float(x.quantile(0.10)),
            "p25": float(x.quantile(0.25)),
            "p33": float(x.quantile(1/3)),
            "median": float(x.median()),
            "p67": float(x.quantile(2/3)),
            "p75": float(x.quantile(0.75)),
            "p90": float(x.quantile(0.90)),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)),
        })
    return pd.DataFrame(rows)


def stratum_counts(merged: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        bin_col = f"{metric}_tertile"
        for stratum in ["low", "middle", "high"]:
            g = merged[merged[bin_col] == stratum]
            if g.empty:
                continue
            rows.append({
                "stratifying_metric": metric,
                "stratum": stratum,
                "n_env": int(g["env_id"].nunique()),
                "n_rows": int(len(g)),
                "metric_min": float(g[metric].min()),
                "metric_mean": float(g[metric].mean()),
                "metric_max": float(g[metric].max()),
            })
    return pd.DataFrame(rows)


def interpretation_summary(best: pd.DataFrame, ranks: pd.DataFrame, pareto_core: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary_best = best[best["outcome"] == PRIMARY].copy()

    for (metric, stratum), g in primary_best.groupby(["stratifying_metric", "stratum"]):
        best_model = str(g["best_model"].iloc[0])
        lve_value = float(g["lve_value"].iloc[0])
        best_value = float(g["best_value"].iloc[0])
        lve_is_best = best_model == LVE

        r0 = ranks[
            (ranks["stratifying_metric"] == metric)
            & (ranks["stratum"] == stratum)
            & (ranks["model"] == LVE)
        ]
        lve_mean_rank = float(r0["mean_rank"].iloc[0]) if len(r0) and "mean_rank" in r0 else np.nan

        p0 = pareto_core[
            (pareto_core["stratifying_metric"] == metric)
            & (pareto_core["stratum"] == stratum)
            & (pareto_core["model"] == LVE)
        ]
        lve_pareto_dominated = bool(p0["pareto_dominated"].iloc[0]) if len(p0) else np.nan

        rows.append({
            "stratifying_metric": metric,
            "stratum": stratum,
            "primary_best_model": best_model,
            "lve_is_primary_best": bool(lve_is_best),
            "primary_best_value": best_value,
            "lve_primary_value": lve_value,
            "lve_minus_best": float(lve_value - best_value),
            "lve_mean_rank": lve_mean_rank,
            "lve_pareto_dominated_core": lve_pareto_dominated,
            "interpretation": make_interpretation(metric, stratum, best_model, lve_is_best, lve_pareto_dominated),
        })

    return pd.DataFrame(rows)


def make_interpretation(metric: str, stratum: str, best_model: str, lve_is_best: bool, lve_pareto_dominated) -> str:
    if lve_is_best:
        return f"LVE was the primary-best model in the {stratum} tertile of {metric}."
    if lve_pareto_dominated is False:
        return f"{best_model} had the highest primary mean, but LVE remained Pareto-non-dominated in the {stratum} tertile of {metric}."
    return f"{best_model} had the highest primary mean and LVE was Pareto-dominated in the {stratum} tertile of {metric}."


def save_figures(summary: pd.DataFrame, ranks: pd.DataFrame, best: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Figure 1: primary mean by model across strata for each structural metric.
    for metric in summary["stratifying_metric"].unique():
        g = summary[summary["stratifying_metric"] == metric].copy()
        pivot = g.pivot(index="model", columns="stratum", values=f"{PRIMARY}_mean")
        ordered_models = [m for m in MODEL_ORDER if m in pivot.index] + [m for m in pivot.index if m not in MODEL_ORDER]
        pivot = pivot.reindex(index=ordered_models, columns=["low", "middle", "high"])

        plt.figure(figsize=(8, 5))
        im = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto")
        plt.colorbar(im, label="Primary mean")
        plt.xticks(np.arange(3), ["low", "middle", "high"])
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.title(f"Primary outcome by {metric} tertile")
        plt.tight_layout()
        plt.savefig(figdir / f"heatmap_primary_by_{metric}.png", dpi=300)
        plt.close()

    # Figure 2: best model per stratum.
    primary_best = best[best["outcome"] == PRIMARY].copy()
    if len(primary_best):
        metrics = list(primary_best["stratifying_metric"].unique())
        strata = ["low", "middle", "high"]

        model_to_int = {m: i for i, m in enumerate(MODEL_ORDER)}
        mat = np.full((len(metrics), len(strata)), np.nan)
        labels = [["" for _ in strata] for __ in metrics]
        for i, metric in enumerate(metrics):
            for j, stratum in enumerate(strata):
                row = primary_best[
                    (primary_best["stratifying_metric"] == metric)
                    & (primary_best["stratum"] == stratum)
                ]
                if len(row):
                    m = str(row["best_model"].iloc[0])
                    mat[i, j] = model_to_int.get(m, len(model_to_int))
                    labels[i][j] = m.replace("_", "\n")

        plt.figure(figsize=(8, 4.5))
        plt.imshow(mat, aspect="auto")
        plt.xticks(np.arange(len(strata)), strata)
        plt.yticks(np.arange(len(metrics)), metrics)
        for i in range(len(metrics)):
            for j in range(len(strata)):
                if labels[i][j]:
                    plt.text(j, i, labels[i][j], ha="center", va="center", fontsize=7)
        plt.title("Primary-best model by structural stratum")
        plt.tight_layout()
        plt.savefig(figdir / "best_model_by_structural_stratum.png", dpi=300)
        plt.close()

    # Figure 3: LVE mean rank by stratum.
    if "mean_rank" in ranks.columns:
        g = ranks[ranks["model"] == LVE].copy()
        if len(g):
            pivot = g.pivot(index="stratifying_metric", columns="stratum", values="mean_rank").reindex(columns=["low", "middle", "high"])
            plt.figure(figsize=(7, 4.5))
            im = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto")
            plt.colorbar(im, label="LVE mean rank (lower is better)")
            plt.xticks(np.arange(3), ["low", "middle", "high"])
            plt.yticks(np.arange(len(pivot.index)), pivot.index)
            plt.title("LVE benchmark rank by structural stratum")
            plt.tight_layout()
            plt.savefig(figdir / "lve_mean_rank_by_structural_stratum.png", dpi=300)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Structural stratification analysis for v8 benchmark outputs.")
    parser.add_argument("--indir", default="~/Desktop/lve_v8_main_progress/main_analysis")
    parser.add_argument("--diagnostics", default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--metrics", default=None, help="Comma-separated structural metrics. Default: core metrics plus optional available metrics.")
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--bootstraps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    diag, res = load_inputs(args)
    metrics = available_metrics(diag, args.metrics)
    print(f"[metrics] {metrics}")

    merged = make_stratified_dataset(diag, res, metrics)

    # Core outputs.
    qtab = metric_quantiles(diag, metrics)
    counts = stratum_counts(merged, metrics)
    summary = summarize_by_stratum(merged, metrics)
    ranks = rank_by_stratum(summary)
    best = best_model_by_stratum(summary)
    lve_tests = lve_vs_models_by_stratum(merged, metrics, args.permutations, args.bootstraps, args.seed)
    all_primary = all_model_pairwise_primary_by_stratum(merged, metrics, args.permutations, args.bootstraps, args.seed)
    pareto_core = pareto_by_stratum(merged, metrics, include_movement=False)
    pareto_eff = pareto_by_stratum(merged, metrics, include_movement=True)
    interp = interpretation_summary(best, ranks, pareto_core)

    # Save.
    merged.to_csv(outdir / "stratified_environment_model_dataset.csv", index=False)
    qtab.to_csv(outdir / "structural_metric_quantiles.csv", index=False)
    counts.to_csv(outdir / "stratum_env_counts.csv", index=False)
    summary.to_csv(outdir / "model_summary_by_stratum.csv", index=False)
    ranks.to_csv(outdir / "model_ranks_by_stratum.csv", index=False)
    best.to_csv(outdir / "best_model_by_stratum.csv", index=False)
    lve_tests.to_csv(outdir / "lve_vs_models_by_stratum.csv", index=False)
    all_primary.to_csv(outdir / "all_model_pairwise_primary_by_stratum.csv", index=False)
    pareto_core.to_csv(outdir / "pareto_core_by_stratum.csv", index=False)
    pareto_eff.to_csv(outdir / "pareto_efficiency_by_stratum.csv", index=False)
    interp.to_csv(outdir / "interpretation_summary.csv", index=False)

    save_figures(summary, ranks, best, outdir)

    print("[done] structural stratification outputs written to:")
    print(outdir)
    print("")
    print("Key files:")
    for name in [
        "interpretation_summary.csv",
        "best_model_by_stratum.csv",
        "model_summary_by_stratum.csv",
        "model_ranks_by_stratum.csv",
        "lve_vs_models_by_stratum.csv",
        "pareto_core_by_stratum.csv",
    ]:
        print(f"  {outdir / name}")


if __name__ == "__main__":
    main()
