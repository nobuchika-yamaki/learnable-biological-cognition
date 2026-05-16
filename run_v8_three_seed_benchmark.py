#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_v8_three_seed_benchmark.py

Runs the final fair-benchmark design:

    n = 200 × 3 independent seeds
    structural stratification for each seed
    cross-seed aggregation

This script DOES NOT change model definitions.
It only executes the already-fixed v8 fair benchmark and the structural
stratification analysis, then aggregates outputs across seeds.

Required scripts
----------------
1. learnable_viable_expansion_v8_progress.py
2. structural_stratification_v8.py

Default expected locations:
    ~/Downloads/learnable_viable_expansion_v8_progress.py
    ~/Downloads/structural_stratification_v8.py

Recommended run
---------------
python3 -u ~/Downloads/run_v8_three_seed_benchmark.py \
  --outdir ~/Desktop/lve_v8_three_seed \
  --v8-script ~/Downloads/learnable_viable_expansion_v8_progress.py \
  --strat-script ~/Downloads/structural_stratification_v8.py \
  --seeds 12345,22345,32345 \
  --envs 200 \
  --episodes 10 \
  --T 80 \
  --H-pi 4 \
  --H-C 4 \
  --eta-M 0.0 \
  --workers 4 \
  --permutations 5000 \
  --bootstraps 3000 \
  2>&1 | tee ~/Desktop/lve_v8_three_seed.log

Outputs
-------
<outdir>/
  seed_12345/
    main_analysis/
    structural_stratification/
  seed_22345/
    main_analysis/
    structural_stratification/
  seed_32345/
    main_analysis/
    structural_stratification/
  aggregate/
    combined_summary_by_model.csv
    model_primary_by_seed.csv
    model_primary_across_seed_summary.csv
    model_benchmark_rank_across_seed_summary.csv
    model_pareto_status_by_seed.csv
    lve_primary_contrasts_by_seed.csv
    lve_primary_contrasts_across_seed_summary.csv
    combined_structural_model_summary.csv
    structural_primary_by_seed.csv
    structural_primary_across_seed_summary.csv
    structural_best_model_counts.csv
    structural_lve_status_by_seed.csv
    structural_lve_status_summary.csv
    aggregate_report.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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

CORE_STRAT_METRICS = [
    "policy_viability_risk",
    "max_learnable_certified_expansion",
]

OPTIONAL_STRAT_METRICS = [
    "viability_control_conflict",
]

SELECTED_OUTCOMES = [
    PRIMARY,
    "survival_adjusted_true_viable_future_expansion",
    "survival_probability",
    "certification_gap_reduction",
    "cumulative_viability_violations",
    "cumulative_movement_cost",
]


def parse_seed_list(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("No seeds provided")
    return out


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    print("")
    print("[cmd] " + " ".join(cmd), flush=True)
    subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=True)


def file_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_one_seed(args, seed: int) -> None:
    seed_dir = Path(args.outdir).expanduser() / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    main_dir = seed_dir / "main_analysis"
    strat_dir = seed_dir / "structural_stratification"

    final_required = main_dir / "summary_by_model.csv"
    final_strat_required = strat_dir / "interpretation_summary.csv"

    if args.skip_existing and file_exists(final_required):
        print(f"[skip] seed {seed}: benchmark already complete: {final_required}", flush=True)
    else:
        if not Path(args.v8_script).expanduser().exists():
            raise FileNotFoundError(Path(args.v8_script).expanduser())

        cmd = [
            sys.executable,
            "-u",
            str(Path(args.v8_script).expanduser()),
            "--outdir",
            str(seed_dir),
            "--envs",
            str(args.envs),
            "--episodes",
            str(args.episodes),
            "--T",
            str(args.T),
            "--H-pi",
            str(args.H_pi),
            "--H-C",
            str(args.H_C),
            "--eta-M",
            str(args.eta_M),
            "--base-seed",
            str(seed),
            "--workers",
            str(args.workers),
            "--permutations",
            str(args.permutations),
            "--bootstraps",
            str(args.bootstraps),
            "--risk-min",
            str(args.risk_min),
            "--risk-max",
            str(args.risk_max),
            "--risk-margin",
            str(args.risk_margin),
            "--expansion-min",
            str(args.expansion_min),
            "--conflict-min",
            str(args.conflict_min),
            "--max-candidates",
            str(args.max_candidates),
            "--screen-batch-size",
            str(args.screen_batch_size),
        ]
        run_cmd(cmd)

    if args.skip_existing and file_exists(final_strat_required):
        print(f"[skip] seed {seed}: stratification already complete: {final_strat_required}", flush=True)
    else:
        if not Path(args.strat_script).expanduser().exists():
            raise FileNotFoundError(Path(args.strat_script).expanduser())

        if not file_exists(main_dir / "accepted_environment_diagnostics.csv"):
            raise FileNotFoundError(main_dir / "accepted_environment_diagnostics.csv")
        if not file_exists(main_dir / "per_environment_results.csv"):
            raise FileNotFoundError(main_dir / "per_environment_results.csv")

        metrics = args.strat_metrics
        if not metrics:
            # Restrict to interpretable metrics by default.
            metrics = ",".join(CORE_STRAT_METRICS + OPTIONAL_STRAT_METRICS)

        cmd = [
            sys.executable,
            "-u",
            str(Path(args.strat_script).expanduser()),
            "--indir",
            str(main_dir),
            "--outdir",
            str(strat_dir),
            "--metrics",
            metrics,
            "--permutations",
            str(args.permutations),
            "--bootstraps",
            str(args.bootstraps),
            "--seed",
            str(seed),
        ]
        run_cmd(cmd)


def safe_read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def aggregate_outputs(outdir: Path, seeds: List[int]) -> None:
    agg_dir = outdir / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    rankings = []
    paretos = []
    primary_tests = []
    all_primary_tests = []
    structural_summaries = []
    structural_best = []
    structural_lve_tests = []
    structural_interpretation = []
    structural_pareto = []
    env_diag = []

    for seed in seeds:
        seed_dir = outdir / f"seed_{seed}"
        main = seed_dir / "main_analysis"
        strat = seed_dir / "structural_stratification"

        summary = safe_read_csv(main / "summary_by_model.csv")
        summary.insert(0, "seed", seed)
        summaries.append(summary)

        rank = safe_read_csv(main / "benchmark_rankings.csv")
        rank.insert(0, "seed", seed)
        rankings.append(rank)

        pareto = safe_read_csv(main / "pareto_front_summary.csv")
        pareto.insert(0, "seed", seed)
        paretos.append(pareto)

        pt = safe_read_csv(main / "primary_pairwise_tests.csv")
        pt.insert(0, "seed", seed)
        primary_tests.append(pt)

        apt = safe_read_csv(main / "all_model_pairwise_primary_tests_fdr.csv")
        apt.insert(0, "seed", seed)
        all_primary_tests.append(apt)

        diag = safe_read_csv(main / "accepted_environment_diagnostics.csv")
        diag.insert(0, "seed", seed)
        env_diag.append(diag)

        ss = safe_read_csv(strat / "model_summary_by_stratum.csv")
        ss.insert(0, "seed", seed)
        structural_summaries.append(ss)

        sb = safe_read_csv(strat / "best_model_by_stratum.csv")
        sb.insert(0, "seed", seed)
        structural_best.append(sb)

        sl = safe_read_csv(strat / "lve_vs_models_by_stratum.csv")
        sl.insert(0, "seed", seed)
        structural_lve_tests.append(sl)

        si = safe_read_csv(strat / "interpretation_summary.csv")
        si.insert(0, "seed", seed)
        structural_interpretation.append(si)

        sp = safe_read_csv(strat / "pareto_core_by_stratum.csv")
        sp.insert(0, "seed", seed)
        structural_pareto.append(sp)

    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_rankings = pd.concat(rankings, ignore_index=True)
    combined_pareto = pd.concat(paretos, ignore_index=True)
    combined_primary_tests = pd.concat(primary_tests, ignore_index=True)
    combined_all_primary = pd.concat(all_primary_tests, ignore_index=True)
    combined_diag = pd.concat(env_diag, ignore_index=True)
    combined_struct_summary = pd.concat(structural_summaries, ignore_index=True)
    combined_struct_best = pd.concat(structural_best, ignore_index=True)
    combined_struct_lve = pd.concat(structural_lve_tests, ignore_index=True)
    combined_struct_interp = pd.concat(structural_interpretation, ignore_index=True)
    combined_struct_pareto = pd.concat(structural_pareto, ignore_index=True)

    combined_summary.to_csv(agg_dir / "combined_summary_by_model.csv", index=False)
    combined_rankings.to_csv(agg_dir / "combined_benchmark_rankings.csv", index=False)
    combined_pareto.to_csv(agg_dir / "combined_pareto_front_summary.csv", index=False)
    combined_primary_tests.to_csv(agg_dir / "combined_lve_primary_pairwise_tests.csv", index=False)
    combined_all_primary.to_csv(agg_dir / "combined_all_model_pairwise_primary_tests.csv", index=False)
    combined_diag.to_csv(agg_dir / "combined_accepted_environment_diagnostics.csv", index=False)
    combined_struct_summary.to_csv(agg_dir / "combined_structural_model_summary.csv", index=False)
    combined_struct_best.to_csv(agg_dir / "combined_structural_best_model.csv", index=False)
    combined_struct_lve.to_csv(agg_dir / "combined_structural_lve_tests.csv", index=False)
    combined_struct_interp.to_csv(agg_dir / "combined_structural_interpretation.csv", index=False)
    combined_struct_pareto.to_csv(agg_dir / "combined_structural_pareto_core.csv", index=False)

    # Model primary by seed.
    primary_col = f"{PRIMARY}_mean"
    model_primary = combined_summary[["seed", "model", primary_col]].copy()
    model_primary = model_primary.rename(columns={primary_col: "primary_mean"})
    model_primary["primary_rank_within_seed"] = model_primary.groupby("seed")["primary_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    model_primary.to_csv(agg_dir / "model_primary_by_seed.csv", index=False)

    primary_summary = model_primary.groupby("model").agg(
        n_seeds=("seed", "nunique"),
        primary_mean_across_seeds=("primary_mean", "mean"),
        primary_sd_across_seeds=("primary_mean", "std"),
        primary_min=("primary_mean", "min"),
        primary_max=("primary_mean", "max"),
        mean_rank=("primary_rank_within_seed", "mean"),
        median_rank=("primary_rank_within_seed", "median"),
        best_seed_count=("primary_rank_within_seed", lambda x: int((x == 1).sum())),
    ).reset_index()
    primary_summary = primary_summary.sort_values(["mean_rank", "primary_mean_across_seeds"], ascending=[True, False])
    primary_summary.to_csv(agg_dir / "model_primary_across_seed_summary.csv", index=False)

    # Benchmark rank summary across seeds.
    if "mean_rank" in combined_rankings.columns:
        rank_summary = combined_rankings.groupby("model").agg(
            n_seeds=("seed", "nunique"),
            benchmark_mean_rank=("mean_rank", "mean"),
            benchmark_sd_rank=("mean_rank", "std"),
            benchmark_median_rank=("mean_rank", "median"),
        ).reset_index()
        rank_summary = rank_summary.sort_values(["benchmark_mean_rank", "benchmark_median_rank", "model"])
        rank_summary.to_csv(agg_dir / "model_benchmark_rank_across_seed_summary.csv", index=False)

    # Pareto status by seed.
    pareto_status = combined_pareto[["seed", "model", "pareto_dominated"]].copy()
    pareto_status.to_csv(agg_dir / "model_pareto_status_by_seed.csv", index=False)
    pareto_summary = pareto_status.groupby("model").agg(
        n_seeds=("seed", "nunique"),
        nondominated_count=("pareto_dominated", lambda x: int((~x.astype(bool)).sum())),
        dominated_count=("pareto_dominated", lambda x: int(x.astype(bool).sum())),
    ).reset_index()
    pareto_summary.to_csv(agg_dir / "model_pareto_status_across_seed_summary.csv", index=False)

    # LVE contrasts across seeds.
    lve_primary = combined_primary_tests.copy()
    lve_primary.to_csv(agg_dir / "lve_primary_contrasts_by_seed.csv", index=False)
    if "comparison" in lve_primary.columns:
        contrast_summary = lve_primary.groupby("comparison").agg(
            n_seeds=("seed", "nunique"),
            mean_difference=("paired_mean_difference", "mean"),
            sd_difference=("paired_mean_difference", "std"),
            min_difference=("paired_mean_difference", "min"),
            max_difference=("paired_mean_difference", "max"),
            positive_count=("paired_mean_difference", lambda x: int((x > 0).sum())),
            p_median=("p_two_sided", "median"),
        ).reset_index()
        contrast_summary.to_csv(agg_dir / "lve_primary_contrasts_across_seed_summary.csv", index=False)

    # Structural primary per seed.
    struct_primary_col = f"{PRIMARY}_mean"
    strat_primary = combined_struct_summary[
        ["seed", "stratifying_metric", "stratum", "model", "n_env", struct_primary_col]
    ].copy()
    strat_primary = strat_primary.rename(columns={struct_primary_col: "primary_mean"})
    strat_primary["primary_rank_within_seed_stratum"] = strat_primary.groupby(
        ["seed", "stratifying_metric", "stratum"]
    )["primary_mean"].rank(ascending=False, method="min").astype(int)
    strat_primary.to_csv(agg_dir / "structural_primary_by_seed.csv", index=False)

    strat_summary = strat_primary.groupby(["stratifying_metric", "stratum", "model"]).agg(
        n_seeds=("seed", "nunique"),
        primary_mean_across_seeds=("primary_mean", "mean"),
        primary_sd_across_seeds=("primary_mean", "std"),
        mean_rank=("primary_rank_within_seed_stratum", "mean"),
        best_count=("primary_rank_within_seed_stratum", lambda x: int((x == 1).sum())),
    ).reset_index()
    strat_summary = strat_summary.sort_values(
        ["stratifying_metric", "stratum", "mean_rank", "primary_mean_across_seeds"],
        ascending=[True, True, True, False],
    )
    strat_summary.to_csv(agg_dir / "structural_primary_across_seed_summary.csv", index=False)

    # Best model counts across seeds and strata.
    primary_best = combined_struct_best[combined_struct_best["outcome"] == PRIMARY].copy()
    best_counts = primary_best.groupby(["stratifying_metric", "stratum", "best_model"]).agg(
        count=("seed", "count"),
        seeds=("seed", lambda x: ",".join(map(str, sorted(set(x))))),
        mean_best_value=("best_value", "mean"),
        mean_lve_value=("lve_value", "mean"),
    ).reset_index()
    best_counts = best_counts.sort_values(["stratifying_metric", "stratum", "count"], ascending=[True, True, False])
    best_counts.to_csv(agg_dir / "structural_best_model_counts.csv", index=False)

    # LVE status in structural strata.
    status_rows = []
    for (metric, stratum), g in primary_best.groupby(["stratifying_metric", "stratum"]):
        lve_best_count = int((g["best_model"] == LVE).sum())
        status_rows.append({
            "stratifying_metric": metric,
            "stratum": stratum,
            "n_seeds": int(g["seed"].nunique()),
            "lve_primary_best_count": lve_best_count,
            "lve_primary_best_rate": lve_best_count / max(1, int(g["seed"].nunique())),
            "mean_lve_minus_best": float((g["lve_value"] - g["best_value"]).mean()),
            "best_models": ";".join(
                f"{m}:{c}" for m, c in g["best_model"].value_counts().items()
            ),
        })
    pd.DataFrame(status_rows).to_csv(agg_dir / "structural_lve_status_summary.csv", index=False)
    primary_best.to_csv(agg_dir / "structural_lve_status_by_seed.csv", index=False)

    # Short plain-text report.
    write_report(agg_dir, seeds, primary_summary, combined_primary_tests, strat_summary, best_counts, pareto_summary)


def write_report(
    agg_dir: Path,
    seeds: List[int],
    primary_summary: pd.DataFrame,
    lve_primary: pd.DataFrame,
    strat_summary: pd.DataFrame,
    best_counts: pd.DataFrame,
    pareto_summary: pd.DataFrame,
) -> None:
    lines = []
    lines.append("V8 THREE-SEED FAIR BENCHMARK AGGREGATE REPORT")
    lines.append("=" * 80)
    lines.append(f"Seeds: {', '.join(map(str, seeds))}")
    lines.append(f"Number of seeds: {len(seeds)}")
    lines.append("")

    lines.append("Overall primary outcome ranking across seeds:")
    for _, row in primary_summary.iterrows():
        lines.append(
            f"  {row['model']}: primary_mean={row['primary_mean_across_seeds']:.6f}, "
            f"mean_rank={row['mean_rank']:.3f}, best_seed_count={int(row['best_seed_count'])}/{len(seeds)}"
        )
    lines.append("")

    if len(pareto_summary):
        lines.append("Pareto non-dominated counts across seeds:")
        for _, row in pareto_summary.iterrows():
            lines.append(
                f"  {row['model']}: non-dominated {int(row['nondominated_count'])}/{len(seeds)}"
            )
        lines.append("")

    if len(lve_primary):
        lines.append("LVE primary contrasts across seeds:")
        for comp, g in lve_primary.groupby("comparison"):
            lines.append(
                f"  {comp}: mean_diff={g['paired_mean_difference'].mean():.6f}, "
                f"positive_count={int((g['paired_mean_difference'] > 0).sum())}/{len(g)}, "
                f"median_p={g['p_two_sided'].median():.6f}"
            )
        lines.append("")

    lines.append("Structural best-model counts:")
    if len(best_counts):
        for (metric, stratum), g in best_counts.groupby(["stratifying_metric", "stratum"]):
            counts = ", ".join([f"{r['best_model']}={int(r['count'])}" for _, r in g.iterrows()])
            lines.append(f"  {metric} / {stratum}: {counts}")
    lines.append("")

    lines.append("Interpretive rule:")
    lines.append(
        "Use this analysis to report condition-dependent policy advantage, not to claim a universal winner."
    )
    lines.append(
        "The main evidence is whether LVE remains high-ranking or Pareto-non-dominated across seeds and whether its advantage depends on structural regimes."
    )

    (agg_dir / "aggregate_report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run n=200 × 3-seed v8 benchmark and structural stratification.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--v8-script", default="~/Downloads/learnable_viable_expansion_v8_progress.py")
    p.add_argument("--strat-script", default="~/Downloads/structural_stratification_v8.py")
    p.add_argument("--seeds", default="12345,22345,32345")

    p.add_argument("--envs", type=int, default=200)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--T", type=int, default=80)
    p.add_argument("--H-pi", "--Hpi", dest="H_pi", type=int, default=4)
    p.add_argument("--H-C", "--HC", dest="H_C", type=int, default=4)
    p.add_argument("--eta-M", dest="eta_M", type=float, default=0.0)

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--permutations", type=int, default=5000)
    p.add_argument("--bootstraps", type=int, default=3000)

    p.add_argument("--risk-min", type=float, default=0.0)
    p.add_argument("--risk-max", type=float, default=1.0)
    p.add_argument("--risk-margin", type=float, default=0.02)
    p.add_argument("--expansion-min", type=float, default=0.001)
    p.add_argument("--conflict-min", type=float, default=-1.0)
    p.add_argument("--max-candidates", type=int, default=50000)
    p.add_argument("--screen-batch-size", type=int, default=64)

    p.add_argument(
        "--strat-metrics",
        default="policy_viability_risk,max_learnable_certified_expansion,viability_control_conflict",
        help="Comma-separated structural metrics used for tertile stratification.",
    )

    p.add_argument("--aggregate-only", action="store_true", help="Skip runs and aggregate existing seed folders.")
    p.add_argument("--skip-existing", action="store_true", help="Skip completed benchmark/stratification runs.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seed_list(args.seeds)

    if not args.aggregate_only:
        for seed in seeds:
            print("")
            print("=" * 80)
            print(f"[seed] {seed}")
            print("=" * 80)
            run_one_seed(args, seed)

    print("")
    print("=" * 80)
    print("[aggregate] combining seed outputs")
    print("=" * 80)
    aggregate_outputs(outdir, seeds)
    print(f"[done] aggregate outputs: {outdir / 'aggregate'}")


if __name__ == "__main__":
    main()
