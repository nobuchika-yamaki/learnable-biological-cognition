#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_v8_ofat_robustness_v2.py

One-factor-at-a-time robustness analysis for the v8 fair benchmark.

This is a supplementary sensitivity check, not a parameter search.

It reuses one fixed set of environments screened under the baseline setting, then
evaluates all models under pre-specified one-factor perturbations.

Baseline:
    Hpi=4, HC=4, theta=0.25, beta=1.0, eta_M=0.0

Perturbations:
    eta_M in {0.01, 0.05}
    Hpi in {3, 5}
    HC in {3, 5}
    theta in {0.20, 0.30}
    beta in {0.5, 1.5}

Recommended supplementary run:
python3 -u ~/Downloads/run_v8_ofat_robustness.py \
  --v8-script ~/Downloads/learnable_viable_expansion_v8_progress.py \
  --outdir ~/Desktop/lve_v8_ofat_robustness \
  --envs 100 \
  --episodes 10 \
  --T 80 \
  --workers 4 \
  --permutations 1000 \
  --bootstraps 1000 \
  2>&1 | tee ~/Desktop/lve_v8_ofat_robustness.log
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


MODEL_NAMES = (
    "random_admissible",
    "homeostasis_only",
    "uncertainty_minimizing",
    "novelty_seeking",
    "model_free_reward",
    "unconstrained_controllability",
    "viability_constrained_endpoint",
    "learnable_viable_expansion",
)

LVE = "learnable_viable_expansion"
PRIMARY = "survival_adjusted_certified_viable_future_expansion"


def load_v8(script_path: str):
    """Load the fixed v8 script under an importable module name.

    The previous version loaded the script under a synthetic name
    "lve_v8_progress_module". That works in the parent process but fails when
    ProcessPoolExecutor pickles functions/classes defined in the loaded module:
    child processes try to import that synthetic module and cannot find it.

    This version loads the script under its real file stem and adds the script
    directory to sys.path. Child processes can then import the module normally.
    """
    path = Path(script_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    module_name = path.stem
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def label_value(x: float) -> str:
    return f"{x:g}".replace(".", "p").replace("-", "m")


def make_base_config(lve, args):
    return lve.Config(
        N=args.N,
        H_pi=4,
        H_C=4,
        theta=0.25,
        beta=1.0,
        eta_M=0.0,
        episodes=args.episodes,
        T=args.T,
        n_env=args.envs,
        base_seed=args.base_seed,
        risk_min=args.risk_min,
        risk_max=args.risk_max,
        risk_margin=args.risk_margin,
        expansion_min=args.expansion_min,
        conflict_min=args.conflict_min,
        max_candidates=args.max_candidates,
        screen_batch_size=args.screen_batch_size,
        n_permutations=args.permutations,
        n_bootstrap=args.bootstraps,
    )


def robustness_settings(base_cfg) -> List[Tuple[str, object, Dict]]:
    settings: List[Tuple[str, object, Dict]] = []

    def add(label: str, cfg, factor: str, value):
        settings.append((label, cfg, {"factor": factor, "value": value}))

    add("baseline", base_cfg, "baseline", "baseline")

    for eta in (0.01, 0.05):
        add(f"etaM_{label_value(eta)}", replace(base_cfg, eta_M=eta), "eta_M", eta)

    for hp in (3, 5):
        add(f"Hpi_{hp}", replace(base_cfg, H_pi=hp), "H_pi", hp)

    for hc in (3, 5):
        add(f"HC_{hc}", replace(base_cfg, H_C=hc), "H_C", hc)

    for theta in (0.20, 0.30):
        add(f"theta_{label_value(theta)}", replace(base_cfg, theta=theta), "theta", theta)

    for beta in (0.5, 1.5):
        add(f"beta_{label_value(beta)}", replace(base_cfg, beta=beta), "beta", beta)

    return settings


def save_reference_environment_info(envs, diag: pd.DataFrame, log: pd.DataFrame, outdir: Path) -> None:
    ref = outdir / "reference_environments"
    ref.mkdir(parents=True, exist_ok=True)
    diag.to_csv(ref / "accepted_environment_diagnostics.csv", index=False)
    log.to_csv(ref / "screening_log.csv", index=False)
    pd.DataFrame(
        [
            {
                "env_id": i,
                "source_seed": int(e.source_seed),
                "start_row": int(e.start_pos[0]),
                "start_col": int(e.start_pos[1]),
            }
            for i, e in enumerate(envs)
        ]
    ).to_csv(ref / "reused_environment_seeds.csv", index=False)


def run_setting(lve, label: str, cfg, envs, meta: Dict, setting_dir: Path, workers: int) -> Dict:
    setting_dir.mkdir(parents=True, exist_ok=True)

    print("")
    print("=" * 80, flush=True)
    print(f"[setting] {label}", flush=True)
    print("=" * 80, flush=True)
    print(
        f"[setting] factor={meta['factor']} value={meta['value']} "
        f"Hpi={cfg.H_pi} HC={cfg.H_C} theta={cfg.theta} beta={cfg.beta} eta_M={cfg.eta_M}",
        flush=True,
    )

    pd.DataFrame([cfg.__dict__ | {"setting_label": label, **meta}]).to_csv(
        setting_dir / "config_used.csv", index=False
    )

    res = lve.run_all(
        envs,
        cfg,
        workers,
        progress_csv=setting_dir / "per_environment_results_incremental.csv",
    )

    summ = lve.summary_by_model(res)
    primary, secondary = lve.stats(res, cfg, label)
    all_primary = lve.all_pairwise_tests(res, cfg, PRIMARY, label)
    rankings = lve.benchmark_rankings(res)
    pareto = lve.pareto_front_table(res)

    res.to_csv(setting_dir / "per_environment_results.csv", index=False)
    summ.to_csv(setting_dir / "summary_by_model.csv", index=False)
    primary.to_csv(setting_dir / "primary_pairwise_tests.csv", index=False)
    secondary.to_csv(setting_dir / "secondary_pairwise_tests_fdr.csv", index=False)
    all_primary.to_csv(setting_dir / "all_model_pairwise_primary_tests_fdr.csv", index=False)
    rankings.to_csv(setting_dir / "benchmark_rankings.csv", index=False)
    pareto.to_csv(setting_dir / "pareto_front_summary.csv", index=False)

    row: Dict = {
        "setting_label": label,
        "factor": meta["factor"],
        "value": meta["value"],
        "n_env": int(cfg.n_env),
        "episodes": int(cfg.episodes),
        "T": int(cfg.T),
        "H_pi": int(cfg.H_pi),
        "H_C": int(cfg.H_C),
        "theta": float(cfg.theta),
        "beta": float(cfg.beta),
        "eta_M": float(cfg.eta_M),
    }

    primary_col = f"{PRIMARY}_mean"
    means = {}
    for _, r in summ.iterrows():
        model = r["model"]
        means[model] = float(r[primary_col])
        row[f"{model}_primary_mean"] = float(r[primary_col])

    best_model = max(means, key=means.get)
    row["best_primary_model"] = best_model
    row["best_primary_value"] = float(means[best_model])
    row["lve_primary_mean"] = float(means.get(LVE, np.nan))
    row["lve_primary_rank"] = int(1 + sum(v > row["lve_primary_mean"] for v in means.values()))
    row["lve_is_best_primary"] = bool(best_model == LVE)

    if len(primary):
        for _, r in primary.iterrows():
            comp = str(r["comparison"])
            row[f"{comp}_mean_diff"] = float(r["paired_mean_difference"])
            row[f"{comp}_p"] = float(r["p_two_sided"])

    if "mean_rank" in rankings.columns and (rankings["model"] == LVE).any():
        row["lve_benchmark_mean_rank"] = float(rankings.loc[rankings["model"] == LVE, "mean_rank"].iloc[0])
        row["best_benchmark_model"] = str(rankings.sort_values(["mean_rank", "median_rank", "model"]).iloc[0]["model"])
    else:
        row["lve_benchmark_mean_rank"] = np.nan
        row["best_benchmark_model"] = ""

    if "pareto_dominated" in pareto.columns and (pareto["model"] == LVE).any():
        row["lve_pareto_dominated"] = bool(pareto.loc[pareto["model"] == LVE, "pareto_dominated"].iloc[0])
    else:
        row["lve_pareto_dominated"] = np.nan

    return row


def aggregate(rows: List[Dict], outdir: Path) -> None:
    agg = outdir / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(agg / "robustness_setting_summary.csv", index=False)

    best_counts = summary["best_primary_model"].value_counts().rename_axis("model").reset_index(name="best_primary_count")
    best_counts.to_csv(agg / "robustness_best_model_counts.csv", index=False)

    contrast_cols = [c for c in summary.columns if c.startswith(f"{LVE}_minus_") and c.endswith("_mean_diff")]
    contrast_rows = []
    for col in contrast_cols:
        pcol = col.replace("_mean_diff", "_p")
        vals = summary[col].dropna()
        if len(vals) == 0:
            continue
        contrast_rows.append(
            {
                "contrast": col.replace("_mean_diff", ""),
                "n_settings": int(len(vals)),
                "mean_diff": float(vals.mean()),
                "min_diff": float(vals.min()),
                "max_diff": float(vals.max()),
                "positive_count": int((vals > 0).sum()),
                "median_p": float(summary[pcol].median()) if pcol in summary.columns else np.nan,
            }
        )
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(agg / "robustness_lve_contrasts.csv", index=False)

    pareto_status = summary[["setting_label", "factor", "value", "lve_pareto_dominated"]].copy()
    pareto_status.to_csv(agg / "robustness_pareto_status.csv", index=False)

    lines = []
    lines.append("V8 OFAT ROBUSTNESS REPORT")
    lines.append("=" * 80)
    lines.append(f"Total settings: {len(summary)}")
    lines.append("")
    lines.append("Best primary model counts:")
    for _, r in best_counts.iterrows():
        lines.append(f"  {r['model']}: {int(r['best_primary_count'])}/{len(summary)}")
    lines.append("")
    lines.append(f"LVE best-primary settings: {int(summary['lve_is_best_primary'].sum())}/{len(summary)}")
    lines.append(f"LVE median primary rank: {float(summary['lve_primary_rank'].median()):.3f}")
    lines.append(f"LVE mean primary rank: {float(summary['lve_primary_rank'].mean()):.3f}")
    if "lve_benchmark_mean_rank" in summary.columns:
        lines.append(f"LVE median benchmark mean-rank: {float(summary['lve_benchmark_mean_rank'].median()):.3f}")
        lines.append(f"LVE mean benchmark mean-rank: {float(summary['lve_benchmark_mean_rank'].mean()):.3f}")
    if "lve_pareto_dominated" in summary.columns:
        nd = int((~summary["lve_pareto_dominated"].astype(bool)).sum())
        lines.append(f"LVE Pareto-non-dominated settings: {nd}/{len(summary)}")
    lines.append("")
    lines.append("LVE primary contrasts:")
    for row in contrast_rows:
        lines.append(
            f"  {row['contrast']}: mean_diff={row['mean_diff']:.6f}, "
            f"positive={row['positive_count']}/{row['n_settings']}, "
            f"median_p={row['median_p']:.6f}"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  This is a sensitivity check, not a parameter search.")
    lines.append("  The main question is whether qualitative ranking/Pareto patterns remain stable")
    lines.append("  when one pre-specified parameter is varied at a time.")

    (agg / "robustness_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("[aggregate] robustness outputs:")
    for fname in [
        "robustness_setting_summary.csv",
        "robustness_lve_contrasts.csv",
        "robustness_best_model_counts.csv",
        "robustness_pareto_status.csv",
        "robustness_report.txt",
    ]:
        print(f"  {agg / fname}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OFAT robustness analysis for v8 fair benchmark.")
    p.add_argument("--v8-script", default="~/Downloads/learnable_viable_expansion_v8_progress.py")
    p.add_argument("--outdir", required=True)

    p.add_argument("--envs", type=int, default=100)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--T", type=int, default=80)
    p.add_argument("--N", type=int, default=15)
    p.add_argument("--base-seed", type=int, default=92345)

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--bootstraps", type=int, default=1000)

    p.add_argument("--risk-min", type=float, default=0.0)
    p.add_argument("--risk-max", type=float, default=1.0)
    p.add_argument("--risk-margin", type=float, default=0.02)
    p.add_argument("--expansion-min", type=float, default=0.001)
    p.add_argument("--conflict-min", type=float, default=-1.0)
    p.add_argument("--max-candidates", type=int, default=50000)
    p.add_argument("--screen-batch-size", type=int, default=64)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    lve = load_v8(args.v8_script)
    base_cfg = make_base_config(lve, args)

    print("[robustness] OFAT robustness for v8 fair benchmark", flush=True)
    print(f"[robustness] outdir={outdir}", flush=True)
    print(f"[robustness] envs={args.envs}, episodes={args.episodes}, T={args.T}", flush=True)

    print("")
    print("=" * 80, flush=True)
    print("[screen] selecting fixed reference environments under baseline", flush=True)
    print("=" * 80, flush=True)
    envs, diag, log = lve.collect_envs(base_cfg, args.workers)
    save_reference_environment_info(envs, diag, log, outdir)

    rows: List[Dict] = []
    settings_dir = outdir / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)

    for label, cfg, meta in robustness_settings(base_cfg):
        row = run_setting(lve, label, cfg, envs, meta, settings_dir / label, args.workers)
        rows.append(row)
        pd.DataFrame(rows).to_csv(outdir / "robustness_setting_summary_incremental.csv", index=False)

    aggregate(rows, outdir)
    print("[done] robustness analysis complete", flush=True)


if __name__ == "__main__":
    main()
