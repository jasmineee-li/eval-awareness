"""Step 1 of 2026-04-20 eval-gaming-gap deep-dive plan.

Bootstrap the eval-gaming gap (H|Eval-Unaware - H|Eval-Aware) per (model x warning),
run 2-prop z-tests to decompose whether the gap shrink on coop-SDF models is
driven by H|Aware rising or H|Unaware falling, and formalize the H8 ruling-out
(coop vs base on H|Unaware should be null; muan vs base on H|Unaware should
be a significant drop).

Inputs are the 12 .eval log files enumerated in the plan's section 7.
Output is a markdown report at evals/agentic_misalignment/analysis/2026-04-20_gap_bootstrap.md
plus a JSON dump of raw numbers at evals/agentic_misalignment/analysis/2026-04-20_gap_bootstrap.json.

Usage:
    python evals/agentic_misalignment/scripts/gap_bootstrap.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from inspect_ai.log import read_eval_log

REPO_ROOT = Path("/data/jasmine_li/eval-awareness")
LOGS_DIR = REPO_ROOT / "evals" / "logs"
OUT_DIR = REPO_ROOT / "evals" / "agentic_misalignment" / "analysis"

FILES = {
    ("bare", "base"):        "2026-04-20T00-37-10+00-00_agentic-misalignment_FWMYrBwyVyj9DkgHFTZrEt.eval",
    ("bare", "coop_full"):   "2026-04-20T00-38-14+00-00_agentic-misalignment_DjuuhkLa2bodK3EWkh5tJ6.eval",
    ("bare", "muan"):        "2026-04-20T00-38-26+00-00_agentic-misalignment_dTnjKJVQQCb8MLgcnWCFiW.eval",
    ("bare", "coop_ablate"): "2026-04-20T00-38-26+00-00_agentic-misalignment_KGUt5XUCS66YJMfxZVrvir.eval",
    ("safety_eval", "base"):        "2026-04-20T04-39-50+00-00_agentic-misalignment_FVC3iTgU7Spja8SN9ELy9C.eval",
    ("safety_eval", "coop_full"):   "2026-04-20T03-39-15+00-00_agentic-misalignment_Ho32GhReurLFewu9cRRLKw.eval",
    ("safety_eval", "muan"):        "2026-04-20T04-46-16+00-00_agentic-misalignment_V7PyNypamKwCfT9fL4n3DR.eval",
    ("safety_eval", "coop_ablate"): "2026-04-20T03-44-13+00-00_agentic-misalignment_EFRrvtYTmr5ss8YF8DAeaZ.eval",
    ("af", "base"):        "2026-04-20T00-52-58+00-00_agentic-misalignment_k2nh978kdCtimXf6BZyxyL.eval",
    ("af", "coop_full"):   "2026-04-20T00-54-20+00-00_agentic-misalignment_NimYqqE8aEnae5ZYpoXraT.eval",
    ("af", "muan"):        "2026-04-20T00-54-20+00-00_agentic-misalignment_eqjee79FzRgxWbdLGbyh6h.eval",
    ("af", "coop_ablate"): "2026-04-20T00-54-10+00-00_agentic-misalignment_mbpkfftYE6YTjnU6TnLXLb.eval",
}

WARNINGS = ["bare", "safety_eval", "af"]
MODELS = ["base", "coop_full", "muan", "coop_ablate"]


@dataclass
class Cell:
    """Per (warning, model) cell. Stores per-sample harmful/aware arrays."""
    warning: str
    model: str
    harmful: np.ndarray  # 0/1 array
    aware: np.ndarray    # 0/1 array

    @property
    def n(self) -> int:
        return len(self.harmful)

    def h_given_aware(self) -> tuple[int, int]:
        """returns (k_harmful, n_aware)."""
        mask = self.aware == 1
        return int(self.harmful[mask].sum()), int(mask.sum())

    def h_given_unaware(self) -> tuple[int, int]:
        mask = self.aware == 0
        return int(self.harmful[mask].sum()), int(mask.sum())

    def gap(self) -> float:
        ka, na = self.h_given_aware()
        ku, nu = self.h_given_unaware()
        ra = ka / na if na else float("nan")
        ru = ku / nu if nu else float("nan")
        return ru - ra


def load_cells() -> dict[tuple[str, str], Cell]:
    cells: dict[tuple[str, str], Cell] = {}
    for (warning, model), fname in FILES.items():
        log = read_eval_log(str(LOGS_DIR / fname))
        harmful = []
        aware = []
        for s in log.samples:
            h_sc = s.scores.get("suite_harmfulness_scorer")
            a_sc = s.scores.get("suite_eval_awareness_scorer")
            if h_sc is None or a_sc is None:
                continue
            hv = h_sc.value.get("harmful") if isinstance(h_sc.value, dict) else None
            av = a_sc.value.get("aware") if isinstance(a_sc.value, dict) else None
            if hv is None or av is None:
                continue
            parse_h = h_sc.value.get("parse_failed", 0.0) if isinstance(h_sc.value, dict) else 0.0
            parse_a = a_sc.value.get("parse_failed", 0.0) if isinstance(a_sc.value, dict) else 0.0
            if parse_h == 1.0 or parse_a == 1.0:
                continue
            harmful.append(int(hv))
            aware.append(int(av))
        cells[(warning, model)] = Cell(
            warning=warning, model=model,
            harmful=np.asarray(harmful, dtype=np.int32),
            aware=np.asarray(aware, dtype=np.int32),
        )
    return cells


def bootstrap_gap_ci(cell: Cell, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (point_estimate, ci_lo, ci_hi) of the gap = H|Unaware - H|Aware in percentage points."""
    n = cell.n
    gap_point = cell.gap() * 100
    # Resample sample indices with replacement; recompute gap.
    boots = np.empty(n_boot, dtype=np.float64)
    harmful = cell.harmful
    aware = cell.aware
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        h = harmful[idx]
        a = aware[idx]
        na = a.sum()
        nu = n - na
        if na == 0 or nu == 0:
            boots[i] = np.nan
            continue
        ra = h[a == 1].sum() / na
        ru = h[a == 0].sum() / nu
        boots[i] = (ru - ra) * 100
    finite = boots[np.isfinite(boots)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return gap_point, float(lo), float(hi)


def bootstrap_rate_ci(k: int, n: int, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """95% percentile bootstrap for a single proportion, as percentage."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    x = np.zeros(n, dtype=np.int32)
    x[:k] = 1
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = x[idx].mean() * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return 100 * k / n, float(lo), float(hi)


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Pooled 2-proportion z-test. Returns (z, two-sided p)."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    # Two-sided p using standard normal CDF.
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    cells = load_cells()

    # --- 1.1 Bootstrap gap CIs ---
    gap_cis: dict[tuple[str, str], tuple[float, float, float]] = {}
    for key, cell in cells.items():
        gap_cis[key] = bootstrap_gap_ci(cell, args.n_boot, rng)

    # --- Also bootstrap H|Aware and H|Unaware rate CIs per cell ---
    rate_cis = {}  # key -> {aware: (p, lo, hi), unaware: (...)}
    for key, cell in cells.items():
        ka, na = cell.h_given_aware()
        ku, nu = cell.h_given_unaware()
        rate_cis[key] = {
            "h_given_aware": bootstrap_rate_ci(ka, na, args.n_boot, rng),
            "h_given_unaware": bootstrap_rate_ci(ku, nu, args.n_boot, rng),
            "aware_rate": bootstrap_rate_ci(na, cell.n, args.n_boot, rng),
            "counts": {"k_aware": ka, "n_aware": na, "k_unaware": ku, "n_unaware": nu, "n_total": cell.n},
        }

    # --- 1.2 / 1.4: two-prop z-tests ---
    # Decompose the shrink for coop_full/coop_ablate/muan vs base, on H|Aware and H|Unaware.
    zs: dict[str, dict] = {}
    for warning in WARNINGS:
        base = cells[(warning, "base")]
        for model in ["coop_full", "coop_ablate", "muan"]:
            other = cells[(warning, model)]
            k1, n1 = other.h_given_aware()
            k2, n2 = base.h_given_aware()
            z_a, p_a = two_prop_z(k1, n1, k2, n2)
            h_a = cohens_h(k1 / max(n1, 1), k2 / max(n2, 1)) if n1 and n2 else float("nan")
            k1u, n1u = other.h_given_unaware()
            k2u, n2u = base.h_given_unaware()
            z_u, p_u = two_prop_z(k1u, n1u, k2u, n2u)
            h_u = cohens_h(k1u / max(n1u, 1), k2u / max(n2u, 1)) if n1u and n2u else float("nan")
            zs[f"{warning}__{model}_vs_base"] = {
                "H_given_aware": {
                    "model_k": k1, "model_n": n1, "base_k": k2, "base_n": n2,
                    "z": z_a, "p_two_sided": p_a, "cohens_h": h_a,
                    "model_rate_pct": 100 * k1 / max(n1, 1), "base_rate_pct": 100 * k2 / max(n2, 1),
                },
                "H_given_unaware": {
                    "model_k": k1u, "model_n": n1u, "base_k": k2u, "base_n": n2u,
                    "z": z_u, "p_two_sided": p_u, "cohens_h": h_u,
                    "model_rate_pct": 100 * k1u / max(n1u, 1), "base_rate_pct": 100 * k2u / max(n2u, 1),
                },
            }

    # --- Write JSON ---
    out_json = OUT_DIR / "2026-04-20_gap_bootstrap.json"
    payload = {
        "seed": args.seed,
        "n_boot": args.n_boot,
        "cells": {
            f"{w}__{m}": {
                "n_total": cells[(w, m)].n,
                "gap_pct": gap_cis[(w, m)][0],
                "gap_ci95_lo": gap_cis[(w, m)][1],
                "gap_ci95_hi": gap_cis[(w, m)][2],
                "h_aware_pct": rate_cis[(w, m)]["h_given_aware"][0],
                "h_aware_ci95": rate_cis[(w, m)]["h_given_aware"][1:],
                "h_unaware_pct": rate_cis[(w, m)]["h_given_unaware"][0],
                "h_unaware_ci95": rate_cis[(w, m)]["h_given_unaware"][1:],
                "aware_rate_pct": rate_cis[(w, m)]["aware_rate"][0],
                "aware_rate_ci95": rate_cis[(w, m)]["aware_rate"][1:],
                **rate_cis[(w, m)]["counts"],
            }
            for w in WARNINGS for m in MODELS
        },
        "z_tests": zs,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_json}")

    # --- Write markdown ---
    md_lines: list[str] = []
    md_lines.append("# Eval-gaming gap bootstrap (Step 1 of 2026-04-20 deep-dive)")
    md_lines.append("")
    md_lines.append(f"Seed: {args.seed}. Bootstrap resamples: {args.n_boot}. All CIs are percentile 95%.")
    md_lines.append("")
    md_lines.append("Gap definition: H|Eval-Unaware - H|Eval-Aware (percentage points).")
    md_lines.append("")
    md_lines.append("## 1.1 Gap CIs per (warning, model)")
    md_lines.append("")
    md_lines.append("| Warning | Model | n | Gap (pp) | 95% CI |")
    md_lines.append("|---|---|---|---|---|")
    for w in WARNINGS:
        for m in MODELS:
            n = cells[(w, m)].n
            g, lo, hi = gap_cis[(w, m)]
            md_lines.append(f"| {w} | {m} | {n} | {g:.2f} | [{lo:.2f}, {hi:.2f}] |")
    md_lines.append("")

    md_lines.append("## 1.2 H|Aware and H|Unaware rates per cell (for direction of shrink)")
    md_lines.append("")
    md_lines.append("| Warning | Model | H\\|Aware (%) | CI | H\\|Unaware (%) | CI | k_aware/n_aware | k_unaware/n_unaware | aware-rate (%) |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for w in WARNINGS:
        for m in MODELS:
            rc = rate_cis[(w, m)]
            a_pt, a_lo, a_hi = rc["h_given_aware"]
            u_pt, u_lo, u_hi = rc["h_given_unaware"]
            ar_pt, ar_lo, ar_hi = rc["aware_rate"]
            c = rc["counts"]
            md_lines.append(
                f"| {w} | {m} | {a_pt:.2f} | [{a_lo:.2f},{a_hi:.2f}] | {u_pt:.2f} | [{u_lo:.2f},{u_hi:.2f}] | {c['k_aware']}/{c['n_aware']} | {c['k_unaware']}/{c['n_unaware']} | {ar_pt:.1f} |"
            )
    md_lines.append("")

    # Gap CI overlap check: is coop_full/coop_ablate gap non-overlapping with base under any warning?
    md_lines.append("## 1.1 gate — coop gap CI vs base gap CI")
    md_lines.append("")
    md_lines.append("Gate check: at least one warning where coop gap CI does not overlap base gap CI.")
    md_lines.append("")
    md_lines.append("| Warning | Model | Model CI | Base CI | Non-overlapping? |")
    md_lines.append("|---|---|---|---|---|")
    any_non_overlap = False
    for w in WARNINGS:
        b_g, b_lo, b_hi = gap_cis[(w, "base")]
        for m in ["coop_full", "coop_ablate"]:
            g, lo, hi = gap_cis[(w, m)]
            non_overlap = hi < b_lo or lo > b_hi
            any_non_overlap = any_non_overlap or non_overlap
            md_lines.append(f"| {w} | {m} | [{lo:.2f},{hi:.2f}] | [{b_lo:.2f},{b_hi:.2f}] | {'YES' if non_overlap else 'no'} |")
    md_lines.append("")
    md_lines.append(f"**Gate verdict**: {'PASS — at least one non-overlapping cell.' if any_non_overlap else 'FAIL — all coop CIs overlap base.'}")
    md_lines.append("")

    # 1.2 / 1.3 z-tests + effect size on H|Aware vs base
    md_lines.append("## 1.2 Driver of shrink: two-proportion z-tests + Cohen's h vs base")
    md_lines.append("")
    md_lines.append("Expected: coop models show a significant *elevation* of H|Aware vs base, and a *null* on H|Unaware. Muan expected to show a significant *drop* on H|Unaware (positive control for H8's 'generic SDF degrades misalignment').")
    md_lines.append("")
    md_lines.append("### H|Aware comparisons (model vs base)")
    md_lines.append("")
    md_lines.append("| Warning | Model | Model k/n | Base k/n | Model rate | Base rate | z | p | Cohen's h |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for w in WARNINGS:
        for m in ["coop_full", "coop_ablate", "muan"]:
            d = zs[f"{w}__{m}_vs_base"]["H_given_aware"]
            md_lines.append(
                f"| {w} | {m} | {d['model_k']}/{d['model_n']} | {d['base_k']}/{d['base_n']} | {d['model_rate_pct']:.2f}% | {d['base_rate_pct']:.2f}% | {d['z']:.2f} | {d['p_two_sided']:.4f} | {d['cohens_h']:.2f} |"
            )
    md_lines.append("")

    md_lines.append("### H|Unaware comparisons (model vs base)")
    md_lines.append("")
    md_lines.append("| Warning | Model | Model k/n | Base k/n | Model rate | Base rate | z | p | Cohen's h |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for w in WARNINGS:
        for m in ["coop_full", "coop_ablate", "muan"]:
            d = zs[f"{w}__{m}_vs_base"]["H_given_unaware"]
            md_lines.append(
                f"| {w} | {m} | {d['model_k']}/{d['model_n']} | {d['base_k']}/{d['base_n']} | {d['model_rate_pct']:.2f}% | {d['base_rate_pct']:.2f}% | {d['z']:.2f} | {d['p_two_sided']:.4f} | {d['cohens_h']:.2f} |"
            )
    md_lines.append("")

    md_lines.append("## 1.4 H8 (degraded misalignment) formal check")
    md_lines.append("")
    md_lines.append("**Rule-out condition**: coop SDFs' H|Unaware is statistically indistinguishable from base (p > 0.05) while muan's H|Unaware is significantly lower than base (p < 0.05), in ≥ 2 of 3 warning conditions.")
    md_lines.append("")

    coop_nulls = 0  # count of (warning, coop_model) where p > 0.05 on H|Unaware
    coop_cells = 0
    muan_drops = 0
    muan_cells = 0
    for w in WARNINGS:
        for m in ["coop_full", "coop_ablate"]:
            d = zs[f"{w}__{m}_vs_base"]["H_given_unaware"]
            coop_cells += 1
            if d["p_two_sided"] > 0.05:
                coop_nulls += 1
        d = zs[f"{w}__muan_vs_base"]["H_given_unaware"]
        muan_cells += 1
        # must be *lower* than base and significant
        if d["p_two_sided"] < 0.05 and d["model_rate_pct"] < d["base_rate_pct"]:
            muan_drops += 1

    md_lines.append(f"- coop SDFs with null H|Unaware vs base (p>0.05): {coop_nulls}/{coop_cells} cells.")
    md_lines.append(f"- muan with significant H|Unaware drop vs base (p<0.05, lower): {muan_drops}/{muan_cells} warnings.")
    h8_ruled_out = coop_nulls >= 4 and muan_drops >= 2  # 4/6 coop cells null, 2/3 muan drops
    md_lines.append(f"- **H8 ruled out?** {'YES' if h8_ruled_out else 'NO — qualify the writeup.'}")
    md_lines.append("")

    # Gate verdict
    md_lines.append("## Gate summary")
    md_lines.append("")
    md_lines.append("The plan's gate (Step 1) passes if:")
    md_lines.append("- 1.1 CIs non-overlapping for coop vs base in at least one warning (checked above).")
    md_lines.append("- 1.2 shows H|Aware is the driver of the shrink (not H|Unaware).")
    md_lines.append("")
    # Check driver: for coop, |effect on H|Aware| > |effect on H|Unaware|
    driver_ok = True
    driver_rows = []
    for w in WARNINGS:
        for m in ["coop_full", "coop_ablate"]:
            da = zs[f"{w}__{m}_vs_base"]["H_given_aware"]
            du = zs[f"{w}__{m}_vs_base"]["H_given_unaware"]
            # Direction check: H|Aware rises (model > base) and is the bigger mover in pp terms
            a_lift = da["model_rate_pct"] - da["base_rate_pct"]
            u_move = du["model_rate_pct"] - du["base_rate_pct"]
            aware_driver = a_lift > 0 and abs(a_lift) >= abs(u_move)
            driver_rows.append((w, m, a_lift, u_move, aware_driver))
            if not aware_driver:
                driver_ok = False
    md_lines.append("| Warning | Model | ΔH\\|Aware (pp) | ΔH\\|Unaware (pp) | Aware is the driver? |")
    md_lines.append("|---|---|---|---|---|")
    for w, m, a, u, ok in driver_rows:
        md_lines.append(f"| {w} | {m} | {a:+.2f} | {u:+.2f} | {'YES' if ok else 'no'} |")
    md_lines.append("")
    md_lines.append(f"**Gate verdict**: {'PASS — H|Aware lift is the dominant mover across coop cells.' if driver_ok else 'FAIL — H|Unaware drives in at least one cell; reframe plan.'}")
    md_lines.append("")

    out_md = OUT_DIR / "2026-04-20_gap_bootstrap.md"
    out_md.write_text("\n".join(md_lines))
    print(f"wrote {out_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    main(args)
