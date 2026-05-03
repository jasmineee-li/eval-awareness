"""
Aggregate safety eval results from inspect-ai .eval logs.

Walks results/safety_capdeg/<cond>/inspect_logs/<eval>/*.eval, extracts
scorer metrics from each log header, writes a single JSON keyed by
condition x eval suitable for plotting alongside the lighteval rounds.

Output: results/safety_capdeg/extended_safety_rescored_all.json
"""
import json
import subprocess
from pathlib import Path

REPO = Path("/data/jasmine_li/eval-awareness")
ROOT = REPO / "evals/capability_battery/results/safety_capdeg"
OUT = ROOT / "extended_safety_rescored_all.json"

CONDS = ["bare_mo", "muan_mo", "coop_full_mo"]
EVALS = ["simpleqa", "strong_reject", "xstest", "harmbench"]


def dump_header(eval_path: Path) -> dict:
    p = subprocess.run(
        ["inspect", "log", "dump", "--header-only", str(eval_path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(p.stdout)


def extract_metrics(header: dict) -> dict:
    """Return condensed metrics from an inspect log header."""
    results = header.get("results") or {}
    out = {
        "status": header.get("status"),
        "total_samples": results.get("total_samples"),
        "completed_samples": results.get("completed_samples"),
        "scores": [],
    }
    for s in results.get("scores", []) or []:
        scored = s.get("scored_samples", 0)
        unscored = s.get("unscored_samples", 0)
        metrics = {}
        for mname, m in (s.get("metrics") or {}).items():
            metrics[mname] = m.get("value") if isinstance(m, dict) else m
        out["scores"].append({
            "name": s.get("name"),
            "scored_samples": scored,
            "unscored_samples": unscored,
            "metrics": metrics,
        })
    return out


def main():
    aggregated = {}
    for cond in CONDS:
        aggregated[cond] = {}
        for ev in EVALS:
            d = ROOT / cond / "inspect_logs" / ev
            if not d.exists():
                aggregated[cond][ev] = {"_missing": True}
                continue
            evals = sorted(d.glob("*.eval"))
            if not evals:
                aggregated[cond][ev] = {"_missing": True}
                continue
            # take latest (sorted lexically, timestamps in filename)
            try:
                hdr = dump_header(evals[-1])
                aggregated[cond][ev] = extract_metrics(hdr)
            except Exception as e:
                aggregated[cond][ev] = {"_error": f"{type(e).__name__}: {e}"}

    OUT.write_text(json.dumps(aggregated, indent=2))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"Conditions: {list(aggregated.keys())}")
    for c in aggregated:
        for e in aggregated[c]:
            v = aggregated[c][e]
            if "_missing" in v:
                print(f"  {c}/{e}: MISSING")
            elif "_error" in v:
                print(f"  {c}/{e}: ERROR {v['_error'][:80]}")
            else:
                scores = v.get("scores") or []
                metric_str = ""
                if scores:
                    for s in scores:
                        m = s.get("metrics") or {}
                        metric_str += f" {s['name']}={','.join(f'{k}={v}' for k,v in m.items())}"
                print(f"  {c}/{e}: status={v['status']} n={v.get('total_samples')}/{v.get('completed_samples')}{metric_str}")


if __name__ == "__main__":
    main()
