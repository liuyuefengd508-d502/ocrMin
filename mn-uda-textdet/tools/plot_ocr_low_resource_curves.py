"""Plot CER / E2E-CER low-resource curves from aggregated JSON results.

Expected input is a list of records, each with:
  - budget_tag (e.g. "10%", "25%", "50%", "100%")
  - seed
  - metrics: {"cer_mean_matched": float, "e2e_cer": float}

This keeps deps minimal (matplotlib only).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _std(xs: list[float], m: float) -> float:
    if not xs:
        return 0.0
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", type=Path, required=True)
    ap.add_argument("--out-png", type=Path, required=True)
    ap.add_argument("--title", type=str, default="Low-Resource OCR Curves")
    args = ap.parse_args()

    data = json.loads(args.in_json.read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) and "records" in data else data

    by_budget = defaultdict(list)
    for r in records:
        by_budget[str(r.get("budget_tag", ""))].append(r)

    # Budget order preference.
    order = ["10%", "25%", "50%", "100%"]
    budgets = [b for b in order if b in by_budget] + [b for b in sorted(by_budget) if b not in order]

    cer_means, cer_stds = [], []
    e2e_means, e2e_stds = [], []
    for b in budgets:
        cer_vals = [float(x["metrics"]["cer_mean_matched"]) for x in by_budget[b]]
        e2e_vals = [float(x["metrics"]["e2e_cer"]) for x in by_budget[b]]
        m1 = _mean(cer_vals)
        m2 = _mean(e2e_vals)
        cer_means.append(m1)
        e2e_means.append(m2)
        cer_stds.append(_std(cer_vals, m1))
        e2e_stds.append(_std(e2e_vals, m2))

    xs = list(range(len(budgets)))
    plt.figure(figsize=(7.2, 4.2), dpi=160)
    plt.errorbar(xs, cer_means, yerr=cer_stds, marker="o", capsize=3, label="CER (matched)")
    plt.errorbar(xs, e2e_means, yerr=e2e_stds, marker="s", capsize=3, label="E2E-CER")
    plt.xticks(xs, budgets)
    plt.ylabel("Error rate (lower is better)")
    plt.title(args.title)
    plt.grid(True, alpha=0.25)
    plt.legend()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_png)
    print(f"Wrote: {args.out_png}")


if __name__ == "__main__":
    main()
