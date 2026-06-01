"""Aggregate per-seed OCR eval JSONs into mean/std tables for plotting/paper.

Input: a directory containing files named like:
  <budget_tag>/seed<seed>/eval.json
or any structure; we simply glob for `eval.json` under --runs-root.

Each eval.json is expected to contain:
  overall: {cer_mean_matched, e2e_cer, matched, gt, pred, gt_chars}
and optionally: budget_tag, seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    runs_root = args.runs_root.resolve()
    eval_paths = sorted(runs_root.glob("**/eval.json"))
    records = []
    for p in eval_paths:
        obj = json.loads(p.read_text(encoding="utf-8"))
        overall = obj.get("overall", {})
        # Infer budget/seed from path if not present.
        parts = p.parts
        budget_tag = obj.get("budget_tag")
        seed = obj.get("seed")
        if budget_tag is None:
            for part in parts:
                if part in ("10%", "25%", "50%", "100%"):
                    budget_tag = part
                    break
        if seed is None:
            for part in parts:
                if part.startswith("seed"):
                    try:
                        seed = int(part.replace("seed", ""))
                    except Exception:
                        pass
        records.append(
            {
                "path": str(p),
                "budget_tag": str(budget_tag or ""),
                "seed": int(seed or 0),
                "metrics": {
                    "cer_mean_matched": float(overall.get("cer_mean_matched", 1.0)),
                    "e2e_cer": float(overall.get("e2e_cer", 1.0)),
                },
                "counts": {
                    "matched": int(overall.get("matched", 0)),
                    "gt": int(overall.get("gt", 0)),
                    "pred": int(overall.get("pred", 0)),
                    "gt_chars": int(overall.get("gt_chars", 0)),
                },
            }
        )

    out = {
        "runs_root": str(runs_root),
        "n_eval_files": len(eval_paths),
        "records": records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Found eval.json files: {len(eval_paths)}")
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
