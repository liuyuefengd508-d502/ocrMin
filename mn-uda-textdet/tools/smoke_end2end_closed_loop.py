"""Smoke test for the end-to-end OCR closed-loop evaluation.

This script uses the *existing* expert transcriptions inside a manifest JSONL
to build:
1) a GT subset jsonl (only transcribed lines)
2) a perfect prediction jsonl (pred_text == gt transcription, bbox identical)
3) a noisy prediction jsonl (small deterministic perturbation)

Then it runs the E2E evaluator and writes eval JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.cer_eval import normalize_text  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _perturb_text(s: str) -> str:
    """Deterministic, lightweight perturbation for a non-zero CER sanity check."""
    s = normalize_text(s)
    if not s:
        return s
    # Drop the last non-space character if possible.
    chars = list(s)
    for i in range(len(chars) - 1, -1, -1):
        if chars[i] != " ":
            del chars[i]
            break
    return "".join(chars)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noisy-rate", type=float, default=0.25, help="Fraction of lines to perturb.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_jsonl(args.manifest_jsonl.resolve())
    gt = [m for m in manifest if str(m.get("transcription") or "").strip() != ""]
    gt_sorted = sorted(gt, key=lambda x: (str(x.get("page_id") or ""), int(x.get("reading_order") or 0)))

    # GT subset
    gt_path = out_dir / "gt_transcribed.jsonl"
    _write_jsonl(gt_sorted, gt_path)

    # Perfect preds
    pred_perfect = []
    for g in gt_sorted:
        pred_perfect.append(
            {
                "page_id": g.get("page_id"),
                "bbox": g.get("bbox"),
                "pred_text": g.get("transcription", ""),
            }
        )
    pred_perfect_path = out_dir / "pred_perfect.jsonl"
    _write_jsonl(pred_perfect, pred_perfect_path)

    # Noisy preds
    pred_noisy = []
    for g in gt_sorted:
        txt = str(g.get("transcription") or "")
        if rng.random() < float(args.noisy_rate):
            txt = _perturb_text(txt)
        pred_noisy.append(
            {
                "page_id": g.get("page_id"),
                "bbox": g.get("bbox"),
                "pred_text": txt,
            }
        )
    pred_noisy_path = out_dir / "pred_noisy.jsonl"
    _write_jsonl(pred_noisy, pred_noisy_path)

    # Run evaluator via import to avoid subprocess issues.
    from tools.eval_end2end_from_jsonl import main as _eval_main  # noqa: E402

    def _run_eval(pred_path: Path, out_json: Path) -> None:
        argv = [
            "eval_end2end_from_jsonl.py",
            "--gt-jsonl",
            str(gt_path),
            "--pred-jsonl",
            str(pred_path),
            "--iou-thr",
            str(args.iou_thr),
            "--out-json",
            str(out_json),
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            _eval_main()
        finally:
            sys.argv = old_argv

    _run_eval(pred_perfect_path, out_dir / "eval_perfect.json")
    _run_eval(pred_noisy_path, out_dir / "eval_noisy.json")

    report = {
        "manifest_jsonl": str(args.manifest_jsonl.resolve()),
        "out_dir": str(out_dir),
        "n_gt_transcribed": len(gt_sorted),
        "gt_pages": sorted({str(x.get("page_id") or "") for x in gt_sorted}),
        "iou_thr": float(args.iou_thr),
        "seed": int(args.seed),
        "noisy_rate": float(args.noisy_rate),
        "artifacts": {
            "gt_transcribed": str(gt_path),
            "pred_perfect": str(pred_perfect_path),
            "pred_noisy": str(pred_noisy_path),
            "eval_perfect": str(out_dir / "eval_perfect.json"),
            "eval_noisy": str(out_dir / "eval_noisy.json"),
        },
    }
    (out_dir / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

