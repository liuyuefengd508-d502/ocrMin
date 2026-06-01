"""Evaluate CER and E2E-CER from prediction JSONL against expert manifest JSONL.

Input:
- GT manifest JSONL: created by export_line_manifest_from_xlsx.py or merged manifests.
  Each line must contain: page_id, line_id, bbox, transcription
- Prediction JSONL: each line must contain: page_id, bbox, pred_text

This script filters GT to rows with non-empty transcription.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.cer_eval import greedy_match_iou, e2e_cer_from_matches


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-jsonl", type=Path, required=True)
    ap.add_argument("--pred-jsonl", type=Path, required=True)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    gt_all = _read_jsonl(args.gt_jsonl.resolve())
    pred_all = _read_jsonl(args.pred_jsonl.resolve())

    # Keep only transcribed GT lines.
    gt_all = [g for g in gt_all if str(g.get("transcription") or "").strip() != ""]

    gt_by_page: dict[str, list[dict]] = defaultdict(list)
    pred_by_page: dict[str, list[dict]] = defaultdict(list)
    for g in gt_all:
        gt_by_page[str(g.get("page_id") or "")].append(g)
    for p in pred_all:
        pred_by_page[str(p.get("page_id") or "")].append(p)

    page_reports = {}
    totals = {
        "cer_mean_matched_weighted_sum": 0.0,
        "cer_mean_matched_weight": 0,
        "e2e_edits_plus_penalty": 0,
        "gt_chars": 0,
        "matched": 0,
        "gt": 0,
        "pred": 0,
    }

    for pid, gt in gt_by_page.items():
        preds = pred_by_page.get(pid, [])
        matches, ms = greedy_match_iou(gt, preds, iou_thr=float(args.iou_thr))
        rep = e2e_cer_from_matches(gt, preds, matches)
        page_reports[pid] = {
            **rep,
            "iou_thr": float(args.iou_thr),
        }

        # Weighted aggregation.
        totals["gt_chars"] += rep["gt_chars"]
        totals["matched"] += rep["matched_count"]
        totals["gt"] += rep["gt_count"]
        totals["pred"] += rep["pred_count"]
        # For CER_mean_matched we weight by matched count.
        totals["cer_mean_matched_weighted_sum"] += rep["cer_mean_matched"] * max(1, rep["matched_count"])
        totals["cer_mean_matched_weight"] += max(1, rep["matched_count"])
        # E2E numerator = edits + penalty.
        totals["e2e_edits_plus_penalty"] += int(round(rep["e2e_cer"] * max(1, rep["gt_chars"])))

    cer_mean = (
        totals["cer_mean_matched_weighted_sum"] / float(max(1, totals["cer_mean_matched_weight"]))
    )
    e2e_cer = totals["e2e_edits_plus_penalty"] / float(max(1, totals["gt_chars"]))

    out = {
        "gt_jsonl": str(args.gt_jsonl.resolve()),
        "pred_jsonl": str(args.pred_jsonl.resolve()),
        "iou_thr": float(args.iou_thr),
        "overall": {
            "cer_mean_matched": float(cer_mean),
            "e2e_cer": float(e2e_cer),
            "matched": int(totals["matched"]),
            "gt": int(totals["gt"]),
            "pred": int(totals["pred"]),
            "gt_chars": int(totals["gt_chars"]),
        },
        "pages": page_reports,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["overall"], ensure_ascii=False, indent=2))
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
