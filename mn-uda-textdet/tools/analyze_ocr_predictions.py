"""Analyze OCR prediction outputs from the minimal CRNN-CTC baseline.

Loads a checkpoint produced by train_ocr_baseline.py, runs inference on one TSV split,
and exports:
- predictions.jsonl: per-sample GT/pred/CER
- summary.json: aggregate diagnostics

Useful diagnostics:
- empty prediction rate
- average predicted length vs GT length
- most frequent predicted characters
- hardest samples by CER
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.cer_eval import cer, normalize_text  # noqa: E402
from tools.train_ocr_baseline import (  # noqa: E402
    CrnnCtc,
    OcrLineDataset,
    collate_ocr,
    load_charset,
    greedy_ctc_decode,
)


@torch.no_grad()
def predict_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx2char: list[str],
) -> list[dict[str, Any]]:
    model.eval()
    out = []
    for batch in loader:
        images = batch["images"].to(device)
        logits = model(images)
        log_probs = logits.log_softmax(dim=-1)
        preds = greedy_ctc_decode(log_probs, idx2char)
        for image_path, pred, gt in zip(batch["image_paths"], preds, batch["texts"]):
            gt_norm = normalize_text(gt)
            pred_norm = normalize_text(pred)
            out.append(
                {
                    "image_path": image_path,
                    "gt_text": gt,
                    "gt_norm": gt_norm,
                    "pred_text": pred,
                    "pred_norm": pred_norm,
                    "gt_len": len(gt_norm),
                    "pred_len": len(pred_norm),
                    "cer": cer(pred_norm, gt_norm),
                }
            )
    return out


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    cers = [float(r["cer"]) for r in records]
    gt_lens = [int(r["gt_len"]) for r in records]
    pred_lens = [int(r["pred_len"]) for r in records]
    empty_preds = sum(1 for r in records if int(r["pred_len"]) == 0)
    exact_matches = sum(1 for r in records if float(r["cer"]) == 0.0)

    pred_char_counter = Counter()
    for r in records:
        pred_char_counter.update(r["pred_norm"])

    hardest = sorted(records, key=lambda x: (float(x["cer"]), -int(x["gt_len"])), reverse=True)[:20]

    return {
        "n_samples": len(records),
        "cer_mean": _mean(cers),
        "cer_min": min(cers) if cers else 0.0,
        "cer_max": max(cers) if cers else 0.0,
        "gt_len_mean": _mean(gt_lens),
        "pred_len_mean": _mean(pred_lens),
        "empty_pred_count": empty_preds,
        "empty_pred_rate": float(empty_preds / max(1, len(records))),
        "exact_match_count": exact_matches,
        "exact_match_rate": float(exact_matches / max(1, len(records))),
        "pred_top20_chars": pred_char_counter.most_common(20),
        "hardest_samples": hardest,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--dict-path", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--img-height", type=int, default=32)
    ap.add_argument("--max-width", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    chars, char2idx = load_charset(args.dict_path.resolve())
    ds = OcrLineDataset(args.tsv.resolve(), args.data_root.resolve(), char2idx, args.img_height, args.max_width)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ocr, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=False)
    model = CrnnCtc(num_classes=len(chars) + 1).to(device)
    model.load_state_dict(ckpt["model"])

    records = predict_split(model, loader, device, chars)
    summary = build_summary(records)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

