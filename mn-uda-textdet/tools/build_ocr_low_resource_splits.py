"""Build transcribed-only OCR splits and low-resource budgets from line manifest.

Outputs:
- fixed base split: train / val / test
- low-resource train subsets for multiple seeds and budgets

This is designed for line-level OCR recognition and end-to-end OCR evaluation.
Only rows with non-empty `transcription` are included.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.experiment_protocol import DEFAULT_SEEDS  # noqa: E402


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
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _sort_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda x: (
            str(x.get("page_id") or ""),
            int(x.get("reading_order") or 0),
            str(x.get("line_id") or ""),
        ),
    )


def _per_page_split(
    items: list[dict[str, Any]],
    rng: random.Random,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split within each page to preserve page coverage.

    Heuristic:
    - n >= 10: at least 1 val and 1 test
    - n >= 5: keep at least 1 holdout total
    - n < 5: still try to keep a test sample when possible
    """
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_page[str(it.get("page_id") or "")].append(it)

    train, val, test = [], [], []
    for page_id, page_items in sorted(by_page.items()):
        page_items = list(page_items)
        rng.shuffle(page_items)
        n = len(page_items)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        if n >= 10:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        elif n >= 5:
            n_test = max(1, n_test)
            n_val = max(0, n_val)
        elif n >= 3:
            n_test = max(1, n_test)
            n_val = 0
        else:
            n_test = 0
            n_val = 0

        # Keep at least one train sample.
        while n_test + n_val >= n:
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break

        test_items = page_items[:n_test]
        val_items = page_items[n_test:n_test + n_val]
        train_items = page_items[n_test + n_val:]
        test.extend(test_items)
        val.extend(val_items)
        train.extend(train_items)

    return _sort_records(train), _sort_records(val), _sort_records(test)


def _choose_budget_subset(
    train_items: list[dict[str, Any]],
    ratio: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    n = len(train_items)
    if ratio >= 0.999:
        return _sort_records(list(train_items))
    k = max(1, int(round(n * ratio)))
    picked = rng.sample(train_items, k=min(k, n))
    return _sort_records(picked)


def _count_by_page(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(x.get("page_id") or "") for x in items).items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-seed", type=int, default=42, help="Seed for fixed train/val/test base split.")
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.20)
    ap.add_argument(
        "--budget-ratios",
        type=str,
        default="0.10,0.25,0.50,1.00",
        help="Comma-separated train budget ratios.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_items = _read_jsonl(args.manifest_jsonl.resolve())
    transcribed = [x for x in all_items if str(x.get("transcription") or "").strip() != ""]
    transcribed = _sort_records(transcribed)

    base_rng = random.Random(int(args.base_seed))
    train_items, val_items, test_items = _per_page_split(
        transcribed,
        rng=base_rng,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )

    base_dir = out_dir / "base_split"
    _write_jsonl(train_items, base_dir / "train.jsonl")
    _write_jsonl(val_items, base_dir / "val.jsonl")
    _write_jsonl(test_items, base_dir / "test.jsonl")

    ratios = []
    for part in args.budget_ratios.split(","):
        part = part.strip()
        if not part:
            continue
        ratios.append(float(part))
    ratio_tags = {r: f"{int(round(r * 100)):02d}%" for r in ratios}

    budgets_summary: dict[str, Any] = {}
    for seed in DEFAULT_SEEDS:
        seed_rng = random.Random(int(seed))
        seed_dir = out_dir / "budgets" / f"seed{seed}"
        for ratio in ratios:
            ratio_tag = ratio_tags[ratio]
            subset = _choose_budget_subset(train_items, ratio=ratio, rng=seed_rng)
            _write_jsonl(subset, seed_dir / ratio_tag / "train.jsonl")
            budgets_summary.setdefault(ratio_tag, {"seeds": {}})
            budgets_summary[ratio_tag]["seeds"][str(seed)] = {
                "n": len(subset),
                "page_counts": _count_by_page(subset),
            }

    summary = {
        "manifest_jsonl": str(args.manifest_jsonl.resolve()),
        "n_all_rows": len(all_items),
        "n_transcribed": len(transcribed),
        "n_pages": len(_count_by_page(transcribed)),
        "base_seed": int(args.base_seed),
        "base_split": {
            "train": {
                "n": len(train_items),
                "page_counts": _count_by_page(train_items),
            },
            "val": {
                "n": len(val_items),
                "page_counts": _count_by_page(val_items),
            },
            "test": {
                "n": len(test_items),
                "page_counts": _count_by_page(test_items),
            },
        },
        "budget_ratios": {ratio_tags[r]: r for r in ratios},
        "budget_seeds": list(DEFAULT_SEEDS),
        "budgets": budgets_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["base_split"], ensure_ascii=False, indent=2))
    print(f"Wrote OCR splits to: {out_dir}")


if __name__ == "__main__":
    main()

