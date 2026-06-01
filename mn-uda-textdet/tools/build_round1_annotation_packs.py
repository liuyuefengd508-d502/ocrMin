"""Build Round1 expert transcription packs from a line manifest.

Outputs two Excel files:
- random_round1.xlsx: stratified random sampling by page_id
- priority_round1.xlsx: stratified sampling by a simple, reproducible priority score

No pandas dependency; uses openpyxl and PIL only.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat
import openpyxl

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


EXCEL_OUT_COLUMNS: tuple[str, ...] = (
    "页面ID",
    "行ID",
    "阅读顺序",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "行图像相对路径",
    "转写文本",
    "难以辨认",
    "专家备注",
    "priority_score",
    "sampling_note",
)


@dataclass(frozen=True)
class Item:
    page_id: str
    line_id: str
    reading_order: int
    bbox: tuple[int, int, int, int]
    rel_img_path: str
    transcription: str
    illegible: str
    expert_note: str
    priority_score: float


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _as_int(v: Any, default: int = 0) -> int:
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _compute_priority_score(img_path: Path) -> float:
    """Simple, reproducible line difficulty heuristic.

    Higher score => higher priority.
    Uses:
    - low contrast (hard)
    - extreme darkness/brightness (hard)
    - very small height/width (hard)
    """
    try:
        im = Image.open(img_path).convert("L")
    except Exception:
        return 1e9
    stat = ImageStat.Stat(im)
    mean = float(stat.mean[0])
    std = float(stat.stddev[0])
    w, h = im.size
    # Contrast term (low std => harder)
    contrast = 1.0 / max(1e-6, std)
    # Exposure term (too dark or too bright)
    exposure = abs(mean - 127.5) / 127.5
    # Size penalty (small crops are harder)
    size_pen = (1.0 / max(10.0, float(h))) + (1.0 / max(50.0, float(w)))
    return 0.6 * contrast + 0.3 * exposure + 0.1 * size_pen


def _to_item(obj: dict[str, Any], data_root: Path) -> Item:
    bbox = obj.get("bbox") or [0, 0, 0, 0]
    rel = str(obj.get("rel_img_path") or "").strip()
    img_path = data_root / rel if rel else None
    score = float(obj.get("priority_score") or 0.0)
    if score == 0.0 and img_path is not None:
        score = _compute_priority_score(img_path)
    return Item(
        page_id=str(obj.get("page_id") or "").strip(),
        line_id=str(obj.get("line_id") or "").strip(),
        reading_order=_as_int(obj.get("reading_order"), 0),
        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
        rel_img_path=rel,
        transcription=str(obj.get("transcription") or ""),
        illegible=str(obj.get("illegible") or ""),
        expert_note=str(obj.get("expert_note") or ""),
        priority_score=float(score),
    )


def _write_excel(items: list[Item], out_path: Path, note: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "round1"
    ws.append(EXCEL_OUT_COLUMNS)
    for it in items:
        x1, y1, x2, y2 = it.bbox
        ws.append(
            (
                it.page_id,
                it.line_id,
                it.reading_order,
                x1,
                y1,
                x2,
                y2,
                it.rel_img_path,
                "",  # expert fills transcription
                "",  # expert fills illegible optionally
                "",  # expert note optional
                float(it.priority_score),
                note,
            )
        )
    wb.save(out_path)


def _stratified_sample(
    items_by_page: dict[str, list[Item]],
    total_target: int,
    min_per_page: int,
    rng: random.Random,
    mode: str,
) -> list[Item]:
    """mode in {'random','priority'}"""
    pages = sorted(items_by_page.keys())
    chosen: list[Item] = []

    # Stage 1: guarantee coverage.
    for pid in pages:
        pool = items_by_page[pid]
        if mode == "priority":
            pool = sorted(pool, key=lambda x: x.priority_score, reverse=True)
            picked = pool[: min(min_per_page, len(pool))]
        else:
            k = min(min_per_page, len(pool))
            picked = rng.sample(pool, k=k) if k > 0 else []
        chosen.extend(picked)

    # Stage 2: fill remaining from global pool without duplicates.
    remaining = max(0, total_target - len(chosen))
    if remaining == 0:
        return chosen[:total_target]

    chosen_ids = {c.line_id for c in chosen}
    rest = [it for pid in pages for it in items_by_page[pid] if it.line_id not in chosen_ids]
    if mode == "priority":
        rest = sorted(rest, key=lambda x: x.priority_score, reverse=True)
        chosen.extend(rest[:remaining])
    else:
        if remaining >= len(rest):
            chosen.extend(rest)
        else:
            chosen.extend(rng.sample(rest, k=remaining))
    return chosen[:total_target]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-jsonl", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target-lines", type=int, default=500)
    ap.add_argument("--min-per-page", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    objs = _read_jsonl(args.manifest_jsonl.resolve())
    # Prefer untranscribed lines for new annotation.
    cand = [o for o in objs if str(o.get("transcription") or "").strip() == ""]
    rng = random.Random(args.seed)

    items = [_to_item(o, data_root) for o in cand]
    items_by_page: dict[str, list[Item]] = {}
    for it in items:
        if not it.page_id or not it.line_id or not it.rel_img_path:
            continue
        items_by_page.setdefault(it.page_id, []).append(it)

    # Auto-adjust min_per_page: if target is smaller than page_count * min_per_page,
    # reduce to a feasible value. This avoids forcing full selection when data is small.
    per_page_counts = {k: len(v) for k, v in items_by_page.items()}
    total_cand = sum(per_page_counts.values())
    target = min(args.target_lines, total_cand)
    min_per_page = max(0, args.min_per_page)
    if len(items_by_page) > 0:
        min_per_page = min(min_per_page, max(1, target // len(items_by_page)))

    random_pack = _stratified_sample(items_by_page, target, min_per_page, rng, mode="random")
    priority_pack = _stratified_sample(items_by_page, target, min_per_page, rng, mode="priority")

    _write_excel(
        random_pack,
        out_dir / "random_round1.xlsx",
        note=f"stratified-random(min_per_page={min_per_page}, seed={args.seed})",
    )
    _write_excel(
        priority_pack,
        out_dir / "priority_round1.xlsx",
        note=f"stratified-priority(min_per_page={min_per_page}, seed={args.seed})",
    )

    def _score_stats(xs: list[Item]) -> dict[str, float]:
        vals = [x.priority_score for x in xs]
        if not vals:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(statistics.mean(vals)),
            "median": float(statistics.median(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    report = {
        "manifest_jsonl": str(args.manifest_jsonl.resolve()),
        "data_root": str(data_root),
        "candidate_untranscribed": len(items),
        "page_count": len(items_by_page),
        "target_lines": int(target),
        "min_per_page": int(min_per_page),
        "random_pack": {
            "n": len(random_pack),
            "score_stats": _score_stats(random_pack),
        },
        "priority_pack": {
            "n": len(priority_pack),
            "score_stats": _score_stats(priority_pack),
        },
    }
    (out_dir / "round1_pack_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_dir/'random_round1.xlsx'}")
    print(f"Wrote: {out_dir/'priority_round1.xlsx'}")


if __name__ == "__main__":
    main()
