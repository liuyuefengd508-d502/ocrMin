"""Report rule-based line crop quality statistics for OCR closed-loop.

Outputs:
- line_quality.jsonl: per-line metrics
- summary.json: overall + per-page summary
- topk_hardest.csv/json: actionable list for next-round annotation prioritization

No pandas/opencv dependency. Uses PIL + numpy only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _laplacian_var(gray: np.ndarray) -> float:
    """Variance of 4-neighborhood Laplacian as a blur indicator (higher => sharper)."""
    if gray.ndim != 2:
        raise ValueError("expected 2D gray image")
    g = gray.astype(np.float32)
    # Pad by edge values to avoid shrinking.
    p = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    c = p[1:-1, 1:-1]
    up = p[0:-2, 1:-1]
    dn = p[2:, 1:-1]
    lf = p[1:-1, 0:-2]
    rt = p[1:-1, 2:]
    lap = (up + dn + lf + rt) - 4.0 * c
    return float(lap.var())


def _gradient_energy(gray: np.ndarray) -> float:
    """Mean squared gradient (Tenengrad-like) as another sharpness proxy."""
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    return float(np.mean(gx * gx + gy * gy))


def _compute_metrics(img_path: Path) -> dict[str, float]:
    im = Image.open(img_path).convert("L")
    arr = np.asarray(im)
    h, w = arr.shape[:2]
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std()) if arr.size else 0.0
    lap_var = _laplacian_var(arr) if min(h, w) >= 3 else 0.0
    grad_e = _gradient_energy(arr) if min(h, w) >= 3 else 0.0
    return {
        "width": float(w),
        "height": float(h),
        "mean": mean,
        "std": std,
        "laplacian_var": lap_var,
        "grad_energy": grad_e,
    }


def _difficulty_score(m: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Higher => more difficult, more valuable to annotate / more likely to fail."""
    mean = m["mean"]
    std = m["std"]
    w = m["width"]
    h = m["height"]
    lap = m["laplacian_var"]
    grad = m["grad_energy"]

    contrast = 1.0 / max(1e-6, std)  # low-contrast -> high
    exposure = abs(mean - 127.5) / 127.5  # extreme lighting -> high
    size = (1.0 / max(10.0, h)) + (1.0 / max(50.0, w))  # tiny crops -> high
    blur = 1.0 / max(1e-6, math.sqrt(lap + 1e-6))  # blurry -> high
    edge = 1.0 / max(1e-6, math.sqrt(grad + 1e-6))  # low edges -> high

    # Conservative weights; we mainly want a stable ranking.
    score = 0.30 * contrast + 0.20 * exposure + 0.15 * size + 0.20 * blur + 0.15 * edge
    return float(score), {
        "term_contrast": float(contrast),
        "term_exposure": float(exposure),
        "term_size": float(size),
        "term_blur": float(blur),
        "term_edge": float(edge),
    }


def _reason_tags(m: dict[str, float], score_terms: dict[str, float]) -> list[str]:
    tags = []
    if m["std"] < 25:
        tags.append("low_contrast")
    if m["mean"] < 60:
        tags.append("too_dark")
    if m["mean"] > 195:
        tags.append("too_bright")
    if m["height"] < 18:
        tags.append("very_short_height")
    if m["width"] < 120:
        tags.append("very_narrow_width")
    if m["laplacian_var"] < 30:
        tags.append("blurry")
    if m["grad_energy"] < 120:
        tags.append("low_edges")
    # Add dominant term for explanation.
    dom = max(score_terms.items(), key=lambda kv: kv[1])[0].replace("term_", "")
    tags.append(f"dom_{dom}")
    return tags


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _std(xs: list[float], m: float) -> float:
    if not xs:
        return 0.0
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-jsonl", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    manifest = _read_jsonl(args.manifest_jsonl.resolve())
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_line = []
    missing_imgs = []
    for obj in manifest:
        pid = str(obj.get("page_id") or "").strip()
        lid = str(obj.get("line_id") or "").strip()
        rel = str(obj.get("rel_img_path") or "").strip()
        if not pid or not lid or not rel:
            continue
        img_path = data_root / rel
        if not img_path.exists():
            missing_imgs.append(str(img_path))
            continue
        m = _compute_metrics(img_path)
        score, terms = _difficulty_score(m)
        tags = _reason_tags(m, terms)
        per_line.append(
            {
                "page_id": pid,
                "line_id": lid,
                "reading_order": int(_safe_float(obj.get("reading_order"), 0)),
                "rel_img_path": rel,
                "has_transcription": bool(str(obj.get("transcription") or "").strip()),
                "metrics": m,
                "difficulty_score": float(score),
                "difficulty_terms": terms,
                "reason_tags": tags,
            }
        )

    # Save per-line jsonl
    line_quality_path = out_dir / "line_quality.jsonl"
    with line_quality_path.open("w", encoding="utf-8") as f:
        for it in sorted(per_line, key=lambda x: (x["page_id"], x["reading_order"], x["line_id"])):
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Aggregate summary
    by_page = defaultdict(list)
    tag_counter = Counter()
    for it in per_line:
        by_page[it["page_id"]].append(it)
        tag_counter.update(it["reason_tags"])

    def _summ(items: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [float(x["difficulty_score"]) for x in items]
        stds = [float(x["metrics"]["std"]) for x in items]
        means = [float(x["metrics"]["mean"]) for x in items]
        hs = [float(x["metrics"]["height"]) for x in items]
        laps = [float(x["metrics"]["laplacian_var"]) for x in items]
        return {
            "n": len(items),
            "difficulty": {
                "mean": _mean(scores),
                "std": _std(scores, _mean(scores)),
                "min": min(scores) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
            },
            "img_stats": {
                "std_mean": _mean(stds),
                "mean_mean": _mean(means),
                "height_mean": _mean(hs),
                "laplacian_var_mean": _mean(laps),
            },
            "transcribed": sum(1 for x in items if x["has_transcription"]),
        }

    summary = {
        "manifest_jsonl": str(args.manifest_jsonl.resolve()),
        "data_root": str(data_root),
        "n_manifest_lines": len(manifest),
        "n_quality_lines": len(per_line),
        "missing_images_count": len(missing_imgs),
        "tag_top20": tag_counter.most_common(20),
        "overall": _summ(per_line),
        "per_page": {pid: _summ(items) for pid, items in sorted(by_page.items())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Top-K hardest
    topk = max(1, int(args.topk))
    hardest = sorted(per_line, key=lambda x: float(x["difficulty_score"]), reverse=True)[:topk]
    (out_dir / "topk_hardest.json").write_text(json.dumps(hardest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "topk_hardest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "page_id",
                "line_id",
                "reading_order",
                "difficulty_score",
                "reason_tags",
                "rel_img_path",
                "has_transcription",
                "mean",
                "std",
                "height",
                "width",
                "laplacian_var",
                "grad_energy",
            ],
        )
        w.writeheader()
        for it in hardest:
            m = it["metrics"]
            w.writerow(
                {
                    "page_id": it["page_id"],
                    "line_id": it["line_id"],
                    "reading_order": it["reading_order"],
                    "difficulty_score": f"{it['difficulty_score']:.6f}",
                    "reason_tags": "|".join(it["reason_tags"]),
                    "rel_img_path": it["rel_img_path"],
                    "has_transcription": int(bool(it["has_transcription"])),
                    "mean": f"{m['mean']:.3f}",
                    "std": f"{m['std']:.3f}",
                    "height": int(m["height"]),
                    "width": int(m["width"]),
                    "laplacian_var": f"{m['laplacian_var']:.3f}",
                    "grad_energy": f"{m['grad_energy']:.3f}",
                }
            )

    print(f"Wrote: {line_quality_path}")
    print(f"Wrote: {out_dir/'summary.json'}")
    print(f"Wrote: {out_dir/'topk_hardest.csv'}")


if __name__ == "__main__":
    main()

