"""Build a qualitative casebook from a line manifest.

Creates per-page contact sheets using the provided `preview_boxed/<page>_boxed.jpg`
and optionally overlays predicted texts if a prediction JSONL is provided.

Dependencies: PIL only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Best-effort font selection; fallback to PIL default.
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode MS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _contact_sheet(
    page_id: str,
    boxed_page_img: Path,
    line_items: list[dict[str, Any]],
    data_root: Path,
    out_path: Path,
    ncols: int = 4,
    thumb_h: int = 96,
    pad: int = 10,
) -> None:
    thumbs: list[tuple[Image.Image, str]] = []
    for it in line_items:
        rel = str(it.get("rel_img_path") or "").strip()
        if not rel:
            continue
        img_path = data_root / rel
        if not img_path.exists():
            continue
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        if h <= 0:
            continue
        scale = thumb_h / float(h)
        im = im.resize((max(1, int(round(w * scale))), thumb_h))
        txt = str(it.get("transcription") or "").strip()
        thumbs.append((im, txt))

    if not thumbs:
        return

    n = len(thumbs)
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))

    font = _load_font(14)
    title_font = _load_font(20)

    # Estimate tile size based on max thumb width.
    max_w = max(im.size[0] for im, _ in thumbs)
    tile_w = max_w + pad * 2
    tile_h = thumb_h + 50

    header_h = 60
    sheet_w = tile_w * ncols
    sheet_h = header_h + tile_h * nrows + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    # Header with boxed page preview.
    try:
        page_prev = Image.open(boxed_page_img).convert("RGB")
        prev_w = min(sheet_w // 2, page_prev.size[0])
        scale = prev_w / float(page_prev.size[0])
        page_prev = page_prev.resize((prev_w, max(1, int(round(page_prev.size[1] * scale)))))
        sheet.paste(page_prev, (pad, pad))
        header_text_x = pad + prev_w + pad
    except Exception:
        header_text_x = pad
    draw.text((header_text_x, pad), f"Page {page_id} | n_lines={n}", fill=(0, 0, 0), font=title_font)

    # Grid.
    y0 = header_h
    for idx, (im, txt) in enumerate(thumbs):
        r = idx // ncols
        c = idx % ncols
        x = c * tile_w + pad
        y = y0 + r * tile_h + pad
        sheet.paste(im, (x, y))
        # Write transcription (truncate).
        s = txt.replace("\n", " ").strip()
        if len(s) > 60:
            s = s[:57] + "..."
        draw.text((x, y + thumb_h + 6), s, fill=(0, 0, 0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-jsonl", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--page-preview-dir", type=Path, required=True, help=".../page_images/preview_boxed")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only-transcribed", action="store_true", help="Include only lines with transcription.")
    args = ap.parse_args()

    items = _read_jsonl(args.manifest_jsonl.resolve())
    if args.only_transcribed:
        items = [it for it in items if str(it.get("transcription") or "").strip() != ""]

    by_page: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        pid = str(it.get("page_id") or "").strip()
        if not pid:
            continue
        by_page.setdefault(pid, []).append(it)
    for pid in by_page:
        by_page[pid].sort(key=lambda x: (int(x.get("reading_order") or 0), str(x.get("line_id") or "")))

    data_root = args.data_root.resolve()
    prev_dir = args.page_preview_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid, lst in sorted(by_page.items(), key=lambda kv: kv[0]):
        prev = prev_dir / f"{pid}_boxed.jpg"
        out = out_dir / f"casebook_{pid}.jpg"
        _contact_sheet(pid, prev, lst, data_root, out)
    print(f"Wrote casebooks to: {out_dir}")


if __name__ == "__main__":
    main()

