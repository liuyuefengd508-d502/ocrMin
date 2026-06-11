"""Export OCR recognition-ready label files from split JSONL files.

Supported outputs per split:
- JSONL with explicit `image_path`, `raw_text`, `norm_text`
- TSV: `<image_path>\t<text>`
- PaddleOCR-style label.txt: same line format as TSV

This exporter walks a split root such as:
  docs_end2end/ocr_splits_round1/
and converts every `*.jsonl` split file it finds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.cer_eval import normalize_text  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _to_export_record(obj: dict[str, Any], data_root: Path, path_mode: str) -> dict[str, Any]:
    rel_img_path = str(obj.get("rel_img_path") or "").strip()
    raw_text = str(obj.get("transcription") or "")
    norm_text = normalize_text(raw_text)

    if path_mode == "absolute":
        image_path = str((data_root / rel_img_path).resolve())
    elif path_mode == "relative":
        image_path = rel_img_path
    else:
        raise ValueError(f"unsupported path_mode: {path_mode}")

    return {
        "page_id": str(obj.get("page_id") or ""),
        "line_id": str(obj.get("line_id") or ""),
        "reading_order": int(obj.get("reading_order") or 0),
        "image_path": image_path,
        "rel_img_path": rel_img_path,
        "raw_text": raw_text,
        "norm_text": norm_text,
        "bbox": obj.get("bbox") or [],
    }


def _write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_label_file(items: list[dict[str, Any]], path: Path, text_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            # Tabs/newlines in text would corrupt the format; normalize them conservatively.
            text = str(item[text_key]).replace("\t", " ").replace("\r", " ").replace("\n", " ")
            f.write(f"{item['image_path']}\t{text}\n")


def _convert_one_split(
    split_jsonl: Path,
    data_root: Path,
    out_dir: Path,
    path_mode: str,
) -> dict[str, Any]:
    src_items = _read_jsonl(split_jsonl)
    items = [_to_export_record(obj, data_root=data_root, path_mode=path_mode) for obj in src_items]

    stem = split_jsonl.stem
    # Unified export names.
    _write_jsonl(items, out_dir / f"{stem}.jsonl")
    _write_label_file(items, out_dir / f"{stem}_raw.tsv", text_key="raw_text")
    _write_label_file(items, out_dir / f"{stem}_norm.tsv", text_key="norm_text")
    _write_label_file(items, out_dir / f"{stem}_raw_label.txt", text_key="raw_text")
    _write_label_file(items, out_dir / f"{stem}_norm_label.txt", text_key="norm_text")

    return {
        "source_jsonl": str(split_jsonl),
        "out_dir": str(out_dir),
        "n": len(items),
        "files": [
            f"{stem}.jsonl",
            f"{stem}_raw.tsv",
            f"{stem}_norm.tsv",
            f"{stem}_raw_label.txt",
            f"{stem}_norm_label.txt",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--path-mode",
        type=str,
        default="relative",
        choices=("relative", "absolute"),
        help="Whether exported label files store image paths as relative or absolute.",
    )
    args = ap.parse_args()

    splits_root = args.splits_root.resolve()
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_files = sorted(p for p in splits_root.glob("**/*.jsonl") if p.is_file())
    reports = []
    for split_jsonl in split_files:
        rel_parent = split_jsonl.parent.relative_to(splits_root)
        target_dir = out_dir / rel_parent
        reports.append(
            _convert_one_split(
                split_jsonl=split_jsonl,
                data_root=data_root,
                out_dir=target_dir,
                path_mode=args.path_mode,
            )
        )

    summary = {
        "splits_root": str(splits_root),
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "path_mode": args.path_mode,
        "n_split_files": len(split_files),
        "reports": reports,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_split_files": len(split_files), "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

