"""Export OCR character dictionaries and charset statistics from recognition label files.

Primary outputs:
- dict_norm.txt: one normalized character per line
- dict_raw.txt: one raw character per line
- charset_summary.json
- charset_freq_norm.csv / charset_freq_raw.csv

The default source is the exported recognition format root:
  docs_end2end/ocr_rec_formats_round1/
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable


def _iter_texts(tsv_paths: Iterable[Path]) -> list[str]:
    texts: list[str] = []
    for path in tsv_paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                texts.append(parts[1])
    return texts


def _char_block(ch: str) -> str:
    cp = ord(ch)
    if ch == " ":
        return "SPACE"
    if 0x1800 <= cp <= 0x18AF:
        return "MONGOLIAN_UNICODE"
    if 0x4E00 <= cp <= 0x9FFF:
        return "CJK_UNIFIED"
    if 0xE000 <= cp <= 0xF8FF:
        return "PRIVATE_USE_AREA"
    if 0xF0000 <= cp <= 0xFFFFD or 0x100000 <= cp <= 0x10FFFD:
        return "PRIVATE_USE_AREA_SUPP"
    if 0x0000 <= cp <= 0x007F:
        return "ASCII"
    return "OTHER"


def _char_info(ch: str) -> dict:
    name = unicodedata.name(ch, "<unnamed>")
    return {
        "char": ch,
        "codepoint": f"U+{ord(ch):04X}",
        "name": name,
        "block": _char_block(ch),
        "category": unicodedata.category(ch),
    }


def _write_dict(chars: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ch in chars:
            f.write(ch + "\n")


def _write_freq(counter: Counter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["rank", "char", "count", "codepoint", "name", "block", "category"],
        )
        w.writeheader()
        for rank, (ch, cnt) in enumerate(counter.most_common(), start=1):
            meta = _char_info(ch)
            w.writerow(
                {
                    "rank": rank,
                    "char": meta["char"],
                    "count": cnt,
                    "codepoint": meta["codepoint"],
                    "name": meta["name"],
                    "block": meta["block"],
                    "category": meta["category"],
                }
            )


def _build_summary(counter: Counter, source_files: list[str]) -> dict:
    unique_chars = sorted(counter.keys())
    block_counter = Counter(_char_block(ch) for ch in unique_chars)
    total_counter = Counter()
    for ch, cnt in counter.items():
        total_counter[_char_block(ch)] += cnt
    sample_chars = [
        {
            **_char_info(ch),
            "count": int(counter[ch]),
        }
        for ch in unique_chars[:80]
    ]
    return {
        "source_files": source_files,
        "n_unique_chars": len(unique_chars),
        "n_total_chars": int(sum(counter.values())),
        "unique_block_counts": dict(block_counter),
        "token_block_counts": dict(total_counter),
        "top20": [
            {
                **_char_info(ch),
                "count": int(cnt),
            }
            for ch, cnt in counter.most_common(20)
        ],
        "sample_chars": sample_chars,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    formats_root = args.formats_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(formats_root.glob("**/*_raw.tsv"))
    norm_files = sorted(formats_root.glob("**/*_norm.tsv"))

    raw_texts = _iter_texts(raw_files)
    norm_texts = _iter_texts(norm_files)

    raw_counter = Counter("".join(raw_texts))
    norm_counter = Counter("".join(norm_texts))

    raw_chars = sorted(raw_counter.keys())
    norm_chars = sorted(norm_counter.keys())

    _write_dict(raw_chars, out_dir / "dict_raw.txt")
    _write_dict(norm_chars, out_dir / "dict_norm.txt")
    _write_freq(raw_counter, out_dir / "charset_freq_raw.csv")
    _write_freq(norm_counter, out_dir / "charset_freq_norm.csv")

    summary = {
        "formats_root": str(formats_root),
        "raw": _build_summary(raw_counter, [str(p) for p in raw_files]),
        "norm": _build_summary(norm_counter, [str(p) for p in norm_files]),
    }

    # Simple warning flags useful before recognition training.
    summary["warnings"] = {
        "norm_has_private_use_area": bool(
            summary["norm"]["unique_block_counts"].get("PRIVATE_USE_AREA", 0)
            or summary["norm"]["unique_block_counts"].get("PRIVATE_USE_AREA_SUPP", 0)
        ),
        "raw_has_private_use_area": bool(
            summary["raw"]["unique_block_counts"].get("PRIVATE_USE_AREA", 0)
            or summary["raw"]["unique_block_counts"].get("PRIVATE_USE_AREA_SUPP", 0)
        ),
        "norm_unique_chars": summary["norm"]["n_unique_chars"],
        "raw_unique_chars": summary["raw"]["n_unique_chars"],
    }

    (out_dir / "charset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "raw_unique_chars": summary["raw"]["n_unique_chars"],
                "norm_unique_chars": summary["norm"]["n_unique_chars"],
                "warnings": summary["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

