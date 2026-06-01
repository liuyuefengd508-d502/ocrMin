"""Export expert rule-line Excel into round-aware line manifest (JSONL/CSV).

Example:
  python tools/export_line_manifest_from_xlsx.py \
    --xlsx /abs/path/forms/all_lines.xlsx \
    --data-root /abs/path/mongol_textline_test_b01 \
    --out-dir docs_end2end/line_manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.ocr_line_manifest import read_expert_xlsx, write_manifest_csv, write_manifest_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, required=True)
    ap.add_argument("--sheet", type=str, default="all_lines")
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory which contains the relative line image paths in the excel.",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    xlsx = args.xlsx.resolve()
    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_expert_xlsx(xlsx, sheet=args.sheet)

    # Basic integrity: ensure all referenced line images exist.
    missing = []
    for r in records:
        p = data_root / r.rel_img_path
        if not p.exists():
            missing.append(str(p))
    report = {
        "xlsx": str(xlsx),
        "sheet": args.sheet,
        "data_root": str(data_root),
        "n_records": len(records),
        "missing_line_images": missing[:50],
        "missing_line_images_count": len(missing),
    }
    (out_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_manifest_jsonl(records, out_dir / "line_manifest_round0.jsonl")
    write_manifest_csv(records, out_dir / "line_manifest_round0.csv")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_dir/'line_manifest_round0.jsonl'}")
    print(f"Wrote: {out_dir/'line_manifest_round0.csv'}")


if __name__ == "__main__":
    main()
