"""Merge expert RoundX Excel back into the line manifest JSONL.

Example:
  python tools/merge_expert_round_xlsx.py \
    --base-manifest docs_end2end/line_manifest/line_manifest_round0.jsonl \
    --expert-xlsx  docs_end2end/round1_packs/random_round1_filled.xlsx \
    --round-id 1 \
    --out-manifest docs_end2end/line_manifest/line_manifest_round1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.ocr_line_manifest import merge_expert_into_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-manifest", type=Path, required=True)
    ap.add_argument("--expert-xlsx", type=Path, required=True)
    ap.add_argument("--sheet", type=str, default="all_lines")
    ap.add_argument("--round-id", type=int, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    args = ap.parse_args()

    report = merge_expert_into_manifest(
        base_manifest_jsonl=args.base_manifest.resolve(),
        expert_xlsx=args.expert_xlsx.resolve(),
        out_manifest_jsonl=args.out_manifest.resolve(),
        round_id=int(args.round_id),
        sheet=args.sheet,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

