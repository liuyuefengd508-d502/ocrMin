"""CER and simple end-to-end E2E-CER evaluation utilities.

No external deps beyond stdlib.
Normalization follows docs/cer_protocol.md.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


def normalize_text(s: object) -> str:
    if s is None:
        s = ""
    s = str(s)
    # Normalize whitespace runs -> single space.
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    out = []
    last_was_space = False
    for ch in s:
        if ch.isspace():
            if not last_was_space:
                out.append(" ")
                last_was_space = True
            continue
        last_was_space = False
        cat = unicodedata.category(ch)
        # Drop Cf/Cc (format/control) to stabilize OCR comparisons.
        if cat in ("Cf", "Cc"):
            continue
        out.append(ch)
    return "".join(out).strip()


def levenshtein(a: str, b: str) -> int:
    # Classic DP with O(min(n,m)) space.
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    # Now len(a) >= len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer(pred: object, gt: object) -> float:
    p = normalize_text(pred)
    g = normalize_text(gt)
    d = levenshtein(p, g)
    return float(d) / float(max(1, len(g)))


@dataclass(frozen=True)
class MatchSummary:
    matched: int
    gt_total: int
    pred_total: int


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def greedy_match_iou(
    gt: list[dict],
    preds: list[dict],
    iou_thr: float = 0.5,
) -> tuple[list[tuple[int, int, float]], MatchSummary]:
    """Return matches as list of (gt_idx, pred_idx, iou)."""
    pairs: list[tuple[int, int, float]] = []
    for i, g in enumerate(gt):
        gb = g.get("bbox")
        if not gb:
            continue
        for j, p in enumerate(preds):
            pb = p.get("bbox")
            if not pb:
                continue
            iou = bbox_iou(list(gb), list(pb))
            if iou >= iou_thr:
                pairs.append((i, j, iou))
    pairs.sort(key=lambda x: x[2], reverse=True)
    used_g, used_p = set(), set()
    matches: list[tuple[int, int, float]] = []
    for i, j, iou in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matches.append((i, j, iou))
    return matches, MatchSummary(matched=len(matches), gt_total=len(gt), pred_total=len(preds))


def e2e_cer_from_matches(
    gt: list[dict],
    preds: list[dict],
    matches: list[tuple[int, int, float]],
) -> dict:
    matched_gt = {i for i, _, _ in matches}
    total_gt_chars = 0
    total_penalty = 0
    total_edits = 0
    cer_list = []
    for i, g in enumerate(gt):
        gtxt = normalize_text(g.get("transcription", ""))
        total_gt_chars += len(gtxt)
        if i in matched_gt:
            # find pred
            j = next(j for ii, j, _ in matches if ii == i)
            ptxt = normalize_text(preds[j].get("pred_text", ""))
            d = levenshtein(ptxt, gtxt)
            total_edits += d
            cer_list.append(float(d) / float(max(1, len(gtxt))))
        else:
            # miss => full length penalty
            total_penalty += len(gtxt)
    denom = max(1, total_gt_chars)
    e2e = float(total_edits + total_penalty) / float(denom)
    cer_mean = float(sum(cer_list) / len(cer_list)) if cer_list else 1.0
    return {
        "cer_mean_matched": cer_mean,
        "e2e_cer": e2e,
        "gt_chars": int(total_gt_chars),
        "miss_penalty_chars": int(total_penalty),
        "matched_count": int(len(matches)),
        "gt_count": int(len(gt)),
        "pred_count": int(len(preds)),
    }

