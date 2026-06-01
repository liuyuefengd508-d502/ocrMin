# CER / E2E-CER Protocol (Mongolian Archive OCR)

This document fixes the evaluation protocol for line-level recognition CER and end-to-end E2E-CER.
It is designed to be reproducible and stable across runs.

## 1. Canonical Text Normalization

Given a string `s`, compute `normalize(s)` as:

1. Convert to Python `str` (if input is `None`, treat as empty).
2. Replace any CRLF/CR with LF; then replace all whitespace runs (spaces, tabs, newlines) with a single ASCII space.
3. Strip leading/trailing spaces.
4. Remove Unicode "format/control" characters that are commonly invisible and unstable in OCR text:
   - All characters in Unicode general category `Cf` (Format), e.g., U+180E (Mongolian Vowel Separator),
     zero-width joiners, etc.
   - All characters in Unicode general category `Cc` (Control), except `\n` which is already normalized.

Rationale: the expert transcription may contain invisible separators or control marks; keeping them would
inflate edit distance without reflecting perceived readability.

## 2. CER (Character Error Rate)

For a predicted string `p` and ground-truth string `g`:

* `p' = normalize(p)`
* `g' = normalize(g)`
* `CER = edit_distance(p', g') / max(1, len(g'))`

Edit distance is Levenshtein distance at the Unicode codepoint level.

## 3. End-to-End Matching (Line Alignment)

End-to-end evaluation assumes we have:

* GT line list: from expert-filled `all_lines*.xlsx` (rows where `转写文本` is non-empty).
* Predicted line list: from system output (`pred_lines.jsonl`), each with:
  - `page_id`
  - `line_id` (optional)
  - `bbox = [x1, y1, x2, y2]`
  - `pred_text`

Matching is performed within each `page_id` using IoU:

* Compute IoU for every GT bbox vs predicted bbox.
* A GT line can match at most one predicted line, and vice versa.
* Greedy rule: repeatedly pick the remaining pair with highest IoU, if IoU >= 0.5.

Unmatched GT lines are counted as misses.
Unmatched predicted lines are counted as spurious predictions (optional to report separately).

## 4. E2E-CER

Let `M` be matched pairs, `U` be unmatched GT lines, and `N_gt` be number of GT lines.

* `E2E-CER = (sum_{(gt,pred) in M} edit_distance(normalize(pred_text), normalize(gt_text)) + sum_{gt in U} len(normalize(gt_text))) / max(1, sum_{gt} len(normalize(gt_text)))`

This definition penalizes missed detections/lines by the full GT length, which is consistent with
end-to-end readability.

## 5. Reporting

Always report:

* `CER_mean` over matched lines (recognition-only, ignoring misses).
* `E2E-CER` (end-to-end).
* Matching summary: `matched / gt_total / pred_total`.

For low-resource curves, report `mean ± std` over multiple seeds.

