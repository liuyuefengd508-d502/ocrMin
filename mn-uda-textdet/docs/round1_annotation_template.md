# Round1 Expert Transcription Template

This document describes how to fill the Round1 Excel files:

* `random_round1.xlsx`
* `priority_round1.xlsx`

## Columns

Do not rename columns, do not delete columns, do not change the sheet name.

Required to fill:
- `转写文本`: expert transcription for the line image.

Optional to fill:
- `难以辨认`: mark with `1` (or any non-empty value) if the line is not readable.
- `专家备注`: free text notes, e.g. script mixing, severe blur, truncation.

Read-only (do not edit):
- `页面ID`, `行ID`, `阅读顺序`
- `bbox_*`
- `行图像相对路径`
- `priority_score`, `sampling_note`

## Notes

- If a line contains mixed scripts (e.g., Mongolian + Chinese + symbols), transcribe as-is.
- Keep spacing as perceived, but do not worry about multiple spaces; the evaluation protocol normalizes whitespace.

