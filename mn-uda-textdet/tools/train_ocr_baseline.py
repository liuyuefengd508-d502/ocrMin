"""Minimal OCR recognition baseline for Mongolian archive line images.

Model:
- CNN feature extractor
- BiLSTM sequence encoder
- Linear + CTC

Inputs:
- train/val/test TSV exported by export_ocr_recognition_formats.py
- charset dict exported by export_ocr_charset.py

This is intentionally lightweight and self-contained to establish a first
recognition baseline and CER budget curve.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.cer_eval import cer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sanitize_for_ckpt(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_ckpt(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_ckpt(v) for v in obj]
    return obj


def load_charset(dict_path: Path) -> tuple[list[str], dict[str, int]]:
    chars = [line.rstrip("\n") for line in dict_path.read_text(encoding="utf-8").splitlines()]
    # CTC blank = 0, chars start at 1
    char2idx = {ch: i + 1 for i, ch in enumerate(chars)}
    return chars, char2idx


def read_tsv(tsv_path: Path) -> list[tuple[str, str]]:
    items = []
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            img_path, text = line.split("\t", 1)
        except ValueError:
            continue
        items.append((img_path, text))
    return items


class OcrLineDataset(Dataset):
    def __init__(
        self,
        tsv_path: Path,
        data_root: Path,
        char2idx: dict[str, int],
        img_height: int = 32,
        max_width: int = 512,
    ) -> None:
        self.tsv_path = tsv_path
        self.data_root = data_root
        self.char2idx = char2idx
        self.img_height = img_height
        self.max_width = max_width
        self.items = read_tsv(tsv_path)

    def __len__(self) -> int:
        return len(self.items)

    def _encode_text(self, text: str) -> list[int]:
        encoded = []
        for ch in text:
            if ch not in self.char2idx:
                continue
            encoded.append(self.char2idx[ch])
        return encoded

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rel_img, text = self.items[idx]
        img_path = Path(rel_img)
        if not img_path.is_absolute():
            img_path = self.data_root / rel_img

        img = Image.open(img_path).convert("L")
        w, h = img.size
        scale = self.img_height / float(max(1, h))
        new_w = max(8, min(self.max_width, int(round(w * scale))))
        img = img.resize((new_w, self.img_height), resample=Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        target = self._encode_text(text)
        return {
            "image": tensor,
            "width": new_w,
            "target": target,
            "text": text,
            "image_path": str(img_path),
        }


def collate_ocr(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_w = max(x["image"].shape[-1] for x in batch)
    imgs = []
    widths = []
    texts = []
    img_paths = []
    targets_flat = []
    target_lengths = []
    for item in batch:
        img = item["image"]
        pad_w = max_w - img.shape[-1]
        if pad_w > 0:
            pad = torch.zeros((1, img.shape[1], pad_w), dtype=img.dtype)
            img = torch.cat([img, pad], dim=-1)
        imgs.append(img)
        widths.append(item["width"])
        texts.append(item["text"])
        img_paths.append(item["image_path"])
        targets_flat.extend(item["target"])
        target_lengths.append(len(item["target"]))

    return {
        "images": torch.stack(imgs, dim=0),
        "widths": torch.tensor(widths, dtype=torch.long),
        "targets": torch.tensor(targets_flat, dtype=torch.long) if targets_flat else torch.zeros((0,), dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
        "image_paths": img_paths,
    }


class CrnnCtc(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # 32xW -> 16xW/2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # 8xW/4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 4xW/4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 2xW/4
            nn.Conv2d(128, 128, kernel_size=2, padding=0),  # 1x(W/4 - 1)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)  # [B,C,1,T]
        feat = feat.squeeze(2)  # [B,C,T]
        feat = feat.permute(2, 0, 1)  # [T,B,C]
        seq, _ = self.rnn(feat)
        logits = self.fc(seq)  # [T,B,C]
        return logits


def greedy_ctc_decode(log_probs: torch.Tensor, idx2char: list[str]) -> list[str]:
    # log_probs: [T, B, C]
    pred = log_probs.argmax(dim=-1).cpu().numpy()  # [T,B]
    outputs = []
    blank = 0
    for b in range(pred.shape[1]):
        seq = pred[:, b].tolist()
        decoded = []
        prev = None
        for token in seq:
            if token != blank and token != prev:
                decoded.append(idx2char[token - 1])
            prev = token
        outputs.append("".join(decoded))
    return outputs


@dataclass
class EvalResult:
    cer_mean: float
    n_samples: int


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, idx2char: list[str]) -> EvalResult:
    model.eval()
    cers = []
    for batch in loader:
        images = batch["images"].to(device)
        logits = model(images)
        log_probs = logits.log_softmax(dim=-1)
        preds = greedy_ctc_decode(log_probs, idx2char)
        for pred, gt in zip(preds, batch["texts"]):
            cers.append(cer(pred, gt))
    return EvalResult(cer_mean=float(sum(cers) / max(1, len(cers))), n_samples=len(cers))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        logits = model(images)
        log_probs = logits.log_softmax(dim=-1)
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=device,
        )
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tsv", type=Path, required=True)
    ap.add_argument("--val-tsv", type=Path, required=True)
    ap.add_argument("--test-tsv", type=Path, required=True)
    ap.add_argument("--dict-path", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img-height", type=int, default=32)
    ap.add_argument("--max-width", type=int, default=512)
    args = ap.parse_args()

    set_seed(int(args.seed))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    idx2char, char2idx = None, None
    chars, char2idx = load_charset(args.dict_path.resolve())
    idx2char = chars

    train_ds = OcrLineDataset(args.train_tsv.resolve(), args.data_root.resolve(), char2idx, args.img_height, args.max_width)
    val_ds = OcrLineDataset(args.val_tsv.resolve(), args.data_root.resolve(), char2idx, args.img_height, args.max_width)
    test_ds = OcrLineDataset(args.test_tsv.resolve(), args.data_root.resolve(), char2idx, args.img_height, args.max_width)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_ocr, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ocr, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_ocr, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrnnCtc(num_classes=len(chars) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    history = []
    best = {"epoch": 0, "val_cer": 1e9}
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_res = evaluate(model, val_loader, device, idx2char)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_cer": val_res.cer_mean,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_res.cer_mean < best["val_cer"]:
            best = {"epoch": epoch, "val_cer": val_res.cer_mean}
            torch.save(
                {
                    "model": model.state_dict(),
                    "chars": chars,
                    "args": _sanitize_for_ckpt(vars(args)),
                },
                best_path,
            )

    # Final test using best model.
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_res = evaluate(model, test_loader, device, idx2char)

    report = {
        "args": _sanitize_for_ckpt(vars(args)),
        "device": str(device),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "n_chars": len(chars),
        "best": best,
        "test": asdict(test_res),
        "history": history,
        "best_checkpoint": str(best_path),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "test": asdict(test_res), "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
