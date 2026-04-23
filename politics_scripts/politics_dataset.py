"""
PoliticsDataset: image-text pairs for political inclination and topic classification.

CSV columns (semicolon-separated):
  local_path | split | label | topic | image_description | origin_webpage | web_source | content_text

Returns per sample:
  image                  : Tensor (3, 224, 224)
  inclination_label_idx  : int  (0=left, 1=right)
  topic_label_idx        : int  (0..19, sorted alphabetical)
  headline               : str  (image_description column)
  text                   : str  (content_text column)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

INCLINATION_CLASSES = ["left", "right"]

TOPIC_CLASSES = sorted([
    "Animal Rights",
    "Border Security",
    "Climate Change",
    "Fracking",
    "Gun Control",
    "Homelessness",
    "Minimum Wage",
    "Vaccines",
    "Welfare",
    "abortion",
    "black lives matter",
    "blue lives matter",
    "immigration",
    "isis",
    "lgbt",
    "racism",
    "religion",
    "terrorism",
    "unemployment",
    "war on drugs",
])

_INCL_TO_IDX = {c: i for i, c in enumerate(INCLINATION_CLASSES)}
_TOPIC_TO_IDX = {c: i for i, c in enumerate(TOPIC_CLASSES)}


def _train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _eval_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class PoliticsDataset(Dataset):
    """
    Dataset for the politics image-text pairs.

    Args:
        data_root: Directory containing the CSV files.
        split: One of "train", "val", "test".
        num_samples: If set, use only the first N rows (smoke-test mode).
        img_root: Root directory for images (local_path is relative to this).
                  Defaults to data_root. Override when CSVs and images live in
                  different locations (e.g. img_root=/ssdstore/gowreesh/politics).
    """

    def __init__(self, data_root: str | Path, split: str,
                 num_samples: int | None = None,
                 img_root: str | Path | None = None):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.data_root = Path(data_root)
        self.img_root = Path(img_root) if img_root is not None else self.data_root
        self.split = split

        csv_path = self.data_root / f"image_text_pairs_final_{split}.csv"
        assert csv_path.exists(), f"CSV not found: {csv_path}"

        df = pd.read_csv(csv_path, sep=";")
        if num_samples is not None:
            df = df.iloc[:num_samples]

        # Drop rows with missing required fields
        required = ["local_path", "label", "topic", "content_text"]
        df = df.dropna(subset=required)
        df = df[df["label"].isin(INCLINATION_CLASSES)]
        df = df[df["topic"].isin(TOPIC_CLASSES)]
        df = df.reset_index(drop=True)

        self.local_paths = df["local_path"].tolist()
        self.incl_labels = [_INCL_TO_IDX[v] for v in df["label"].tolist()]
        self.topic_labels = [_TOPIC_TO_IDX[v] for v in df["topic"].tolist()]
        self.headlines = df["image_description"].fillna("").tolist()
        self.texts = df["content_text"].fillna("").tolist()

        self.transform = _train_transforms() if split == "train" else _eval_transforms()

    def __len__(self) -> int:
        return len(self.local_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.img_root / self.local_paths[idx]

        assert img_path.exists(), f"Image not found: {img_path}"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "inclination_label_idx": self.incl_labels[idx],
            "topic_label_idx": self.topic_labels[idx],
            "headline": self.headlines[idx],
            "text": self.texts[idx],
        }
