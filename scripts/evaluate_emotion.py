# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Evaluate emotion classification models on Emoset dataset.

Supports:
- Zero-shot CLIP/MERU evaluation with hand-crafted prompts
- Trained CLIP+CoOp or MERU+CoOp models

Usage:
    # Zero-shot evaluation
    python scripts/evaluate_emotion.py --config configs/emotion_zero_shot_clip.py --checkpoint path/to/clip_checkpoint.pth

    # Trained model evaluation
    python scripts/evaluate_emotion.py --config configs/emotion_clip_coop.py --checkpoint path/to/trained_checkpoint.pth
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.cuda import amp
from torch.utils.data import DataLoader, Subset

# Register Emoset dataset
import meru.emotion.dataset_integration  # noqa: F401
from emoset.Emoset import EmoSet
from hydra.utils import instantiate
from meru.config import LazyConfig
from meru.emotion.emotion_coop_models import CLIPCoOpEmotion, MERUCoOpEmotion
from meru.emotion.emotion_classes import EMOTION_CLASS_NAMES

# Emotion class names
EMOTION_NAMES = EMOTION_CLASS_NAMES

# fmt: off
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True, help="Path to config .py file.")
parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
parser.add_argument("--data-root", default="datasets/emoset", help="Path to Emoset dataset.")
parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to evaluate.")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation.")
parser.add_argument("--log-dir", default=None, help="Directory to save log files. If not specified, logs only to console.")
parser.add_argument("--num-samples", type=int, default=None, help="Limit dataset to N samples (useful for quick testing).")
# fmt: on


def evaluate_model(model, dataloader, device, use_amp=True):
    """
    Evaluate model on a dataset.

    Args:
        model: Emotion classification model
        dataloader: DataLoader for evaluation
        device: torch device
        use_amp: Use automatic mixed precision

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds, all_labels = [], []
    all_logits = []

    logger.info(f"Evaluating on {len(dataloader.dataset)} samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["emotion_label_idx"]

            with amp.autocast(enabled=use_amp):
                output = model(images, labels=None)  # No loss computation
                logits = output["logits"]

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_logits.append(logits.cpu())

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    all_logits = torch.cat(all_logits, dim=0)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro") * 100
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted") * 100

    # Per-class accuracy
    conf_matrix = confusion_matrix(all_labels, all_preds)
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1) * 100

    # Classification report
    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=EMOTION_NAMES,
        digits=2,
    )

    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_accuracy": {
            name: acc for name, acc in zip(EMOTION_NAMES, per_class_acc)
        },
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
    }

    return results


def main(_A: argparse.Namespace):
    # Load config - prefer the saved config from training if it exists
    checkpoint_path = Path(_A.checkpoint)
    saved_config = checkpoint_path.parent.parent / "config.yaml"

    if saved_config.exists():
        logger.info(f"Loading config from checkpoint directory: {saved_config}")
        _C = LazyConfig.load(saved_config)
    else:
        logger.info(f"Loading config from argument: {_A.config}")
        _C = LazyConfig.load(_A.config)

    # Setup logging to file if log_dir is specified
    if _A.log_dir is not None:
        log_dir = Path(_A.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"eval_{_A.split}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, rotation="500 MB", level="INFO")
        logger.info(f"Logging to file: {log_file}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # -------------------------------------------------------------------------
    #   BUILD DATASET AND DATALOADER
    # -------------------------------------------------------------------------
    logger.info(f"Loading Emoset {_A.split} split from {_A.data_root}...")

    dataset = EmoSet(
        data_root=_A.data_root,
        num_emotion_classes=8,
        phase=_A.split,
    )

    # Limit dataset size if --num-samples is specified
    if _A.num_samples is not None:
        logger.info(f"Limiting dataset to {_A.num_samples} samples for quick testing")
        indices = list(range(min(_A.num_samples, len(dataset))))
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=_A.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Loaded {len(dataset)} samples")

    # -------------------------------------------------------------------------
    #   BUILD MODEL
    # -------------------------------------------------------------------------
    logger.info("Building model...")

    # Build emotion model (which internally builds the base model)
    # The config uses references like "${..clip_base_model}" so instantiate handles it
    model = instantiate(_C.model)
    model = model.to(device)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {_A.checkpoint}...")
    checkpoint = torch.load(_A.checkpoint, map_location=device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
        if "best_val_acc" in checkpoint:
            logger.info(f"Checkpoint best val acc: {checkpoint['best_val_acc']:.2f}%")
        if "epoch" in checkpoint:
            logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # -------------------------------------------------------------------------
    #   EVALUATION
    # -------------------------------------------------------------------------
    logger.info(f"Evaluating on {_A.split} split...")

    results = evaluate_model(model, dataloader, device, use_amp=True)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info(f"EVALUATION RESULTS ({_A.split} split)")
    logger.info("=" * 80)
    logger.info(f"Overall Accuracy: {results['accuracy']:.2f}%")
    logger.info(f"Macro F1: {results['macro_f1']:.2f}%")
    logger.info(f"Weighted F1: {results['weighted_f1']:.2f}%")
    logger.info("\nPer-class Accuracy:")
    for emotion, acc in results["per_class_accuracy"].items():
        logger.info(f"  {emotion:15s}: {acc:.2f}%")

    logger.info("\nClassification Report:")
    logger.info(results["classification_report"])

    logger.info("\nConfusion Matrix:")
    logger.info(results["confusion_matrix"])

    # Save results to file
    output_file = Path(_A.checkpoint).parent / f"eval_results_{_A.split}.txt"
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"EVALUATION RESULTS ({_A.split} split)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Macro F1: {results['macro_f1']:.2f}%\n")
        f.write(f"Weighted F1: {results['weighted_f1']:.2f}%\n")
        f.write("\nPer-class Accuracy:\n")
        for emotion, acc in results["per_class_accuracy"].items():
            f.write(f"  {emotion:15s}: {acc:.2f}%\n")
        f.write("\nClassification Report:\n")
        f.write(results["classification_report"])
        f.write("\nConfusion Matrix:\n")
        f.write(str(results["confusion_matrix"]))

    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
