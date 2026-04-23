
"""
Train EmotionCLIP on EmoSet using its hybrid fine-tuning recipe
(prefix tuning + prompt tuning + LayerNorm tuning) with the train_emotion
training pipeline: AMP, gradient clipping, WandB, per-class accuracy, checkpointing.

Usage:
    python scripts/train_emotionclip.py \
        --data-root /ivi/xfs/gmago/emoset \
        --output-dir outputs/emotionclip_finetune
"""

import argparse
import importlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Register Emoset dataset
import meru.emotion.dataset_integration  # noqa: F401
from emoset.Emoset import EmoSet
from meru.emotion.emotion_classes import EMOTION_CLASS_NAMES
from meru.emotion.emotion_coop_models import EmotionCLIPEmotion
from meru.utils.checkpointing import CheckpointManager
from meru.utils.wandb_logger import WandbLogger

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

# ---------------------------------------------------------------------------
# EmotionCLIP conditional import (requires PyTorch >= 2.3 for torch.nn.attention)
# ---------------------------------------------------------------------------
_EC_DIR = os.environ.get(
    "EMOTIONCLIP_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "EmotionCLIP-V2"),
)
_EC_TORCH_OK = hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel")

if not _EC_TORCH_OK:
    raise RuntimeError(
        f"EmotionCLIP requires PyTorch >= 2.3 (torch.nn.attention.sdpa_kernel), "
        f"current version is {torch.__version__}."
    )
if not os.path.isdir(_EC_DIR):
    raise RuntimeError(
        f"EmotionCLIP-V2 not found at '{_EC_DIR}'. "
        "Clone it: git clone https://huggingface.co/jiangchengchengNLP/EmotionCLIP-V2 "
        "or set EMOTIONCLIP_DIR."
    )

if _EC_DIR not in sys.path:
    sys.path.insert(0, _EC_DIR)

_clip_clip_mod = importlib.import_module("clip.clip")
if not hasattr(_clip_clip_mod, "_convert_image_to_rgb"):
    _clip_clip_mod._convert_image_to_rgb = lambda img: img.convert("RGB")

_prev_cwd = os.getcwd()
os.chdir(_EC_DIR)
from EmotionCLIP import (  # noqa: E402
    model as EMOTIONCLIP_MODEL,
    preprocess as EMOTIONCLIP_PREPROCESS,
    tokenizer as EMOTIONCLIP_TOKENIZER,
)
os.chdir(_prev_cwd)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _EmoSetRaw(torch.utils.data.Dataset):
    """EmoSet variant that applies EmotionCLIP's preprocessor to raw PIL images."""

    def __init__(self, data_root: str, split: str, transform):
        base = EmoSet(data_root=data_root, num_emotion_classes=8, phase=split)
        self.data_store = base.data_store
        self.transform = transform

    def __len__(self):
        return len(self.data_store)

    def __getitem__(self, idx):
        emotion_label_idx, _, image_path, _ = self.data_store[idx]
        image = Image.open(image_path).convert("RGB")
        return {"image": self.transform(image), "emotion_label_idx": emotion_label_idx}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_classwise_accuracy(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class_correct = np.diag(cm)
    per_class_total = cm.sum(axis=1)
    per_class_acc = {}
    for i, name in enumerate(class_names):
        if per_class_total[i] > 0:
            per_class_acc[name] = float(per_class_correct[i] / per_class_total[i] * 100)
        else:
            per_class_acc[name] = 0.0
    return per_class_acc, cm.tolist()


def save_metrics_json(metrics_dict: dict, filepath: Path):
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(i) for i in obj]
        return obj
    with open(filepath, "w") as f:
        json.dump(_convert(metrics_dict), f, indent=2)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", required=True, help="Path to EmoSet dataset root.")
parser.add_argument("--output-dir", default="outputs/emotionclip_finetune")
parser.add_argument("--log-dir", default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--num-samples", type=int, default=None, help="Limit to N samples for quick testing.")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--gradient-clip", type=float, default=1.0)
parser.add_argument("--eval-period", type=int, default=1)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-amp", action="store_true", help="Disable AMP.")


def main(_A: argparse.Namespace):
    # Reproducibility
    random.seed(_A.seed)
    np.random.seed(_A.seed)
    torch.manual_seed(_A.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_A.seed)

    # EmotionCLIP's model is hardcoded to cuda:0 internally
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_dir = Path(_A.output_dir)
    metrics_dir = output_dir / "metrics"
    ckpt_dir = output_dir / "checkpoints"
    for d in (output_dir, metrics_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    if _A.log_dir is not None:
        log_dir = Path(_A.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log", level="INFO")

    logger.info(f"Device: {device}")

    # Apply env var overrides (mirrors train_emotion.py convention)
    data_root = os.environ.get("EMOSET_ROOT", _A.data_root)

    # -------------------------------------------------------------------------
    #   DATASET
    # -------------------------------------------------------------------------
    logger.info("Building datasets with EmotionCLIP preprocessor...")
    train_dataset = _EmoSetRaw(data_root, "train", EMOTIONCLIP_PREPROCESS)
    val_dataset = _EmoSetRaw(data_root, "val", EMOTIONCLIP_PREPROCESS)

    if _A.num_samples is not None:
        logger.info(f"Limiting to {_A.num_samples} samples for quick testing")
        train_dataset = Subset(train_dataset, list(range(min(_A.num_samples, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(_A.num_samples, len(val_dataset)))))

    persistent = _A.num_workers > 0
    prefetch = 4 if _A.num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=_A.batch_size,
        shuffle=True,
        num_workers=_A.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=_A.batch_size,
        shuffle=False,
        num_workers=_A.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # -------------------------------------------------------------------------
    #   MODEL
    # -------------------------------------------------------------------------
    logger.info("Wrapping EmotionCLIP in EmotionCLIPEmotion...")
    model = EmotionCLIPEmotion(EMOTIONCLIP_MODEL, EMOTIONCLIP_TOKENIZER, EMOTION_CLASS_NAMES)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Frozen-param invariant check: every param must be either in optimizer or frozen
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert any(k in name for k in ("prefix", "prompt", "ln")), (
                f"Unexpected trainable param: {name}"
            )

    torch.save({"epoch": 0, "model": model.state_dict()}, ckpt_dir / "initial_model.pth")

    # -------------------------------------------------------------------------
    #   OPTIMIZER + SCHEDULER
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=_A.lr,
        betas=(0.9, 0.98),
        weight_decay=_A.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = _A.epochs * steps_per_epoch
    warmup_steps = total_steps // 10

    from meru.optim import LinearWarmupCosineDecayLR
    scheduler = LinearWarmupCosineDecayLR(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps
    )

    use_amp = not _A.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_manager = CheckpointManager(
        ckpt_dir, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
    )

    start_epoch = 0
    best_val_acc = 0.0
    if _A.resume:
        start_epoch = checkpoint_manager.resume()
        logger.info(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(log_dir=output_dir / "tensorboard")

    wandb_config = {
        "model_type": "EmotionCLIP finetune",
        "num_epochs": _A.epochs,
        "batch_size": _A.batch_size,
        "lr": _A.lr,
        "trainable_params": trainable_params,
        "seed": _A.seed,
    }
    run_name = f"emotionclip_finetune"
    wandb_logger = WandbLogger(config=wandb_config, name=run_name)

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    logger.info(f"Starting training for {_A.epochs} epochs...")
    global_step = 0
    best_metrics = {"best_val_acc": 0.0, "best_epoch": 0}

    for epoch in range(start_epoch, _A.epochs):
        epoch_start = time.time()
        logger.info("=" * 80)
        logger.info(f"EPOCH {epoch + 1}/{_A.epochs}")
        logger.info("=" * 80)

        # ----------------------------------------------------------------
        #   TRAIN
        # ----------------------------------------------------------------
        model.train()
        train_loss = 0.0
        train_preds, train_labels_list = [], []
        scaler_skip_count = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["emotion_label_idx"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(images, labels)
                loss = output["loss"]

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            if _A.gradient_clip is not None:
                params_to_clip = [p for g in optimizer.param_groups for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(params_to_clip, _A.gradient_clip)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                scaler_skip_count += 1

            scheduler.step()

            train_loss += loss.item()
            preds = output["preds"].cpu().numpy()
            train_preds.extend(preds)
            train_labels_list.extend(labels.cpu().numpy())

            batch_acc = accuracy_score(labels.cpu().numpy(), preds) * 100
            wandb_logger.log(
                {
                    "train/step_loss": loss.item(),
                    "train/step_accuracy": batch_acc,
                    "train/lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                },
                step=global_step,
            )
            global_step += 1

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{_A.epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} Acc: {batch_acc:.2f}%"
                )

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels_list, train_preds) * 100
        train_f1 = f1_score(train_labels_list, train_preds, average="macro") * 100
        train_classwise_acc, train_cm = compute_classwise_accuracy(
            train_labels_list, train_preds, EMOTION_CLASS_NAMES
        )

        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ) ** 0.5

        logger.info(f"  Train → Loss: {train_loss:.4f} Acc: {train_acc:.2f}% F1: {train_f1:.2f}%")
        logger.info(
            f"  AMP   → Scale: {scaler.get_scale():.0f} "
            f"Skipped: {scaler_skip_count}/{len(train_loader)} "
            f"Grad norm: {grad_norm:.4f}"
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        wandb_logger.log(
            {"train/epoch_loss": train_loss, "train/epoch_accuracy": train_acc, "train/epoch_f1": train_f1},
            step=global_step,
        )

        # ----------------------------------------------------------------
        #   VALIDATION
        # ----------------------------------------------------------------
        if (epoch + 1) % _A.eval_period == 0:
            model.eval()
            val_loss = 0.0
            val_preds, val_labels_list = [], []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    images = batch["image"].to(device)
                    labels = batch["emotion_label_idx"].to(device)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        output = model(images, labels)
                        loss = output["loss"]

                    val_loss += loss.item()
                    preds = output["preds"].cpu().numpy()
                    val_preds.extend(preds)
                    val_labels_list.extend(labels.cpu().numpy())

                    batch_acc = accuracy_score(labels.cpu().numpy(), preds) * 100
                    if batch_idx < 3:
                        logits = output["logits"]
                        logger.info(
                            f"  Val Batch {batch_idx}: Loss={loss.item():.6f} Acc={batch_acc:.2f}% "
                            f"Logits mean={logits.mean().item():.4f} std={logits.std().item():.4f} "
                            f"Pred classes: {set(preds.tolist())}"
                        )

                    wandb_logger.log(
                        {"val/step_loss": loss.item(), "val/step_accuracy": batch_acc},
                        step=global_step,
                    )
                    global_step += 1

            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_labels_list, val_preds) * 100
            val_f1 = f1_score(val_labels_list, val_preds, average="macro") * 100
            val_classwise_acc, val_cm = compute_classwise_accuracy(
                val_labels_list, val_preds, EMOTION_CLASS_NAMES
            )

            logger.info(f"  Val   → Loss: {val_loss:.4f} Acc: {val_acc:.2f}% F1: {val_f1:.2f}%")
            epoch_time = time.time() - epoch_start
            logger.info(
                f"  Summary → Train: {train_acc:.2f}% Val: {val_acc:.2f}% "
                f"Best: {max(best_val_acc, val_acc):.2f}% Time: {epoch_time:.1f}s"
            )

            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            wandb_logger.log(
                {"val/epoch_loss": val_loss, "val/epoch_accuracy": val_acc, "val/epoch_f1": val_f1},
                step=global_step,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_metrics = {
                    "best_val_acc": best_val_acc,
                    "best_epoch": epoch,
                    "best_train_acc": train_acc,
                    "best_val_f1": val_f1,
                    "best_classwise_accuracy": val_classwise_acc,
                }
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_acc": best_val_acc,
                    },
                    ckpt_dir / "best_model.pth",
                )
                save_metrics_json(best_metrics, metrics_dir / "best_metrics.json")
                logger.info(f"  ✓ New best model! Val Acc: {best_val_acc:.2f}%")
        else:
            logger.info(f"  Time: {time.time() - epoch_start:.1f}s")

        logger.info("")

    # Final model
    torch.save(
        {
            "epoch": _A.epochs - 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
        },
        ckpt_dir / "last_model.pth",
    )
    save_metrics_json(
        {"best_val_acc": best_val_acc, "best_epoch": best_metrics.get("best_epoch", 0)},
        metrics_dir / "latest_metrics.json",
    )

    wandb_logger.log({"best_val_accuracy": best_val_acc})
    writer.close()
    wandb_logger.finish()
    logger.info(f"Training complete! Best val acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
