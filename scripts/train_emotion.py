# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Train emotion classification models (CLIP+CoOp or MERU+CoOp) on Emoset dataset.

Usage:
    python scripts/train_emotion.py --config configs/emotion_clip_coop.py --output-dir output/clip_coop
    python scripts/train_emotion.py --config configs/emotion_meru_coop.py --output-dir output/meru_coop
"""

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from loguru import logger
from omegaconf.errors import ConfigAttributeError
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda import amp
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Register Emoset dataset
import meru.emotion.dataset_integration  # noqa: F401
from emoset.Emoset import EmoSet
from meru.config import LazyConfig
from meru.emotion.emotion_coop_models import CLIPCoOpEmotion, MERUCoOpEmotion
from meru.utils.checkpointing import CheckpointManager

# fmt: off
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True, help="Path to config .py file.")
parser.add_argument("--output-dir", default="./output", help="Path to save checkpoints and logs.")
parser.add_argument("--log-dir", default=None, help="Directory to save log files. If not specified, logs only to console and tensorboard.")
parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint.")
parser.add_argument("--pretrained", default="", help="Path to pretrained model checkpoint.")
parser.add_argument("--num-samples", type=int, default=None, help="Limit dataset to N samples (useful for quick testing).")
# fmt: on


def main(_A: argparse.Namespace):
    # Load config
    _C = LazyConfig.load(_A.config)

    # Set random seed
    seed = _C.train.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    output_dir = Path(_A.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file if log_dir is specified
    if _A.log_dir is not None:
        log_dir = Path(_A.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, rotation="500 MB", level="INFO")
        logger.info(f"Logging to file: {log_file}")

    logger.info(f"Using device: {device}")

    # -------------------------------------------------------------------------
    #   OVERRIDE CONFIG FROM ENVIRONMENT VARIABLES
    # -------------------------------------------------------------------------
    # Override data_root from environment variable if set (for shell scripts)
    import os
    if "EMOSET_ROOT" in os.environ:
        _C.dataset["data_root"] = os.environ["EMOSET_ROOT"]
        logger.info(f"Overriding data_root from EMOSET_ROOT: {os.environ['EMOSET_ROOT']}")
    elif "DATASET_ROOT" in os.environ:
        _C.dataset["data_root"] = os.environ["DATASET_ROOT"]
        logger.info(f"Overriding data_root from DATASET_ROOT: {os.environ['DATASET_ROOT']}")

    # Override training hyperparameters from environment variables
    if "NUM_EPOCHS" in os.environ:
        _C.train["num_epochs"] = int(os.environ["NUM_EPOCHS"])
        logger.info(f"Overriding num_epochs from NUM_EPOCHS: {os.environ['NUM_EPOCHS']}")

    if "BATCH_SIZE" in os.environ:
        _C.dataset["batch_size"] = int(os.environ["BATCH_SIZE"])
        logger.info(f"Overriding batch_size from BATCH_SIZE: {os.environ['BATCH_SIZE']}")

    if "LEARNING_RATE" in os.environ:
        _C.optim["optimizer"].lr = float(os.environ["LEARNING_RATE"])
        logger.info(f"Overriding learning_rate from LEARNING_RATE: {os.environ['LEARNING_RATE']}")

    if "EVAL_PERIOD" in os.environ:
        _C.train["eval_period"] = int(os.environ["EVAL_PERIOD"])
        logger.info(f"Overriding eval_period from EVAL_PERIOD: {os.environ['EVAL_PERIOD']}")

    if "CHECKPOINT_PERIOD" in os.environ:
        _C.train["checkpoint_period"] = int(os.environ["CHECKPOINT_PERIOD"])
        logger.info(f"Overriding checkpoint_period from CHECKPOINT_PERIOD: {os.environ['CHECKPOINT_PERIOD']}")

    if "NUM_WORKERS" in os.environ:
        _C.train["num_workers"] = int(os.environ["NUM_WORKERS"])
        logger.info(f"Overriding num_workers from NUM_WORKERS: {os.environ['NUM_WORKERS']}")

    if "SEED" in os.environ:
        _C.train["seed"] = int(os.environ["SEED"])
        seed = _C.train["seed"]
        logger.info(f"Overriding seed from SEED: {os.environ['SEED']}")

    if "NUM_CTX" in os.environ:
        _C.model.n_ctx = int(os.environ["NUM_CTX"])
        logger.info(f"Overriding n_ctx from NUM_CTX: {os.environ['NUM_CTX']}")

    if "ENTAIL_WEIGHT" in os.environ:
        # Only applies to MERU model
        if hasattr(_C.model, "entail_weight"):
            _C.model.entail_weight = float(os.environ["ENTAIL_WEIGHT"])
            logger.info(f"Overriding entail_weight from ENTAIL_WEIGHT: {os.environ['ENTAIL_WEIGHT']}")

    # Save config AFTER environment variable overrides are applied
    LazyConfig.save(_C, output_dir / "config.yaml")
    logger.info(f"Saved config to: {output_dir / 'config.yaml'}")

    # -------------------------------------------------------------------------
    #   BUILD DATASET AND DATALOADER
    # -------------------------------------------------------------------------
    logger.info("Building datasets...")

    train_dataset = EmoSet(
        data_root=_C.dataset["data_root"],
        num_emotion_classes=_C.dataset["num_emotion_classes"],
        phase="train",
    )
    val_dataset = EmoSet(
        data_root=_C.dataset["data_root"],
        num_emotion_classes=_C.dataset["num_emotion_classes"],
        phase="val",
    )

    # Limit dataset size if --num-samples is specified
    if _A.num_samples is not None:
        logger.info(f"Limiting dataset to {_A.num_samples} samples for quick testing")
        train_indices = list(range(min(_A.num_samples, len(train_dataset))))
        val_indices = list(range(min(_A.num_samples, len(val_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    batch_size = _C.dataset["batch_size"]
    num_workers = _C.train.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # -------------------------------------------------------------------------
    #   BUILD MODEL
    # -------------------------------------------------------------------------
    logger.info("Building model...")

    # Build emotion model (which internally builds the base model)
    # The config uses references like "${..clip_base_model}" so instantiate handles it
    model = instantiate(_C.model)
    model = model.to(device)

    # Load pretrained weights if specified
    pretrained_path = _A.pretrained or _C.train.get("pretrained_checkpoint", "")
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Save initial model (epoch 0) for evaluation
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 0, "model": model.state_dict()}, output_dir / "checkpoints" / "initial_model.pth")
    logger.info(f"Saved initial model to {output_dir / 'checkpoints' / 'initial_model.pth'}")

    # -------------------------------------------------------------------------
    #   BUILD OPTIMIZER AND SCHEDULER
    # -------------------------------------------------------------------------
    logger.info("Building optimizer and scheduler...")

    # Get trainable parameters (only prompt_learner and MERU hyperbolic params)
    if isinstance(model, MERUCoOpEmotion):
        # Separate param groups for MERU
        param_groups = [
            {
                "params": model.prompt_learner.parameters(),
                "lr": _C.optim["optimizer"].lr,
                "weight_decay": _C.optim["optimizer"].weight_decay,
            },
            {
                "params": [
                    model.meru.curv,
                    model.meru.visual_alpha,
                    model.meru.textual_alpha,
                ],
                "lr": _C.optim["optimizer"].lr * 0.25,  # Lower LR for hyperbolic params
                "weight_decay": 0.0,
            },
        ]
        # Build optimizer with param groups
        # Extract optimizer kwargs from LazyCall config
        optimizer_kwargs = {
            "betas": _C.optim["optimizer"].betas,
        }
        optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
    else:
        # Single param group for CLIP
        optimizer = instantiate(_C.optim["optimizer"], params=model.prompt_learner.parameters())

    # Update steps_per_epoch in config
    steps_per_epoch = len(train_loader)
    _C.train["steps_per_epoch"] = steps_per_epoch

    # Update scheduler parameters if they need to be computed dynamically
    # This is needed for LinearWarmupCosineDecayLR (MERU), but not for CosineAnnealingLR (CLIP)
    try:
        if _C.optim["lr_scheduler"].total_steps == 0:
            # Compute total_steps and warmup_steps for LinearWarmupCosineDecayLR
            total_steps = _C.train["num_epochs"] * steps_per_epoch
            warmup_steps = total_steps // 10  # 10% warmup
            _C.optim["lr_scheduler"].total_steps = total_steps
            _C.optim["lr_scheduler"].warmup_steps = warmup_steps
            logger.info(f"Computed scheduler steps: total={total_steps}, warmup={warmup_steps}")
    except (AttributeError, KeyError, ConfigAttributeError):
        # Scheduler doesn't have total_steps (e.g., CosineAnnealingLR)
        pass

    # Build scheduler
    scheduler = instantiate(_C.optim["lr_scheduler"], optimizer=optimizer)

    # Setup AMP scaler
    scaler = amp.GradScaler(enabled=_C.train.get("amp", True))

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir / "checkpoints",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    start_epoch = 0
    best_val_acc = 0.0

    if _A.resume:
        start_epoch = checkpoint_manager.resume()
        logger.info(f"Resumed from epoch {start_epoch}")

    # Setup tensorboard
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    num_epochs = _C.train["num_epochs"]
    gradient_clip = _C.train.get("gradient_clip_max_norm", None)
    eval_period = _C.train.get("eval_period", 1)
    checkpoint_period = _C.train.get("checkpoint_period", 5)

    logger.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(start_epoch, num_epochs):
        # ----------------------------------------------------------------
        #   TRAINING
        # ----------------------------------------------------------------
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["emotion_label_idx"].to(device)

            optimizer.zero_grad()

            with amp.autocast(enabled=_C.train.get("amp", True)):
                output = model(images, labels)
                loss = output["loss"]

            scaler.scale(loss).backward()

            if gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            # Clamp MERU hyperbolic parameters if needed
            if isinstance(model, MERUCoOpEmotion):
                with torch.no_grad():
                    model.meru.curv.data.clamp_(min=math.log(0.1), max=math.log(10.0))
                    model.meru.visual_alpha.data.clamp_(max=0.0)
                    model.meru.textual_alpha.data.clamp_(max=0.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Track metrics
            train_loss += loss.item()
            preds = output["logits"].argmax(dim=-1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        # Compute epoch metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds) * 100
        train_f1 = f1_score(train_labels, train_preds, average="macro") * 100

        logger.info(
            f"Epoch {epoch} Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}%"
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("train/f1", train_f1, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # ----------------------------------------------------------------
        #   VALIDATION
        # ----------------------------------------------------------------
        if (epoch + 1) % eval_period == 0:
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["emotion_label_idx"].to(device)

                    with amp.autocast(enabled=_C.train.get("amp", True)):
                        output = model(images, labels)
                        loss = output["loss"]

                    val_loss += loss.item()
                    preds = output["logits"].argmax(dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds) * 100
            val_f1 = f1_score(val_labels, val_preds, average="macro") * 100

            logger.info(
                f"Epoch {epoch} Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%"
            )

            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            writer.add_scalar("val/f1", val_f1, epoch)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_acc": best_val_acc,
                    },
                    output_dir / "checkpoints" / "best_model.pth",
                )
                logger.info(f"Saved best model with val acc: {best_val_acc:.2f}%")

        # ----------------------------------------------------------------
        #   CHECKPOINT SAVING
        # ----------------------------------------------------------------
        if (epoch + 1) % checkpoint_period == 0:
            checkpoint_manager.step(epoch)

    # Save final model
    checkpoint_manager.final_step()
    writer.close()
    logger.info(f"Training complete! Best val acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
