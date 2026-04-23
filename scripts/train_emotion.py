# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Train emotion classification models (CLIP+CoOp or MERU+CoOp) on Emoset dataset.

Usage:
    python scripts/train_emotion.py --config configs/emotion_clip_coop.py --output-dir output/clip_coop
    python scripts/train_emotion.py --config configs/emotion_meru_coop.py --output-dir output/meru_coop
"""

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from loguru import logger
from omegaconf.errors import ConfigAttributeError
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

# Register Emoset dataset
import meru.emotion.dataset_integration  # noqa: F401
from emoset.Emoset import EmoSet
from meru.config import LazyConfig
from meru.emotion.emotion_classes import EMOTION_CLASS_NAMES
from meru.emotion.emotion_coop_models import CLIPCoOpEmotion, CLIPCoOpOpenAI, MERUCoOpEmotion
from meru.emotion.emotionclipv2_model import EmotionCLIPV2Emotion
from meru.utils.checkpointing import CheckpointManager
from meru.utils.wandb_logger import WandbLogger

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)


def save_metrics_json(metrics_dict: dict, filepath: Path):
    """Save metrics dictionary to JSON file."""
    # Convert any numpy types to Python native types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable = convert_to_serializable(metrics_dict)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)


def compute_classwise_accuracy(y_true, y_pred, class_names):
    """Compute per-class accuracy and return as dictionary."""
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


# fmt: off
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True, help="Path to config .py file.")
parser.add_argument("--output-dir", default="./output", help="Path to save checkpoints and logs.")
parser.add_argument("--log-dir", default=None, help="Directory to save log files. If not specified, logs only to console and tensorboard.")
parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint.")
parser.add_argument("--pretrained", default="", help="Path to pretrained model checkpoint.")
parser.add_argument("--num-samples", type=int, default=None, help="Limit dataset to N samples (useful for quick testing).")
# fmt: on


class CachedDataset(torch.utils.data.Dataset):
    """Lazily caches dataset items in worker memory after first access.

    With persistent_workers=True, each worker's cache survives across epochs.
    Safe only for datasets with deterministic transforms (val/test).
    """
    def __init__(self, dataset):
        self._dataset = dataset
        self._cache = {}

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        if idx not in self._cache:
            self._cache[idx] = self._dataset[idx]
        return self._cache[idx]


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

    # Create output directory and metrics subdirectory
    output_dir = Path(_A.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

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
    persistent = num_workers > 0
    prefetch = 4 if num_workers > 0 else None

    # Train: no item-level cache — random crop/flip must vary each epoch.
    # persistent_workers keeps worker processes (and their OS page cache) alive
    # between epochs, avoiding per-epoch worker restart overhead.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    # Val: deterministic transforms — wrap in CachedDataset so each worker
    # caches its assigned items after the first epoch and serves from memory
    # for all subsequent epochs.
    val_loader = DataLoader(
        CachedDataset(val_dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # -------------------------------------------------------------------------
    #   BUILD MODEL
    # -------------------------------------------------------------------------
    logger.info("Building model...")

    # Load pretrained weights into base model BEFORE wrapping with CoOp
    pretrained_path = _A.pretrained or _C.train.get("pretrained_checkpoint", "")
    if pretrained_path and Path(pretrained_path).exists():
        logger.info(f"Loading pretrained weights into base model from {pretrained_path}")

        # First, build and load the base model (keep on CPU for now)
        if hasattr(_C, 'clip_base_model'):
            logger.info("Building CLIPBaseline base model...")
            base_model = instantiate(_C.clip_base_model)

            # Load pretrained weights into base model (on CPU)
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if "model" in checkpoint:
                base_model.load_state_dict(checkpoint["model"], strict=True)
            else:
                base_model.load_state_dict(checkpoint, strict=True)
            logger.info(f"✓ Loaded pretrained CLIP weights successfully")

            # Replace the LazyCall reference with the actual loaded model
            _C.model.clip_model = base_model

        elif hasattr(_C, 'meru_base_model'):
            logger.info("Building MERU base model...")
            base_model = instantiate(_C.meru_base_model)

            # Load pretrained weights into base model (on CPU)
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if "model" in checkpoint:
                base_model.load_state_dict(checkpoint["model"], strict=True)
            else:
                base_model.load_state_dict(checkpoint, strict=True)
            logger.info(f"✓ Loaded pretrained MERU weights successfully")

            # Replace the LazyCall reference with the actual loaded model
            _C.model.meru_model = base_model
        else:
            logger.warning("No base model found in config, skipping pretrained weight loading")

    # Build emotion model (wraps the base model with CoOp components)
    logger.info("Building emotion classification model...")
    model = instantiate(_C.model)
    model = model.to(device)

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
    scaler = torch.cuda.amp.GradScaler(enabled=_C.train.get("amp", True))

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

    # Setup wandb (optional, controlled by environment variables)
    wandb_config = {
        "model_type": (
            "CLIP+CoOp (OpenAI)" if isinstance(model, CLIPCoOpOpenAI)
            else "CLIP+CoOp" if isinstance(model, CLIPCoOpEmotion)
            else "MERU+CoOp" if isinstance(model, MERUCoOpEmotion)
            else "EmotionCLIPV2+Prompt"
        ),
        "num_epochs": _C.train["num_epochs"],
        "batch_size": _C.dataset["batch_size"],
        "learning_rate": _C.optim["optimizer"].lr,
        "num_ctx": model.n_ctx if hasattr(model, 'n_ctx') else None,
        "dataset": _C.dataset["name"],
        "seed": seed,
    }
    wandb_logger = WandbLogger(config=wandb_config)

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    num_epochs = _C.train["num_epochs"]
    gradient_clip = _C.train.get("gradient_clip_max_norm", None)
    eval_period = _C.train.get("eval_period", 1)

    logger.info(f"Starting training for {num_epochs} epochs...")
    logger.info(f"Evaluation period: Every {eval_period} epoch(s)")
    logger.info(f"Checkpoints: Saving best model and final model only")
    logger.info("")

    # Global step counter for wandb logging
    global_step = 0

    # Track best metrics for JSON saving
    best_metrics = {"best_val_acc": 0.0, "best_epoch": 0}

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        logger.info("=" * 80)
        logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
        logger.info("=" * 80)

        # ----------------------------------------------------------------
        #   TRAINING
        # ----------------------------------------------------------------
        logger.info(f"Training...")
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        scaler_skip_count = 0  # track how many steps AMP skips due to overflow

        # Accumulators for MERU-specific metrics (for epoch averages)
        meru_metrics_accum = {
            "image_norm_mean": [], "image_norm_std": [], "image_norm_min": [], "image_norm_max": [],
            "text_norm_mean": [], "text_norm_std": [], "text_norm_min": [], "text_norm_max": [],
            "distance_mean": [], "distance_std": [],
            "entailment_violations": [], "contrastive_violations": [],
            "entailment_violation_rate": [], "contrastive_violation_rate": [],
            "num_misclassified": [],
            "misclassified_with_entail_violation": [], "misclassified_pred_entail_valid": [],
            "misclassified_avg_distance_margin": [],
            "angle_mean": [], "angle_std": [], "aperture_mean": [], "aperture_std": [],
            "contrastive_loss": [], "entailment_loss": [],
        }

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["emotion_label_idx"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=_C.train.get("amp", True)):
                output = model(images, labels)
                # Use the model's computed loss: for MERU this includes the
                # entailment term; for CLIP it is plain cross-entropy.
                loss = output["loss"]

            scaler.scale(loss).backward()

            # Always unscale before clipping so grad norms are in true scale
            scaler.unscale_(optimizer)
            if gradient_clip is not None:
                # Clip only the parameters in the optimizer (prompt_learner).
                # Using model.parameters() includes frozen/unoptimised params
                # (e.g. logit_scale) whose gradients are still at AMP scale,
                # which makes the total norm ≈ scale_factor and drives
                # clip_coeff ≈ 0, zeroing ctx.grad on every step.
                params_to_clip = [
                    p for group in optimizer.param_groups for p in group["params"]
                ]
                torch.nn.utils.clip_grad_norm_(params_to_clip, gradient_clip)

            # Hyperbolic parameter clamping is handled inside MERUCoOpEmotion.forward()
            # at the top of every forward pass (training and validation), matching
            # vanilla MERU's behavior. No post-step clamping needed here.

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                scaler_skip_count += 1

            scheduler.step()

            # Track metrics
            train_loss += loss.item()
            preds = output["preds"].cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())

            # Compute batch accuracy
            batch_acc = accuracy_score(labels.cpu().numpy(), preds) * 100

            # Log step-level metrics to wandb (streamlined - only essential)
            step_metrics = {
                "train/step_loss": loss.item(),
                "train/step_accuracy": batch_acc,
                "train/lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            }

            # Add MERU-specific metrics if applicable
            if isinstance(model, MERUCoOpEmotion) and "contrastive_loss" in output:
                # Loss components
                step_metrics["train/step_contrastive_loss"] = output["contrastive_loss"].item()
                step_metrics["train/step_entailment_loss"] = output["entailment_loss"].item()

                # Hyperbolic parameters
                step_metrics["meru/curv"] = model.meru.curv.exp().item()
                step_metrics["meru/visual_alpha"] = model.meru.visual_alpha.exp().item()
                step_metrics["meru/textual_alpha"] = model.meru.textual_alpha.exp().item()

                # All metrics from MERUCoOpEmotion.forward()
                metrics = output.get("metrics", {})

                # Image norm statistics
                step_metrics["meru/image_norm_mean"] = metrics.get("image_norm_mean", 0)
                step_metrics["meru/image_norm_std"] = metrics.get("image_norm_std", 0)
                step_metrics["meru/image_norm_min"] = metrics.get("image_norm_min", 0)
                step_metrics["meru/image_norm_max"] = metrics.get("image_norm_max", 0)

                # Text norm statistics
                step_metrics["meru/text_norm_mean"] = metrics.get("text_norm_mean", 0)
                step_metrics["meru/text_norm_std"] = metrics.get("text_norm_std", 0)
                step_metrics["meru/text_norm_min"] = metrics.get("text_norm_min", 0)
                step_metrics["meru/text_norm_max"] = metrics.get("text_norm_max", 0)

                # Distance statistics
                step_metrics["meru/distance_mean"] = metrics.get("distance_mean", 0)
                step_metrics["meru/distance_std"] = metrics.get("distance_std", 0)

                # Violation counts and rates
                step_metrics["meru/entailment_violations"] = metrics.get("entailment_violations", 0)
                step_metrics["meru/entailment_violation_rate"] = metrics.get("entailment_violation_rate", 0)
                step_metrics["meru/contrastive_violations"] = metrics.get("contrastive_violations", 0)
                step_metrics["meru/contrastive_violation_rate"] = metrics.get("contrastive_violation_rate", 0)

                # Misclassification diagnosis
                step_metrics["meru/num_misclassified"] = metrics.get("num_misclassified", 0)
                step_metrics["meru/misclassified_with_entail_violation"] = metrics.get("misclassified_with_entail_violation", 0)
                step_metrics["meru/misclassified_pred_entail_valid"] = metrics.get("misclassified_pred_entail_valid", 0)
                step_metrics["meru/misclassified_avg_distance_margin"] = metrics.get("misclassified_avg_distance_margin", 0)

                # Angle and aperture statistics (entailment cone geometry)
                step_metrics["meru/angle_mean"] = metrics.get("angle_mean", 0)
                step_metrics["meru/angle_std"] = metrics.get("angle_std", 0)
                step_metrics["meru/aperture_mean"] = metrics.get("aperture_mean", 0)
                step_metrics["meru/aperture_std"] = metrics.get("aperture_std", 0)

                # Accumulate for epoch-level JSON
                for key in meru_metrics_accum:
                    if key in metrics:
                        meru_metrics_accum[key].append(metrics[key])
                    elif key == "contrastive_loss":
                        meru_metrics_accum[key].append(output["contrastive_loss"].item())
                    elif key == "entailment_loss":
                        meru_metrics_accum[key].append(output["entailment_loss"].item())

            wandb_logger.log(step_metrics, step=global_step)
            global_step += 1

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%"
                )

        # Compute epoch metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds) * 100
        train_f1 = f1_score(train_labels, train_preds, average="macro") * 100

        # Compute class-wise accuracy (verbose - saved to JSON, not WandB)
        train_classwise_acc, train_confusion_matrix = compute_classwise_accuracy(
            train_labels, train_preds, EMOTION_CLASS_NAMES
        )

        # Gradient norm of trainable params (prompt ctx) at end of epoch
        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ) ** 0.5

        logger.info(
            f"  Train → Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}%"
        )
        logger.info(
            f"  AMP  → Scaler scale: {scaler.get_scale():.0f}, "
            f"Skipped steps: {scaler_skip_count}/{len(train_loader)}, "
            f"Grad norm (ctx): {grad_norm:.4f}"
        )

        # Compute epoch-averaged MERU metrics for JSON
        epoch_meru_metrics = {}
        if isinstance(model, MERUCoOpEmotion) and meru_metrics_accum["image_norm_mean"]:
            for key, values in meru_metrics_accum.items():
                if values:
                    epoch_meru_metrics[f"train_{key}_mean"] = float(np.mean(values))
                    epoch_meru_metrics[f"train_{key}_std"] = float(np.std(values))

            # Log summary MERU metrics
            logger.info(
                f"  MERU → Image norm: {epoch_meru_metrics.get('train_image_norm_mean_mean', 0):.4f}, "
                f"Text norm: {epoch_meru_metrics.get('train_text_norm_mean_mean', 0):.4f}, "
                f"Entail viol rate: {epoch_meru_metrics.get('train_entailment_violation_rate_mean', 0):.4f}"
            )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("train/f1", train_f1, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # Log epoch-level train summary to wandb (essential metrics only)
        epoch_wandb_metrics = {
            "train/epoch_loss": train_loss,
            "train/epoch_accuracy": train_acc,
            "train/epoch_f1": train_f1,
        }
        # Add MERU epoch-level summaries to WandB
        if isinstance(model, MERUCoOpEmotion) and epoch_meru_metrics:
            # Norm statistics
            epoch_wandb_metrics["meru/epoch_image_norm"] = epoch_meru_metrics.get("train_image_norm_mean_mean", 0)
            epoch_wandb_metrics["meru/epoch_text_norm"] = epoch_meru_metrics.get("train_text_norm_mean_mean", 0)

            # Violation rates
            epoch_wandb_metrics["meru/epoch_entail_viol_rate"] = epoch_meru_metrics.get("train_entailment_violation_rate_mean", 0)
            epoch_wandb_metrics["meru/epoch_contrastive_viol_rate"] = epoch_meru_metrics.get("train_contrastive_violation_rate_mean", 0)

            # Loss components
            epoch_wandb_metrics["meru/epoch_contrastive_loss"] = epoch_meru_metrics.get("train_contrastive_loss_mean", 0)
            epoch_wandb_metrics["meru/epoch_entailment_loss"] = epoch_meru_metrics.get("train_entailment_loss_mean", 0)

            # Geometry statistics
            epoch_wandb_metrics["meru/epoch_angle_mean"] = epoch_meru_metrics.get("train_angle_mean_mean", 0)
            epoch_wandb_metrics["meru/epoch_aperture_mean"] = epoch_meru_metrics.get("train_aperture_mean_mean", 0)

        wandb_logger.log(epoch_wandb_metrics, step=global_step)

        # ----------------------------------------------------------------
        #   VALIDATION
        # ----------------------------------------------------------------
        if (epoch + 1) % eval_period == 0:
            logger.info(f"Validating...")
            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []

            # Accumulators for MERU-specific validation metrics
            val_meru_metrics_accum = {
                "image_norm_mean": [], "image_norm_std": [], "image_norm_min": [], "image_norm_max": [],
                "text_norm_mean": [], "text_norm_std": [], "text_norm_min": [], "text_norm_max": [],
                "distance_mean": [], "distance_std": [],
                "entailment_violations": [], "contrastive_violations": [],
                "entailment_violation_rate": [], "contrastive_violation_rate": [],
                "num_misclassified": [],
                "misclassified_with_entail_violation": [], "misclassified_pred_entail_valid": [],
                "misclassified_avg_distance_margin": [],
                "angle_mean": [], "angle_std": [], "aperture_mean": [], "aperture_std": [],
                "contrastive_loss": [], "entailment_loss": [],
            }

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    images = batch["image"].to(device)
                    labels = batch["emotion_label_idx"].to(device)

                    with torch.cuda.amp.autocast(enabled=_C.train.get("amp", True)):
                        output = model(images, labels)
                        loss = output["loss"]

                    val_loss += loss.item()
                    preds = output["preds"].cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())

                    # Compute batch accuracy
                    batch_acc = accuracy_score(labels.cpu().numpy(), preds) * 100

                    # Debug: Log logits statistics for first few batches
                    if batch_idx < 3:
                        logits = output["logits"]
                        debug_msg = (
                            f"  Val Batch {batch_idx}: Loss={loss.item():.6f}, "
                            f"Acc={batch_acc:.2f}%, "
                            f"Logits mean={logits.mean().item():.4f}, std={logits.std().item():.4f}, "
                            f"Pred classes: {set(preds.tolist())}"
                        )
                        # Add CLIP-specific debug info
                        if isinstance(model, CLIPCoOpEmotion) and "debug" in output:
                            debug_info = output["debug"]
                            debug_msg += (
                                f", Text sim mean={debug_info['text_similarity_mean']:.4f}, "
                                f"std={debug_info['text_similarity_std']:.4f}"
                            )
                        logger.info(debug_msg)

                    # Log step-level validation metrics to wandb (streamlined)
                    val_step_metrics = {
                        "val/step_loss": loss.item(),
                        "val/step_accuracy": batch_acc,
                        "epoch": epoch,
                    }

                    # Add MERU-specific metrics if applicable
                    if isinstance(model, MERUCoOpEmotion) and "contrastive_loss" in output:
                        # Loss components
                        val_step_metrics["val/step_contrastive_loss"] = output["contrastive_loss"].item()
                        val_step_metrics["val/step_entailment_loss"] = output["entailment_loss"].item()

                        # All metrics from MERUCoOpEmotion.forward()
                        metrics = output.get("metrics", {})

                        # Image norm statistics
                        val_step_metrics["meru_val/image_norm_mean"] = metrics.get("image_norm_mean", 0)
                        val_step_metrics["meru_val/image_norm_std"] = metrics.get("image_norm_std", 0)
                        val_step_metrics["meru_val/image_norm_min"] = metrics.get("image_norm_min", 0)
                        val_step_metrics["meru_val/image_norm_max"] = metrics.get("image_norm_max", 0)

                        # Text norm statistics
                        val_step_metrics["meru_val/text_norm_mean"] = metrics.get("text_norm_mean", 0)
                        val_step_metrics["meru_val/text_norm_std"] = metrics.get("text_norm_std", 0)
                        val_step_metrics["meru_val/text_norm_min"] = metrics.get("text_norm_min", 0)
                        val_step_metrics["meru_val/text_norm_max"] = metrics.get("text_norm_max", 0)

                        # Distance statistics
                        val_step_metrics["meru_val/distance_mean"] = metrics.get("distance_mean", 0)
                        val_step_metrics["meru_val/distance_std"] = metrics.get("distance_std", 0)

                        # Violation counts and rates
                        val_step_metrics["meru_val/entailment_violations"] = metrics.get("entailment_violations", 0)
                        val_step_metrics["meru_val/entailment_violation_rate"] = metrics.get("entailment_violation_rate", 0)
                        val_step_metrics["meru_val/contrastive_violations"] = metrics.get("contrastive_violations", 0)
                        val_step_metrics["meru_val/contrastive_violation_rate"] = metrics.get("contrastive_violation_rate", 0)

                        # Misclassification diagnosis
                        val_step_metrics["meru_val/num_misclassified"] = metrics.get("num_misclassified", 0)
                        val_step_metrics["meru_val/misclassified_with_entail_violation"] = metrics.get("misclassified_with_entail_violation", 0)
                        val_step_metrics["meru_val/misclassified_pred_entail_valid"] = metrics.get("misclassified_pred_entail_valid", 0)
                        val_step_metrics["meru_val/misclassified_avg_distance_margin"] = metrics.get("misclassified_avg_distance_margin", 0)

                        # Angle and aperture statistics (entailment cone geometry)
                        val_step_metrics["meru_val/angle_mean"] = metrics.get("angle_mean", 0)
                        val_step_metrics["meru_val/angle_std"] = metrics.get("angle_std", 0)
                        val_step_metrics["meru_val/aperture_mean"] = metrics.get("aperture_mean", 0)
                        val_step_metrics["meru_val/aperture_std"] = metrics.get("aperture_std", 0)

                        # Accumulate for epoch-level JSON
                        for key in val_meru_metrics_accum:
                            if key in metrics:
                                val_meru_metrics_accum[key].append(metrics[key])
                            elif key == "contrastive_loss":
                                val_meru_metrics_accum[key].append(output["contrastive_loss"].item())
                            elif key == "entailment_loss":
                                val_meru_metrics_accum[key].append(output["entailment_loss"].item())

                    wandb_logger.log(val_step_metrics, step=global_step)
                    global_step += 1

            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds) * 100
            val_f1 = f1_score(val_labels, val_preds, average="macro") * 100

            # Compute class-wise accuracy (verbose - saved to JSON, not WandB)
            val_classwise_acc, val_confusion_matrix = compute_classwise_accuracy(
                val_labels, val_preds, EMOTION_CLASS_NAMES
            )

            # Compute epoch-averaged MERU validation metrics for JSON
            val_epoch_meru_metrics = {}
            if isinstance(model, MERUCoOpEmotion) and val_meru_metrics_accum["image_norm_mean"]:
                for key, values in val_meru_metrics_accum.items():
                    if values:
                        val_epoch_meru_metrics[f"val_{key}_mean"] = float(np.mean(values))
                        val_epoch_meru_metrics[f"val_{key}_std"] = float(np.std(values))

            logger.info(
                f"  Val   → Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%"
            )

            # Log MERU validation summary
            if isinstance(model, MERUCoOpEmotion) and val_epoch_meru_metrics:
                logger.info(
                    f"  MERU Val → Entail viol rate: {val_epoch_meru_metrics.get('val_entailment_violation_rate_mean', 0):.4f}, "
                    f"Misclass w/ entail viol: {val_epoch_meru_metrics.get('val_misclassified_with_entail_violation_mean', 0):.2f}"
                )

            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            writer.add_scalar("val/f1", val_f1, epoch)

            # Log epoch-level validation summary to wandb (essential metrics only)
            val_wandb_metrics = {
                "val/epoch_loss": val_loss,
                "val/epoch_accuracy": val_acc,
                "val/epoch_f1": val_f1,
            }
            # Add MERU epoch-level summaries to WandB
            if isinstance(model, MERUCoOpEmotion) and val_epoch_meru_metrics:
                # Violation rates
                val_wandb_metrics["meru/val_entail_viol_rate"] = val_epoch_meru_metrics.get("val_entailment_violation_rate_mean", 0)
                val_wandb_metrics["meru/val_contrastive_viol_rate"] = val_epoch_meru_metrics.get("val_contrastive_violation_rate_mean", 0)

                # Loss components
                val_wandb_metrics["meru/val_contrastive_loss"] = val_epoch_meru_metrics.get("val_contrastive_loss_mean", 0)
                val_wandb_metrics["meru/val_entailment_loss"] = val_epoch_meru_metrics.get("val_entailment_loss_mean", 0)

                # Norm statistics
                val_wandb_metrics["meru/val_image_norm"] = val_epoch_meru_metrics.get("val_image_norm_mean_mean", 0)
                val_wandb_metrics["meru/val_text_norm"] = val_epoch_meru_metrics.get("val_text_norm_mean_mean", 0)

                # Geometry statistics
                val_wandb_metrics["meru/val_angle_mean"] = val_epoch_meru_metrics.get("val_angle_mean_mean", 0)
                val_wandb_metrics["meru/val_aperture_mean"] = val_epoch_meru_metrics.get("val_aperture_mean_mean", 0)

                # Misclassification analysis
                val_wandb_metrics["meru/val_misclassified_with_entail_viol"] = val_epoch_meru_metrics.get("val_misclassified_with_entail_violation_mean", 0)
                val_wandb_metrics["meru/val_misclassified_pred_entail_valid"] = val_epoch_meru_metrics.get("val_misclassified_pred_entail_valid_mean", 0)

            wandb_logger.log(val_wandb_metrics, step=global_step)

            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"  Summary → Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                f"Best Val: {max(best_val_acc, val_acc):.2f}% | Time: {epoch_time:.1f}s"
            )

            # ----------------------------------------------------------------
            # Save verbose metrics to JSON (class-wise accuracy, confusion matrix, etc.)
            # ----------------------------------------------------------------
            epoch_verbose_metrics = {
                "epoch": epoch,
                "train": {
                    "loss": train_loss,
                    "accuracy": train_acc,
                    "f1": train_f1,
                    "classwise_accuracy": train_classwise_acc,
                    "confusion_matrix": train_confusion_matrix,
                },
                "val": {
                    "loss": val_loss,
                    "accuracy": val_acc,
                    "f1": val_f1,
                    "classwise_accuracy": val_classwise_acc,
                    "confusion_matrix": val_confusion_matrix,
                },
            }
            # Add MERU-specific detailed metrics
            if isinstance(model, MERUCoOpEmotion):
                epoch_verbose_metrics["meru"] = {
                    "curv": model.meru.curv.exp().item(),
                    "visual_alpha": model.meru.visual_alpha.exp().item(),
                    "textual_alpha": model.meru.textual_alpha.exp().item(),
                    "train_metrics": epoch_meru_metrics,
                    "val_metrics": val_epoch_meru_metrics,
                }

            # Save to epoch-specific JSON
            # save_metrics_json(epoch_verbose_metrics, metrics_dir / f"epoch_{epoch:04d}.json")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_metrics = {
                    "best_val_acc": best_val_acc,
                    "best_epoch": epoch,
                    "best_train_acc": train_acc,
                    "best_val_f1": val_f1,
                    "best_classwise_accuracy": val_classwise_acc,
                }
                if isinstance(model, MERUCoOpEmotion):
                    best_metrics["meru"] = epoch_verbose_metrics.get("meru", {})

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
                # Save best metrics JSON
                save_metrics_json(best_metrics, metrics_dir / "best_metrics.json")
                logger.info(f"  ✓ New best model! Val Acc: {best_val_acc:.2f}%")
        else:
            # Still log epoch time even if no validation
            epoch_time = time.time() - epoch_start_time
            logger.info(f"  Time: {epoch_time:.1f}s")

        logger.info("")  # Blank line between epochs

    # Save final/last model
    logger.info("Saving final model...")
    torch.save(
        {
            "epoch": num_epochs - 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
        },
        output_dir / "checkpoints" / "last_model.pth",
    )
    logger.info(f"✓ Final model saved to {output_dir / 'checkpoints' / 'last_model.pth'}")

    # Save latest metrics JSON
    latest_metrics = {
        "final_epoch": num_epochs - 1,
        "best_val_acc": best_val_acc,
        "best_epoch": best_metrics.get("best_epoch", 0),
        "train": {
            "final_loss": train_loss,
            "final_accuracy": train_acc,
            "final_f1": train_f1,
        },
    }
    if isinstance(model, MERUCoOpEmotion):
        latest_metrics["meru"] = {
            "curv": model.meru.curv.exp().item(),
            "visual_alpha": model.meru.visual_alpha.exp().item(),
            "textual_alpha": model.meru.textual_alpha.exp().item(),
        }
    save_metrics_json(latest_metrics, metrics_dir / "latest_metrics.json")
    logger.info(f"✓ Latest metrics saved to {metrics_dir / 'latest_metrics.json'}")

    # Log final results to wandb (essential summary only)
    wandb_logger.log({"best_val_accuracy": best_val_acc})

    writer.close()
    wandb_logger.finish()
    logger.info(f"Training complete! Best val acc: {best_val_acc:.2f}%")
    logger.info(f"Verbose metrics saved to: {metrics_dir}")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
