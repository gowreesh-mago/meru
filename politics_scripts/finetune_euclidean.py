"""
Euclidean fine-tuning baseline for political inclination and topic classification.

Architecture:
  - Frozen CLIPBaseline (image + text encoders)
  - Learnable fusion layer: Linear(1024 → hidden_dim) + ReLU + LayerNorm
  - Inclination head: Linear(hidden_dim → 2)
  - Topic head:       Linear(hidden_dim → 20)

Text input: content_text (truncated to 77 tokens by CLIP tokenizer).

Usage:
    python politics_scripts/finetune_euclidean.py \\
        --checkpoint /path/to/clip.pth \\
        --data-root /home/gmago/Emotions/politics_data \\
        --epochs 20 \\
        --lr 1e-3 \\
        --no-wandb \\
        --num-samples 10        # smoke test
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder
from meru.models import CLIPBaseline
from meru.tokenizer import Tokenizer
from meru.utils.wandb_logger import WandbLogger
from politics_scripts.politics_dataset import (
    INCLINATION_CLASSES,
    TOPIC_CLASSES,
    PoliticsDataset,
)

NUM_INCL = len(INCLINATION_CLASSES)
NUM_TOPIC = len(TOPIC_CLASSES)

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--data-root", required=True)
parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="Evaluation split (training always uses train).")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden-dim", type=int, default=256)
parser.add_argument("--task-weight", type=float, default=1.0,
                    help="Weight applied equally to both CE losses (both are summed).")
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--output-dir", default="/home/gmago/Emotions/outputs/politics/euclidean")
parser.add_argument("--img-root", default=None,
                    help="Root dir for images (default: same as --data-root). "
                         "Set to /ssdstore/gowreesh/politics when CSVs and images differ.")
parser.add_argument("--num-samples", type=int, default=None,
                    help="Limit dataset to first N samples for smoke testing.")
parser.add_argument("--no-wandb", action="store_true")
parser.add_argument("--seed", type=int, default=42)


# ── model ────────────────────────────────────────────────────────────────────
class PoliticsEuclideanClassifier(nn.Module):
    """
    Frozen CLIP + learnable fusion + two classification heads.
    CLIP params have requires_grad=False; only fusion/head params are optimised.
    """

    def __init__(self, clip_model: CLIPBaseline, hidden_dim: int = 256):
        super().__init__()
        self.clip = clip_model

        # Enforce frozen-param invariant: all CLIP params → requires_grad=False
        for p in self.clip.parameters():
            p.requires_grad = False

        embed_dim = clip_model.embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.inclination_head = nn.Linear(hidden_dim, NUM_INCL)
        self.topic_head = nn.Linear(hidden_dim, NUM_TOPIC)

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: list[torch.Tensor],
        incl_labels: torch.Tensor | None = None,
        topic_labels: torch.Tensor | None = None,
    ) -> dict:
        with torch.no_grad():
            img_feat = self.clip.encode_image(images, project=True)    # (B, D)
            txt_feat = self.clip.encode_text(text_tokens, project=True)  # (B, D)

        fused = self.fusion(torch.cat([img_feat, txt_feat], dim=-1))  # (B, H)
        incl_logits = self.inclination_head(fused)   # (B, 2)
        topic_logits = self.topic_head(fused)         # (B, 20)

        out = {
            "incl_logits": incl_logits,
            "topic_logits": topic_logits,
            "incl_preds": incl_logits.argmax(dim=-1),
            "topic_preds": topic_logits.argmax(dim=-1),
        }

        if incl_labels is not None and topic_labels is not None:
            incl_loss = F.cross_entropy(incl_logits, incl_labels)
            topic_loss = F.cross_entropy(topic_logits, topic_labels)
            out["loss"] = incl_loss + topic_loss
            out["incl_loss"] = incl_loss
            out["topic_loss"] = topic_loss

        return out


# ── metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(
    incl_preds: list[int], incl_labels: list[int],
    topic_preds: list[int], topic_labels: list[int],
) -> dict:
    cm_incl = confusion_matrix(incl_labels, incl_preds, labels=list(range(NUM_INCL)))
    per_incl = (cm_incl.diagonal() / cm_incl.sum(axis=1).clip(min=1) * 100).tolist()

    cm_topic = confusion_matrix(topic_labels, topic_preds, labels=list(range(NUM_TOPIC)))
    per_topic = (cm_topic.diagonal() / cm_topic.sum(axis=1).clip(min=1) * 100).tolist()

    incl_arr = np.array(incl_labels)
    topic_pred_arr = np.array(topic_preds)
    topic_label_arr = np.array(topic_labels)
    conditioned: dict[str, float] = {}
    for i, name in enumerate(INCLINATION_CLASSES):
        mask = incl_arr == i
        if mask.sum() > 0:
            conditioned[name] = float(
                accuracy_score(topic_label_arr[mask], topic_pred_arr[mask]) * 100)

    return {
        "incl_accuracy": float(accuracy_score(incl_labels, incl_preds) * 100),
        "topic_accuracy": float(accuracy_score(topic_labels, topic_preds) * 100),
        "incl_macro_f1": float(f1_score(incl_labels, incl_preds, average="macro",
                                         zero_division=0) * 100),
        "topic_macro_f1": float(f1_score(topic_labels, topic_preds, average="macro",
                                          zero_division=0) * 100),
        "per_class_incl": {INCLINATION_CLASSES[i]: per_incl[i] for i in range(NUM_INCL)},
        "per_class_topic": {TOPIC_CLASSES[i]: per_topic[i] for i in range(NUM_TOPIC)},
        "conditioned_topic_accuracy_given_incl": conditioned,
    }


def log_metrics(metrics: dict, epoch: int, prefix: str = "val") -> None:
    logger.info(f"[{prefix}] epoch={epoch}  "
                f"incl_acc={metrics['incl_accuracy']:.2f}%  "
                f"topic_acc={metrics['topic_accuracy']:.2f}%  "
                f"incl_f1={metrics['incl_macro_f1']:.2f}%  "
                f"topic_f1={metrics['topic_macro_f1']:.2f}%")
    logger.info("  Per-class inclination:")
    for n, a in metrics["per_class_incl"].items():
        logger.info(f"    {n}: {a:.2f}%")
    logger.info("  Per-class topic (top-5):")
    top5 = sorted(metrics["per_class_topic"].items(), key=lambda x: -x[1])[:5]
    for n, a in top5:
        logger.info(f"    {n}: {a:.2f}%")
    logger.info("  Conditioned topic accuracy (given true inclination):")
    for n, a in metrics["conditioned_topic_accuracy_given_incl"].items():
        logger.info(f"    {n}: {a:.2f}%")


# ── train / eval loops ────────────────────────────────────────────────────────
def run_epoch_train(model, loader, optimizer, scaler, scheduler, tokenizer, device,
                    grad_clip: float) -> dict:
    model.train()
    total_loss = incl_loss_sum = topic_loss_sum = 0.0
    n = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        incl_labels = batch["inclination_label_idx"].to(device)
        topic_labels = batch["topic_label_idx"].to(device)
        texts = batch["text"]
        tokens = tokenizer(list(texts))

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(imgs, tokens, incl_labels, topic_labels)
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        params_to_clip = [p for g in optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        incl_loss_sum += out["incl_loss"].item() * bs
        topic_loss_sum += out["topic_loss"].item() * bs
        n += bs

    return {
        "loss": total_loss / n,
        "incl_loss": incl_loss_sum / n,
        "topic_loss": topic_loss_sum / n,
    }


@torch.no_grad()
def run_epoch_eval(model, loader, tokenizer, device) -> tuple[dict, dict]:
    model.eval()
    all_incl_preds, all_incl_labels = [], []
    all_topic_preds, all_topic_labels = [], []
    total_loss = n = 0

    for batch in loader:
        imgs = batch["image"].to(device)
        incl_labels = batch["inclination_label_idx"].to(device)
        topic_labels = batch["topic_label_idx"].to(device)
        tokens = tokenizer(list(batch["text"]))

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(imgs, tokens, incl_labels, topic_labels)

        all_incl_preds.extend(out["incl_preds"].cpu().tolist())
        all_incl_labels.extend(incl_labels.cpu().tolist())
        all_topic_preds.extend(out["topic_preds"].cpu().tolist())
        all_topic_labels.extend(topic_labels.cpu().tolist())
        total_loss += out["loss"].item() * imgs.size(0)
        n += imgs.size(0)

    metrics = compute_metrics(all_incl_preds, all_incl_labels,
                               all_topic_preds, all_topic_labels)
    metrics["loss"] = total_loss / n
    return metrics


# ── main ─────────────────────────────────────────────────────────────────────
def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    os.environ.setdefault("WANDB_PROJECT", "politics")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = output_dir / "confusion_matrices"
    conf_dir.mkdir(exist_ok=True)

    # Datasets
    train_dataset = PoliticsDataset(args.data_root, "train",
                                    num_samples=args.num_samples, img_root=args.img_root)
    val_dataset = PoliticsDataset(args.data_root, "val",
                                  num_samples=args.num_samples, img_root=args.img_root)
    test_dataset = PoliticsDataset(args.data_root, "test",
                                   num_samples=args.num_samples, img_root=args.img_root)
    logger.info(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    train_bs = min(args.batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == "cuda")

    # Build frozen CLIP
    clip_model = CLIPBaseline(
        visual=build_timm_vit(arch="vit_base_patch16_224", global_pool="token",
                              use_sincos2d_pos=True),
        textual=TransformerTextEncoder(arch="L12_W512", vocab_size=49408,
                                       context_length=77),
        embed_dim=512,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    clip_model.load_state_dict(state, strict=False)

    model = PoliticsEuclideanClassifier(clip_model, hidden_dim=args.hidden_dim).to(device)
    tokenizer = Tokenizer()

    # Only learnable params in optimizer
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in learnable_params):,}")
    optimizer = torch.optim.AdamW(learnable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    wandb_cfg = {
        "model": "euclidean",
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint": args.checkpoint,
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_logger = WandbLogger(config=wandb_cfg, name=f"euclidean_politics_{ts}")

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_losses = run_epoch_train(model, train_loader, optimizer, scaler, scheduler,
                                       tokenizer, device, args.grad_clip)
        val_metrics = run_epoch_eval(model, val_loader, tokenizer, device)

        logger.info(f"Epoch {epoch}/{args.epochs}  "
                    f"train_loss={train_losses['loss']:.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}")
        log_metrics(val_metrics, epoch)

        # Save metrics JSON
        epoch_results = {
            "epoch": epoch, **train_losses, **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        with open(output_dir / f"metrics_epoch_{epoch:03d}.json", "w") as f:
            json.dump(epoch_results, f, indent=2)

        # WandB
        wlog = {
            "epoch": epoch,
            "train/loss": train_losses["loss"],
            "train/incl_loss": train_losses["incl_loss"],
            "train/topic_loss": train_losses["topic_loss"],
            "val/loss": val_metrics["loss"],
            "val/incl_accuracy": val_metrics["incl_accuracy"],
            "val/topic_accuracy": val_metrics["topic_accuracy"],
            "val/incl_macro_f1": val_metrics["incl_macro_f1"],
            "val/topic_macro_f1": val_metrics["topic_macro_f1"],
        }
        for name, acc in val_metrics["per_class_incl"].items():
            wlog[f"val/per_class_incl/{name}"] = acc
        for name, acc in val_metrics["per_class_topic"].items():
            wlog[f"val/per_class_topic/{name}"] = acc
        for name, acc in val_metrics["conditioned_topic_accuracy_given_incl"].items():
            wlog[f"val/conditioned_topic/{name}"] = acc
        wandb_logger.log(wlog, step=epoch)

        # Checkpoint
        combined_acc = val_metrics["incl_accuracy"] + val_metrics["topic_accuracy"]
        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "metrics": val_metrics}, output_dir / "best_checkpoint.pth")
            logger.info(f"  → New best checkpoint (combined acc={combined_acc:.2f}%)")

    # Final checkpoint
    torch.save({"epoch": args.epochs, "model": model.state_dict()},
               output_dir / "final_checkpoint.pth")

    logger.info(f"\nBest epoch: {best_epoch}  combined_acc: {best_val_acc:.2f}%")

    # Test-set evaluation using best checkpoint
    best_ckpt = torch.load(output_dir / "best_checkpoint.pth",
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    logger.info("Loaded best checkpoint — evaluating on test set")
    test_metrics = run_epoch_eval(model, test_loader, tokenizer, device)
    log_metrics(test_metrics, best_epoch, prefix="test")

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump({"best_epoch": best_epoch, **test_metrics}, f, indent=2)
    logger.info(f"Test metrics saved → {output_dir / 'test_metrics.json'}")

    wlog_test = {
        "test/loss": test_metrics["loss"],
        "test/incl_accuracy": test_metrics["incl_accuracy"],
        "test/topic_accuracy": test_metrics["topic_accuracy"],
        "test/incl_macro_f1": test_metrics["incl_macro_f1"],
        "test/topic_macro_f1": test_metrics["topic_macro_f1"],
    }
    for name, acc in test_metrics["per_class_incl"].items():
        wlog_test[f"test/per_class_incl/{name}"] = acc
    for name, acc in test_metrics["per_class_topic"].items():
        wlog_test[f"test/per_class_topic/{name}"] = acc
    for name, acc in test_metrics["conditioned_topic_accuracy_given_incl"].items():
        wlog_test[f"test/conditioned_topic/{name}"] = acc
    wandb_logger.log(wlog_test, step=args.epochs + 1)

    wandb_logger.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
