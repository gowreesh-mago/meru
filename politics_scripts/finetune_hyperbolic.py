"""
Hyperbolic fine-tuning for political inclination and topic classification.

Architecture:
  - Frozen CLIPBaseline (image + text encoders)
  - Learnable hyperbolic projection layers (image_proj, text_proj)
  - Learnable scale params: visual_alpha, textual_alpha (log-scale, ≤ 0)
  - Learnable curvature: curv (log-scale, clamped in [log(0.1), log(10)])
  - Learnable class prototypes on the tangent space: incl_proto (2,512), topic_proto (20,512)

Entailment hierarchy: Topic → Inclination → Image

Loss:
  CE(incl_logits, incl_labels) + CE(topic_logits, topic_labels)
  + entail_weight * (topic→incl entailment + incl→image entailment)

Usage:
    python politics_scripts/finetune_hyperbolic.py \\
        --checkpoint /path/to/clip.pth \\
        --data-root /home/gmago/Emotions/politics_data \\
        --epochs 30 \\
        --lr 1e-3 \\
        --entail-weight 0.2 \\
        --no-wandb \\
        --num-samples 10
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
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

import meru.lorentz as L
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

_CURV_MIN = math.log(0.1)
_CURV_MAX = math.log(10.0)

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--data-root", required=True)
parser.add_argument("--split", default="test", choices=["train", "val", "test"])
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--entail-weight", type=float, default=0.2)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--output-dir", default="/home/gmago/Emotions/outputs/politics/hyperbolic")
parser.add_argument("--img-root", default=None,
                    help="Root dir for images (default: same as --data-root). "
                         "Set to /ssdstore/gowreesh/politics when CSVs and images differ.")
parser.add_argument("--num-samples", type=int, default=None)
parser.add_argument("--no-wandb", action="store_true")
parser.add_argument("--seed", type=int, default=42)


# ── model ────────────────────────────────────────────────────────────────────
class PoliticsHyperbolicClassifier(nn.Module):
    """
    Frozen CLIP + learnable hyperbolic projection + entailment hierarchy.

    Entailment: Topic → Inclination → Image
      - topic prototype's cone must contain the inclination prototype
      - inclination prototype's cone must contain the image embedding
    """

    def __init__(self, clip_model: CLIPBaseline, entail_weight: float = 0.2):
        super().__init__()
        self.clip = clip_model
        self.entail_weight = entail_weight

        # Freeze all CLIP params — frozen-param invariant
        for p in self.clip.parameters():
            p.requires_grad = False

        embed_dim = clip_model.embed_dim

        # Hyperbolic projection layers (tangent-space linear maps)
        self.image_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.text_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.image_proj.weight)
        nn.init.eye_(self.text_proj.weight)

        # Log-scale scalings (clamped ≤ 0 so norms don't explode)
        self.visual_alpha = nn.Parameter(torch.zeros(1))
        self.textual_alpha = nn.Parameter(torch.zeros(1))

        # Curvature in log-space (clamped to [log(0.1), log(10)])
        self.curv = nn.Parameter(torch.zeros(1))  # log(1.0) = 0

        # Class prototypes in tangent space. Larger init (0.1) ensures logit std ≈ 0.08
        # at init vs ≈ 0.02 with 0.02 scale, giving the CE gradient a clearer direction.
        self.incl_proto = nn.Parameter(torch.randn(NUM_INCL, embed_dim) * 0.1)
        self.topic_proto = nn.Parameter(torch.randn(NUM_TOPIC, embed_dim) * 0.1)

    def _to_hyperbolic(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Scale + exp_map0 to project tangent-space vectors onto the hyperboloid."""
        x = x * alpha.exp()
        return L.exp_map0(x, self.curv.exp())

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: list[torch.Tensor],
        incl_labels: torch.Tensor | None = None,
        topic_labels: torch.Tensor | None = None,
    ) -> dict:
        # Clamp hyperbolic params at top of forward (covers train + val)
        self.curv.data.clamp_(_CURV_MIN, _CURV_MAX)
        self.visual_alpha.data.clamp_(max=0.0)
        self.textual_alpha.data.clamp_(max=0.0)

        curv = self.curv.exp()

        # Frozen CLIP encodings (Euclidean, no projection)
        with torch.no_grad():
            img_eucl = self.clip.encode_image(images, project=False)    # (B, D)
            txt_eucl = self.clip.encode_text(text_tokens, project=False)  # (B, D)

        # Map to tangent space via learnable linear, then to hyperboloid
        img_hyp = self._to_hyperbolic(self.image_proj(img_eucl), self.visual_alpha)    # (B, D)
        txt_hyp = self._to_hyperbolic(self.text_proj(txt_eucl), self.textual_alpha)    # (B, D)

        # Project prototypes directly — no alpha scaling. textual_alpha is clamped ≤ 0
        # (exp(alpha) ≤ 1), which can only compress prototypes toward the origin and
        # weakens the CE gradient. Prototypes learn their own scale via gradient descent.
        incl_hyp = L.exp_map0(self.incl_proto, curv)   # (2, D)
        topic_hyp = L.exp_map0(self.topic_proto, curv)  # (20, D)

        # Classification via negative Lorentzian distance
        incl_logits = -L.pairwise_dist(img_hyp, incl_hyp, curv)    # (B, 2)
        topic_logits = -L.pairwise_dist(img_hyp, topic_hyp, curv)  # (B, 20)

        out = {
            "incl_logits": incl_logits,
            "topic_logits": topic_logits,
            "incl_preds": incl_logits.argmax(dim=-1),
            "topic_preds": topic_logits.argmax(dim=-1),
            "curv": curv.item(),
            "visual_alpha": self.visual_alpha.item(),
            "textual_alpha": self.textual_alpha.item(),
        }

        if incl_labels is not None and topic_labels is not None:
            incl_loss = F.cross_entropy(incl_logits, incl_labels)
            topic_loss = F.cross_entropy(topic_logits, topic_labels)

            # ── Entailment: Topic → Inclination → Image ──────────────────────
            # (a) topic prototype must entail inclination prototype
            t_batch = topic_hyp[topic_labels]   # (B, D)
            i_batch = incl_hyp[incl_labels]     # (B, D)
            angle_t2incl = L.oxy_angle(t_batch, i_batch, curv)
            aper_topic = L.half_aperture(t_batch, curv)
            loss_t2incl = torch.clamp(angle_t2incl - aper_topic, min=0).mean()

            # (b) inclination prototype must entail image
            angle_incl2img = L.oxy_angle(i_batch, img_hyp, curv)
            aper_incl = L.half_aperture(i_batch, curv)
            loss_incl2img = torch.clamp(angle_incl2img - aper_incl, min=0).mean()

            entail_loss = loss_t2incl + loss_incl2img
            total_loss = incl_loss + topic_loss + self.entail_weight * entail_loss

            # Entailment violation rates (diagnostic)
            with torch.no_grad():
                t2incl_viols = (angle_t2incl > aper_topic).float().mean().item()
                incl2img_viols = (angle_incl2img > aper_incl).float().mean().item()

            out.update({
                "loss": total_loss,
                "incl_loss": incl_loss,
                "topic_loss": topic_loss,
                "entail_loss": entail_loss,
                "loss_t2incl": loss_t2incl,
                "loss_incl2img": loss_incl2img,
                "t2incl_violations": t2incl_viols,
                "incl2img_violations": incl2img_viols,
            })

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


def log_metrics(metrics: dict, epoch: int, hyperbolic_diag: dict | None = None,
                prefix: str = "val") -> None:
    logger.info(f"[{prefix}] epoch={epoch}  "
                f"incl_acc={metrics['incl_accuracy']:.2f}%  "
                f"topic_acc={metrics['topic_accuracy']:.2f}%")
    if hyperbolic_diag:
        logger.info(f"  curv={hyperbolic_diag.get('curv', 0):.4f}  "
                    f"visual_alpha={hyperbolic_diag.get('visual_alpha', 0):.4f}  "
                    f"textual_alpha={hyperbolic_diag.get('textual_alpha', 0):.4f}")
        logger.info(f"  topic→incl violations={hyperbolic_diag.get('t2incl_violations', 0):.3f}  "
                    f"incl→img violations={hyperbolic_diag.get('incl2img_violations', 0):.3f}")
    logger.info("  Per-class inclination:")
    for n, a in metrics["per_class_incl"].items():
        logger.info(f"    {n}: {a:.2f}%")
    logger.info("  Conditioned topic accuracy (given true inclination):")
    for n, a in metrics["conditioned_topic_accuracy_given_incl"].items():
        logger.info(f"    {n}: {a:.2f}%")


# ── train / eval loops ────────────────────────────────────────────────────────
def run_epoch_train(model, loader, optimizer, scaler, scheduler, tokenizer, device,
                    grad_clip: float) -> dict:
    model.train()
    totals: dict[str, float] = {
        "loss": 0, "incl_loss": 0, "topic_loss": 0, "entail_loss": 0,
        "t2incl_violations": 0, "incl2img_violations": 0,
    }
    n = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        incl_labels = batch["inclination_label_idx"].to(device)
        topic_labels = batch["topic_label_idx"].to(device)
        tokens = tokenizer(list(batch["text"]))

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
        for key in totals:
            totals[key] += out.get(key, 0) * bs if isinstance(out.get(key, 0), float) \
                else out.get(key, torch.tensor(0.0)).item() * bs
        n += bs

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def run_epoch_eval(model, loader, tokenizer, device) -> tuple[dict, dict]:
    model.eval()
    all_incl_preds, all_incl_labels = [], []
    all_topic_preds, all_topic_labels = [], []
    hyp_diag: dict[str, list] = {
        "curv": [], "visual_alpha": [], "textual_alpha": [],
        "t2incl_violations": [], "incl2img_violations": [],
    }
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
        for k in hyp_diag:
            v = out.get(k, 0)
            hyp_diag[k].append(float(v) if not isinstance(v, float) else v)

    metrics = compute_metrics(all_incl_preds, all_incl_labels,
                               all_topic_preds, all_topic_labels)
    metrics["loss"] = total_loss / n
    diag = {k: float(np.mean(v)) for k, v in hyp_diag.items()}
    return metrics, diag


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

    model = PoliticsHyperbolicClassifier(clip_model, entail_weight=args.entail_weight)
    model = model.to(device)
    tokenizer = Tokenizer()

    # Two optimizer param groups: projection/prototypes at LR, hyperbolic params at 0.25×LR
    proj_params = list(model.image_proj.parameters()) + \
                  list(model.text_proj.parameters()) + \
                  [model.incl_proto, model.topic_proto]
    hyp_params = [model.curv, model.visual_alpha, model.textual_alpha]
    optimizer = torch.optim.AdamW(
        [{"params": proj_params, "lr": args.lr},
         {"params": hyp_params, "lr": args.lr * 0.25}],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,}")

    wandb_cfg = {
        "model": "hyperbolic",
        "entail_weight": args.entail_weight,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint": args.checkpoint,
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_logger = WandbLogger(config=wandb_cfg, name=f"hyperbolic_politics_{ts}")

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_losses = run_epoch_train(model, train_loader, optimizer, scaler, scheduler,
                                       tokenizer, device, args.grad_clip)
        val_metrics, hyp_diag = run_epoch_eval(model, val_loader, tokenizer, device)

        logger.info(f"Epoch {epoch}/{args.epochs}  "
                    f"train_loss={train_losses['loss']:.4f}  "
                    f"entail_loss={train_losses['entail_loss']:.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}")
        log_metrics(val_metrics, epoch, hyp_diag)

        epoch_results = {
            "epoch": epoch, **train_losses,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"hyp_{k}": v for k, v in hyp_diag.items()},
        }
        with open(output_dir / f"metrics_epoch_{epoch:03d}.json", "w") as f:
            json.dump(epoch_results, f, indent=2)

        wlog = {
            "epoch": epoch,
            "train/loss": train_losses["loss"],
            "train/incl_loss": train_losses["incl_loss"],
            "train/topic_loss": train_losses["topic_loss"],
            "train/entail_loss": train_losses["entail_loss"],
            "train/t2incl_violations": train_losses["t2incl_violations"],
            "train/incl2img_violations": train_losses["incl2img_violations"],
            "val/loss": val_metrics["loss"],
            "val/incl_accuracy": val_metrics["incl_accuracy"],
            "val/topic_accuracy": val_metrics["topic_accuracy"],
            "val/incl_macro_f1": val_metrics["incl_macro_f1"],
            "val/topic_macro_f1": val_metrics["topic_macro_f1"],
            "hyp/curv": hyp_diag["curv"],
            "hyp/visual_alpha": hyp_diag["visual_alpha"],
            "hyp/textual_alpha": hyp_diag["textual_alpha"],
            "hyp/t2incl_violations": hyp_diag["t2incl_violations"],
            "hyp/incl2img_violations": hyp_diag["incl2img_violations"],
        }
        for name, acc in val_metrics["per_class_incl"].items():
            wlog[f"val/per_class_incl/{name}"] = acc
        for name, acc in val_metrics["per_class_topic"].items():
            wlog[f"val/per_class_topic/{name}"] = acc
        for name, acc in val_metrics["conditioned_topic_accuracy_given_incl"].items():
            wlog[f"val/conditioned_topic/{name}"] = acc
        wandb_logger.log(wlog, step=epoch)

        combined_acc = val_metrics["incl_accuracy"] + val_metrics["topic_accuracy"]
        if combined_acc > best_val_acc:
            best_val_acc = combined_acc
            best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "metrics": val_metrics, "hyp_diag": hyp_diag},
                       output_dir / "best_checkpoint.pth")
            logger.info(f"  → New best checkpoint (combined acc={combined_acc:.2f}%)")

    torch.save({"epoch": args.epochs, "model": model.state_dict()},
               output_dir / "final_checkpoint.pth")
    logger.info(f"\nBest epoch: {best_epoch}  combined_acc: {best_val_acc:.2f}%")

    # Test-set evaluation using best checkpoint
    best_ckpt = torch.load(output_dir / "best_checkpoint.pth",
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    logger.info("Loaded best checkpoint — evaluating on test set")
    test_metrics, test_hyp_diag = run_epoch_eval(model, test_loader, tokenizer, device)
    log_metrics(test_metrics, best_epoch, test_hyp_diag, prefix="test")

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump({"best_epoch": best_epoch, **test_metrics, **test_hyp_diag}, f, indent=2)
    logger.info(f"Test metrics saved → {output_dir / 'test_metrics.json'}")

    wlog_test = {
        "test/loss": test_metrics["loss"],
        "test/incl_accuracy": test_metrics["incl_accuracy"],
        "test/topic_accuracy": test_metrics["topic_accuracy"],
        "test/incl_macro_f1": test_metrics["incl_macro_f1"],
        "test/topic_macro_f1": test_metrics["topic_macro_f1"],
        "test/curv": test_hyp_diag["curv"],
        "test/t2incl_violations": test_hyp_diag["t2incl_violations"],
        "test/incl2img_violations": test_hyp_diag["incl2img_violations"],
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
