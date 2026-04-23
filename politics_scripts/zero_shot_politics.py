"""
Zero-shot CLIP/MERU evaluation on the politics dataset.

Two modes:
  prompts   -- Zero-shot classification via configurable text prompts.
               Separately predicts inclination (2 classes) and topic (20 classes).
               Ignores headline and article text; uses label/topic names in prompts.

  retrieval -- Text-image retrieval using article content_text.
               Ignores inclination/topic labels; evaluates R@1, R@5, R@10
               for all N image-text pairs.

Usage:
    python politics_scripts/zero_shot_politics.py \\
        --checkpoint /path/to/clip.pth \\
        --data-root /home/gmago/Emotions/politics_data \\
        --mode prompts \\
        --model-type clip \\
        --split test \\
        --no-wandb

    python politics_scripts/zero_shot_politics.py \\
        --checkpoint /path/to/clip.pth \\
        --data-root /home/gmago/Emotions/politics_data \\
        --mode retrieval \\
        --split test \\
        --no-wandb
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
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

# ── project root on path ────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer
import meru.lorentz as L
from meru.utils.wandb_logger import WandbLogger
from politics_scripts.politics_dataset import (
    INCLINATION_CLASSES,
    TOPIC_CLASSES,
    PoliticsDataset,
)

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--data-root", required=True)
parser.add_argument("--split", default="test", choices=["train", "val", "test"])
parser.add_argument("--model-type", default="clip", choices=["clip", "meru"])
parser.add_argument("--mode", required=True, choices=["prompts", "retrieval"])
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--output-dir", default="/home/gmago/Emotions/outputs/politics/zero_shot")
parser.add_argument("--img-root", default=None,
                    help="Root dir for images (default: same as --data-root). "
                         "Set to /ssdstore/gowreesh/politics when CSVs and images differ.")
parser.add_argument("--num-samples", type=int, default=None,
                    help="Limit to first N samples for smoke testing.")
parser.add_argument("--no-wandb", action="store_true")
# prompt-mode options
parser.add_argument("--incl-template", default="a {label} political viewpoint",
                    help="Template for inclination prompts. Use {label} placeholder.")
parser.add_argument("--topic-template", default="a political image about {topic}",
                    help="Template for topic prompts. Use {topic} placeholder.")
parser.add_argument("--seed", type=int, default=42)


# ── model builder ────────────────────────────────────────────────────────────
def build_model(model_type: str, checkpoint_path: str, device: torch.device) -> CLIPBaseline:
    if model_type == "clip":
        model = CLIPBaseline(
            visual=build_timm_vit(arch="vit_base_patch16_224", global_pool="token",
                                  use_sincos2d_pos=True),
            textual=TransformerTextEncoder(arch="L12_W512", vocab_size=49408,
                                           context_length=77),
            embed_dim=512,
        )
    else:
        model = MERU(
            visual=build_timm_vit(arch="vit_base_patch16_224", global_pool="token",
                                  use_sincos2d_pos=True),
            textual=TransformerTextEncoder(arch="L12_W512", vocab_size=49408,
                                           context_length=77),
            embed_dim=512,
        )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


# ── similarity helpers ────────────────────────────────────────────────────────
def cosine_sim(img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
    """(B, D) x (C, D) → (B, C)"""
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)
    return img @ txt.T


def lorentz_sim(img: torch.Tensor, txt: torch.Tensor, curv: torch.Tensor) -> torch.Tensor:
    """(B, D) x (C, D) → (B, C)  (negative geodesic distance)"""
    return -L.pairwise_dist(img, txt, curv)


def encode_texts(model: CLIPBaseline, texts: list[str],
                 tokenizer: Tokenizer, device: torch.device,
                 batch_size: int = 256) -> torch.Tensor:
    """Encode a list of strings → (N, D) normalised embeddings."""
    all_feats = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start: start + batch_size]
        tokens = tokenizer(chunk)
        with torch.no_grad():
            feats = model.encode_text(tokens, project=True)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def encode_images_batched(model: CLIPBaseline, dataloader: DataLoader,
                          device: torch.device) -> torch.Tensor:
    """Encode all images in the dataloader → (N, D) normalised embeddings."""
    all_feats = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                feats = model.encode_image(imgs, project=True)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


# ── metrics ──────────────────────────────────────────────────────────────────
def compute_classification_metrics(
    incl_preds: list[int], incl_labels: list[int],
    topic_preds: list[int], topic_labels: list[int],
) -> dict:
    cm_incl = confusion_matrix(incl_labels, incl_preds, labels=list(range(len(INCLINATION_CLASSES))))
    per_incl = (cm_incl.diagonal() / cm_incl.sum(axis=1).clip(min=1) * 100).tolist()

    cm_topic = confusion_matrix(topic_labels, topic_preds, labels=list(range(len(TOPIC_CLASSES))))
    per_topic = (cm_topic.diagonal() / cm_topic.sum(axis=1).clip(min=1) * 100).tolist()

    conditioned: dict[str, float] = {}
    incl_arr = np.array(incl_labels)
    topic_pred_arr = np.array(topic_preds)
    topic_label_arr = np.array(topic_labels)
    for i, name in enumerate(INCLINATION_CLASSES):
        mask = incl_arr == i
        if mask.sum() > 0:
            conditioned[name] = float(accuracy_score(topic_label_arr[mask], topic_pred_arr[mask]) * 100)

    return {
        "incl_accuracy": float(accuracy_score(incl_labels, incl_preds) * 100),
        "topic_accuracy": float(accuracy_score(topic_labels, topic_preds) * 100),
        "incl_macro_f1": float(f1_score(incl_labels, incl_preds, average="macro") * 100),
        "topic_macro_f1": float(f1_score(topic_labels, topic_preds, average="macro") * 100),
        "incl_weighted_f1": float(f1_score(incl_labels, incl_preds, average="weighted") * 100),
        "topic_weighted_f1": float(f1_score(topic_labels, topic_preds, average="weighted") * 100),
        "per_class_incl": {INCLINATION_CLASSES[i]: per_incl[i] for i in range(len(INCLINATION_CLASSES))},
        "per_class_topic": {TOPIC_CLASSES[i]: per_topic[i] for i in range(len(TOPIC_CLASSES))},
        "conditioned_topic_accuracy_given_incl": conditioned,
    }


def compute_retrieval_metrics(sim_matrix: torch.Tensor, ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """
    sim_matrix: (N, N) float — sim_matrix[i, j] = similarity(query_i, key_j).
    Ground truth: diagonal (i matches i).
    Returns R@k, mean rank, median rank for each k.
    """
    N = sim_matrix.shape[0]
    # Rank of the correct match for each query (1-based)
    ranks = (sim_matrix.argsort(dim=1, descending=True) == torch.arange(N).unsqueeze(1)).float()
    # ranks[i, j] = 1 if ground truth is at position j (0-indexed)
    rank_positions = ranks.argmax(dim=1).float() + 1  # 1-indexed

    result = {}
    for k in ks:
        result[f"R@{k}"] = float((rank_positions <= k).float().mean().item() * 100)
    result["mean_rank"] = float(rank_positions.mean().item())
    result["median_rank"] = float(rank_positions.median().item())
    return result


# ── mode: prompts ─────────────────────────────────────────────────────────────
def run_prompts_mode(model: CLIPBaseline, dataloader: DataLoader,
                     tokenizer: Tokenizer, args, device: torch.device) -> dict:
    logger.info("Mode: prompts — zero-shot classification via configurable prompts")

    # Build class text embeddings
    incl_texts = [args.incl_template.format(label=c) for c in INCLINATION_CLASSES]
    topic_texts = [args.topic_template.format(topic=t) for t in TOPIC_CLASSES]
    logger.info(f"Inclination prompts: {incl_texts}")
    logger.info(f"Topic prompts (first 3): {topic_texts[:3]} ...")

    incl_feats = encode_texts(model, incl_texts, tokenizer, device).to(device)   # (2, D)
    topic_feats = encode_texts(model, topic_texts, tokenizer, device).to(device)  # (20, D)

    is_meru = isinstance(model, MERU)
    curv = model.curv.exp() if is_meru else None

    all_incl_preds, all_incl_labels = [], []
    all_topic_preds, all_topic_labels = [], []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                img_feats = model.encode_image(imgs, project=True)

            if is_meru:
                incl_sim = lorentz_sim(img_feats, incl_feats, curv)
                topic_sim = lorentz_sim(img_feats, topic_feats, curv)
            else:
                incl_sim = cosine_sim(img_feats, incl_feats)
                topic_sim = cosine_sim(img_feats, topic_feats)

            all_incl_preds.extend(incl_sim.argmax(dim=-1).cpu().tolist())
            all_topic_preds.extend(topic_sim.argmax(dim=-1).cpu().tolist())
            all_incl_labels.extend(batch["inclination_label_idx"].tolist())
            all_topic_labels.extend(batch["topic_label_idx"].tolist())

    metrics = compute_classification_metrics(
        all_incl_preds, all_incl_labels, all_topic_preds, all_topic_labels)
    return metrics


# ── mode: retrieval ───────────────────────────────────────────────────────────
def run_retrieval_mode(model: CLIPBaseline, dataset, dataloader: DataLoader,
                       tokenizer: Tokenizer, args, device: torch.device) -> dict:
    logger.info("Mode: retrieval — text-image retrieval on all N pairs")

    # Encode all images
    logger.info("Encoding images ...")
    img_embs = encode_images_batched(model, dataloader, device)  # (N, D)
    logger.info(f"Image embeddings: {img_embs.shape}")

    # Encode all texts in batches
    logger.info("Encoding texts ...")
    all_texts = [dataset[i]["text"] for i in range(len(dataset))]
    txt_embs = encode_texts(model, all_texts, tokenizer, device, batch_size=256)  # (N, D)
    logger.info(f"Text embeddings: {txt_embs.shape}")

    N = img_embs.shape[0]
    img_embs = F.normalize(img_embs, dim=-1)
    txt_embs = F.normalize(txt_embs, dim=-1)

    is_meru = isinstance(model, MERU)
    curv_val = model.curv.exp().item() if is_meru else None

    # Compute NxN similarity in chunks to avoid OOM
    chunk_size = 512
    logger.info(f"Computing {N}x{N} similarity matrix in chunks of {chunk_size} ...")

    # Image→Text: rows=images, cols=texts
    i2t_sim = torch.zeros(N, N)
    img_embs_dev = img_embs.to(device)
    txt_embs_dev = txt_embs.to(device)
    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            q = img_embs_dev[start:end]
            if is_meru:
                i2t_sim[start:end] = lorentz_sim(q, txt_embs_dev,
                                                   model.curv.exp()).cpu()
            else:
                i2t_sim[start:end] = (q @ txt_embs_dev.T).cpu()

    # Text→Image: transpose
    t2i_sim = i2t_sim.T

    logger.info("Computing retrieval metrics ...")
    i2t_metrics = compute_retrieval_metrics(i2t_sim)
    t2i_metrics = compute_retrieval_metrics(t2i_sim)

    return {
        "image_to_text": {f"i2t_{k}": v for k, v in i2t_metrics.items()},
        "text_to_image": {f"t2i_{k}": v for k, v in t2i_metrics.items()},
    }


# ── logging helpers ──────────────────────────────────────────────────────────
def print_classification_results(metrics: dict, split: str, model_type: str) -> None:
    logger.info("\n" + "=" * 70)
    logger.info(f"ZERO-SHOT PROMPTS RESULTS  [{split}  {model_type.upper()}]")
    logger.info("=" * 70)
    logger.info(f"Inclination accuracy : {metrics['incl_accuracy']:.2f}%")
    logger.info(f"Topic accuracy       : {metrics['topic_accuracy']:.2f}%")
    logger.info(f"Inclination macro-F1 : {metrics['incl_macro_f1']:.2f}%")
    logger.info(f"Topic macro-F1       : {metrics['topic_macro_f1']:.2f}%")
    logger.info("\nPer-class inclination accuracy:")
    for name, acc in metrics["per_class_incl"].items():
        logger.info(f"  {name:12s}: {acc:.2f}%")
    logger.info("\nPer-class topic accuracy:")
    for name, acc in metrics["per_class_topic"].items():
        logger.info(f"  {name:25s}: {acc:.2f}%")
    logger.info("\nConditioned topic accuracy (given true inclination):")
    for name, acc in metrics["conditioned_topic_accuracy_given_incl"].items():
        logger.info(f"  {name:12s}: {acc:.2f}%")


def print_retrieval_results(metrics: dict, split: str, model_type: str) -> None:
    logger.info("\n" + "=" * 70)
    logger.info(f"ZERO-SHOT RETRIEVAL RESULTS  [{split}  {model_type.upper()}]")
    logger.info("=" * 70)
    for direction, sub in [("Image→Text", "image_to_text"), ("Text→Image", "text_to_image")]:
        logger.info(f"\n{direction}:")
        for k, v in metrics[sub].items():
            logger.info(f"  {k:15s}: {v:.2f}")


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

    dataset = PoliticsDataset(
        data_root=args.data_root,
        split=args.split,
        num_samples=args.num_samples,
        img_root=args.img_root,
    )
    logger.info(f"Dataset: {len(dataset)} samples  ({args.split})")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model_type, args.checkpoint, device)
    tokenizer = Tokenizer()

    wandb_cfg = {
        "model_type": args.model_type,
        "mode": args.mode,
        "split": args.split,
        "num_samples": args.num_samples,
        "incl_template": getattr(args, "incl_template", ""),
        "topic_template": getattr(args, "topic_template", ""),
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_logger = WandbLogger(config=wandb_cfg,
                               name=f"zero_shot_{args.mode}_{args.model_type}_{args.split}_{ts}")

    if args.mode == "prompts":
        metrics = run_prompts_mode(model, dataloader, tokenizer, args, device)
        print_classification_results(metrics, args.split, args.model_type)

        out_path = output_dir / f"{args.model_type}_zero_shot_prompts_{args.split}.json"
        with open(out_path, "w") as f:
            json.dump({"mode": "prompts", "split": args.split,
                       "model_type": args.model_type, **metrics}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

        wandb_flat = {
            "incl_accuracy": metrics["incl_accuracy"],
            "topic_accuracy": metrics["topic_accuracy"],
            "incl_macro_f1": metrics["incl_macro_f1"],
            "topic_macro_f1": metrics["topic_macro_f1"],
        }
        for name, acc in metrics["per_class_incl"].items():
            wandb_flat[f"per_class_incl/{name}"] = acc
        for name, acc in metrics["per_class_topic"].items():
            wandb_flat[f"per_class_topic/{name}"] = acc
        for name, acc in metrics["conditioned_topic_accuracy_given_incl"].items():
            wandb_flat[f"conditioned_topic/{name}"] = acc
        wandb_logger.log(wandb_flat)

    else:
        metrics = run_retrieval_mode(model, dataset, dataloader, tokenizer, args, device)
        print_retrieval_results(metrics, args.split, args.model_type)

        out_path = output_dir / f"{args.model_type}_zero_shot_retrieval_{args.split}.json"
        with open(out_path, "w") as f:
            json.dump({"mode": "retrieval", "split": args.split,
                       "model_type": args.model_type, **metrics}, f, indent=2)
        logger.info(f"Results saved → {out_path}")

        wandb_flat = {}
        for sub in ("image_to_text", "text_to_image"):
            for k, v in metrics[sub].items():
                wandb_flat[k] = v
        wandb_logger.log(wandb_flat)

    wandb_logger.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
