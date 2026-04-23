# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Zero-shot emotion classification on Emoset using CLIPBaseline, MERU, EmotionCLIP, or HyCoClip.
Uses hand-crafted emotion prompts - no training required.

Usage:
    python scripts/zero_shot_emotion.py \
        --checkpoint path/to/checkpoint.pth \
        --data-root /path/to/emoset \
        --split test \
        --model-type clip

    python scripts/zero_shot_emotion.py \
        --data-root /path/to/emoset \
        --split test \
        --model-type emotionclip

    python scripts/zero_shot_emotion.py \
        --checkpoint checkpoints/hycoclip_vit_b.pth \
        --data-root /path/to/emoset \
        --split test \
        --model-type hycoclip
"""

import argparse
import importlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, Subset

# ── EmotionCLIP-V2 setup ──────────────────────────────────────────────────────
# EmotionCLIP.py uses relative paths for weights/pickles and imports sibling
# modules (VIT, Text_Encoder), so its directory must be both on sys.path and
# the CWD at import time. We restore CWD immediately after.
_EC_DIR = os.environ.get(
    "EMOTIONCLIP_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "EmotionCLIP-V2"),
)
# EmotionCLIP's VIT.py requires torch.nn.attention.sdpa_kernel (PyTorch >= 2.3).
_EC_TORCH_OK = hasattr(torch.nn, "attention") and hasattr(
    torch.nn.attention, "sdpa_kernel"
)
if os.path.isdir(_EC_DIR) and _EC_TORCH_OK:
    if _EC_DIR not in sys.path:
        sys.path.insert(0, _EC_DIR)
    # preprocess.pkl was pickled against the real openai/clip package which
    # has clip.clip._convert_image_to_rgb. CoOp ships its own clip package
    # that lacks this function; add it before pickle.load runs.
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
    EMOTIONCLIP_AVAILABLE = True
else:
    if os.path.isdir(_EC_DIR) and not _EC_TORCH_OK:
        print(
            f"⚠️  EmotionCLIP disabled: requires PyTorch >= 2.3 "
            f"(torch.nn.attention.sdpa_kernel), got {torch.__version__}"
        )
    EMOTIONCLIP_MODEL = None
    EMOTIONCLIP_PREPROCESS = None
    EMOTIONCLIP_TOKENIZER = None
    EMOTIONCLIP_AVAILABLE = False

# ── HyCoClip setup ────────────────────────────────────────────────────────────
_HC_DIR = os.environ.get(
    "HYCOCLIP_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "hycoclip"),
)
if os.path.isdir(_HC_DIR):
    if _HC_DIR not in sys.path:
        sys.path.insert(0, _HC_DIR)
    from hycoclip.models import HyCoCLIP  # noqa: E402
    from hycoclip.encoders.image_encoders import build_timm_vit as _hc_build_timm_vit  # noqa: E402
    from hycoclip.encoders.text_encoders import TransformerTextEncoder as _HCTextEncoder  # noqa: E402
    from hycoclip.tokenizer import Tokenizer as HyCoTokenizer  # noqa: E402
    import hycoclip.lorentz as HC_L  # noqa: E402
    HYCOCLIP_AVAILABLE = True
else:
    HyCoCLIP = None
    HyCoTokenizer = None
    HC_L = None
    HYCOCLIP_AVAILABLE = False

# Register Emoset dataset
import meru.emotion.dataset_integration  # noqa: F401
from emoset.Emoset import EmoSet
from meru.emotion.emotion_classes import EMOTION_CLASS_NAMES
from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder
from meru.models import CLIPBaseline, MERU
from meru.tokenizer import Tokenizer
import meru.lorentz as L
from meru.utils.wandb_logger import WandbLogger

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

# Emotion class names
EMOTION_NAMES = EMOTION_CLASS_NAMES

# Emotion prompts for zero-shot classification
EMOTION_PROMPTS = [
    "a photo of {}",
    "an image showing {}",
    "a picture depicting {}",
    "this evokes {}",
    "this makes me feel {}",
    "an image that conveys {}",
    "a photo expressing {}",
]

# Original EmotionCLIP paper prompt
EMOTIONCLIP_PROMPTS = [
    "This picture conveys a sense of {}",
]


class _EmoSetRaw(Dataset):
    """EmoSet variant that returns PIL-preprocessed images for external preprocessors."""

    def __init__(self, data_root: str, split: str, transform):
        base = EmoSet(data_root=data_root, num_emotion_classes=8, phase=split)
        self.data_store = base.data_store
        self.transform = transform

    def __len__(self):
        return len(self.data_store)

    def __getitem__(self, idx):
        emotion_label_idx, _, image_path, _ = self.data_store[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {"image": image, "emotion_label_idx": emotion_label_idx}


# fmt: off
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=None, help="Path to pretrained CLIP or MERU checkpoint (not required for --model-type emotionclip).")
parser.add_argument("--data-root", required=True, help="Path to Emoset dataset root.")
parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to evaluate.")
parser.add_argument("--model-type", default="clip", choices=["clip", "meru", "emotionclip", "hycoclip"], help="Model type to load.")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation.")
parser.add_argument("--output-dir", default="output/zero_shot", help="Directory to save results.")
parser.add_argument("--num-samples", type=int, default=None, help="Limit dataset to N samples (useful for quick testing).")
# fmt: on


def build_model(model_type, checkpoint_path, device):
    """Build CLIP, MERU, or EmotionCLIP model."""
    logger.info(f"Building {model_type.upper()} model...")

    if model_type == "emotionclip":
        if not EMOTIONCLIP_AVAILABLE:
            if os.path.isdir(_EC_DIR) and not _EC_TORCH_OK:
                raise RuntimeError(
                    f"EmotionCLIP requires PyTorch >= 2.3 (torch.nn.attention.sdpa_kernel), "
                    f"current version is {torch.__version__}."
                )
            raise RuntimeError(
                f"EmotionCLIP-V2 not found at '{_EC_DIR}'. "
                "Clone it with: git clone https://huggingface.co/jiangchengchengNLP/EmotionCLIP-V2 "
                "or set EMOTIONCLIP_DIR to its path."
            )
        model = EMOTIONCLIP_MODEL.to(device)
        model.eval()
        return model

    if model_type == "hycoclip":
        if not HYCOCLIP_AVAILABLE:
            raise RuntimeError(
                f"HyCoClip not found at '{_HC_DIR}'. "
                "Clone it: git clone https://github.com/PalAvik/hycoclip.git "
                "or set HYCOCLIP_DIR to its path."
            )
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required for hycoclip")
        model = HyCoCLIP(
            visual=_hc_build_timm_vit(
                arch="vit_base_patch16_224",
                global_pool="token",
                use_sincos2d_pos=True,
            ),
            textual=_HCTextEncoder(
                arch="L12_W512",
                vocab_size=49408,
                context_length=77,
            ),
            embed_dim=512,
            curv_init=1.0,
            learn_curv=True,
            entail_weight=0.2,
            use_boxes=True,
        )
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        # weights_only=False: hycoclip checkpoints contain non-tensor objects
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state, strict=False)
        model = model.to(device)
        model.eval()
        return model

    if checkpoint_path is None:
        raise ValueError("--checkpoint is required for clip and meru model types")

    if model_type == "clip":
        model = CLIPBaseline(
            visual=build_timm_vit(
                arch="vit_base_patch16_224",
                global_pool="token",
                use_sincos2d_pos=True,
            ),
            textual=TransformerTextEncoder(
                arch="L12_W512",
                vocab_size=49408,
                context_length=77,
            ),
            embed_dim=512,
        )
    elif model_type == "meru":
        model = MERU(
            visual=build_timm_vit(
                arch="vit_base_patch16_224",
                global_pool="token",
                use_sincos2d_pos=True,
            ),
            textual=TransformerTextEncoder(
                arch="L12_W512",
                vocab_size=49408,
                context_length=77,
            ),
            embed_dim=512,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    return model


def generate_prompts(emotion_names, prompt_templates):
    """Generate text prompts for each emotion."""
    prompts = []
    for emotion in emotion_names:
        for template in prompt_templates:
            prompts.append(template.format(emotion))
    return prompts


def encode_text(model, texts, device, model_type="clip"):
    """Encode text prompts using the model."""
    if model_type == "emotionclip":
        # context_length=57 = max_position_embeddings(77) - prompt_num(20),
        # which activates the prompt block in EmotionCLIP's encode_text
        tokens = EMOTIONCLIP_TOKENIZER(texts, context_length=57).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens).float()
        return text_features

    if model_type == "hycoclip":
        hc_tokenizer = HyCoTokenizer()
        tokens = hc_tokenizer(texts)
        with torch.no_grad():
            text_features = model.encode_text(tokens, project=True)
        return text_features

    tokenizer = Tokenizer()
    tokens = tokenizer(texts)
    with torch.no_grad():
        text_features = model.encode_text(tokens, project=True)
    return text_features


def zero_shot_predict(model, images, text_features, num_classes, device, model_type="clip", prompts_per_class=None):
    """
    Make zero-shot predictions.

    Args:
        model: CLIP, MERU, or EmotionCLIP model
        images: Batch of images
        text_features: Precomputed text features (num_classes * prompts_per_class, embed_dim)
        num_classes: Number of emotion classes
        device: torch device
        model_type: One of "clip", "meru", "emotionclip"
        prompts_per_class: Number of prompt templates used per class

    Returns:
        logits: (batch_size, num_classes)
    """
    if prompts_per_class is None:
        prompts_per_class = len(EMOTION_PROMPTS)

    with torch.no_grad():
        if model_type == "emotionclip":
            image_features = model.encode_image(images.to(model.dtype)).float()
        else:
            image_features = model.encode_image(images, project=True)

        if isinstance(model, MERU):
            similarity = -L.pairwise_dist(
                image_features,
                text_features,
                model.curv.exp()
            )
        elif model_type == "hycoclip":
            similarity = -HC_L.pairwise_dist(
                image_features,
                text_features,
                model.curv.exp()
            )
        else:
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features_norm.T

        similarity = similarity.view(images.size(0), num_classes, prompts_per_class)
        logits = similarity.mean(dim=-1)

    return logits


def evaluate_zero_shot(model, dataloader, device, model_type):
    """Evaluate model in zero-shot setting."""
    num_classes = len(EMOTION_NAMES)

    prompts = EMOTIONCLIP_PROMPTS if model_type == "emotionclip" else EMOTION_PROMPTS

    # Generate and encode text prompts
    logger.info("Generating and encoding text prompts...")
    all_prompts = generate_prompts(EMOTION_NAMES, prompts)
    text_features = encode_text(model, all_prompts, device, model_type=model_type)
    logger.info(f"Encoded {len(all_prompts)} prompts ({num_classes} classes × {len(prompts)} templates)")

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []

    total_batches = len(dataloader)
    log_interval = max(1, total_batches // 20)  # ~5% increments
    logger.info(f"Evaluating on {len(dataloader.dataset)} samples ({total_batches} batches)...")
    t_start = time.monotonic()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["emotion_label_idx"]

            with torch.amp.autocast("cuda", enabled=True):
                logits = zero_shot_predict(
                    model, images, text_features, num_classes, device,
                    model_type=model_type, prompts_per_class=len(prompts)
                )

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.monotonic() - t_start
                pct = (batch_idx + 1) / total_batches * 100
                eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
                logger.info(
                    f"[{model_type}] batch {batch_idx + 1}/{total_batches} "
                    f"({pct:.0f}%) | elapsed {elapsed:.0f}s | ETA {eta:.0f}s"
                )

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

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_accuracy": {name: acc for name, acc in zip(EMOTION_NAMES, per_class_acc)},
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
    }


def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading Emoset {args.split} split from {args.data_root}...")
    if args.model_type == "emotionclip":
        dataset = _EmoSetRaw(
            data_root=args.data_root,
            split=args.split,
            transform=EMOTIONCLIP_PREPROCESS,
        )
    else:
        dataset = EmoSet(
            data_root=args.data_root,
            num_emotion_classes=8,
            phase=args.split,
        )

    # Limit dataset size if --num-samples is specified
    if args.num_samples is not None:
        logger.info(f"Limiting dataset to {args.num_samples} samples for quick testing")
        indices = list(range(min(args.num_samples, len(dataset))))
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Loaded {len(dataset)} samples")

    # Build model
    model = build_model(args.model_type, args.checkpoint, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {total_params:,}")

    # Setup wandb (optional, controlled by environment variables)
    active_prompts = EMOTIONCLIP_PROMPTS if args.model_type == "emotionclip" else EMOTION_PROMPTS
    wandb_config = {
        "model_type": f"{args.model_type.upper()} Zero-Shot",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_prompts": len(active_prompts),
        "prompt_templates": active_prompts,
    }
    run_name = f"{args.model_type}_zero_shot_{args.split}"
    wandb_logger = WandbLogger(config=wandb_config, name=run_name)

    # Evaluate
    logger.info(f"Running zero-shot evaluation on {args.split} split...")
    results = evaluate_zero_shot(model, dataloader, device, args.model_type)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info(f"ZERO-SHOT EVALUATION RESULTS ({args.split} split, {args.model_type.upper()})")
    logger.info("=" * 80)
    logger.info(f"Overall Accuracy: {results['accuracy']:.2f}%")
    logger.info(f"Macro F1: {results['macro_f1']:.2f}%")
    logger.info(f"Weighted F1: {results['weighted_f1']:.2f}%")
    logger.info("\nPer-class Accuracy:")
    for emotion, acc in results["per_class_accuracy"].items():
        logger.info(f"  {emotion:15s}: {acc:.2f}%")

    logger.info("\nClassification Report:")
    logger.info(results["classification_report"])

    # Save results (JSON)
    results_file = output_dir / f"{args.model_type}_zero_shot_{args.split}.json"
    with open(results_file, "w") as f:
        json.dump({
            "model_type": args.model_type,
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "accuracy": results["accuracy"],
            "macro_f1": results["macro_f1"],
            "weighted_f1": results["weighted_f1"],
            "per_class_accuracy": results["per_class_accuracy"],
        }, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    # Save results (text format for comparison script)
    text_results_file = output_dir / f"eval_results_{args.split}.txt"
    with open(text_results_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"ZERO-SHOT EVALUATION RESULTS ({args.split} split, {args.model_type.upper()})\n")
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

    logger.info(f"Text results saved to: {text_results_file}")

    # Log to wandb
    wandb_metrics = {
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
    }
    # Add per-class accuracies
    for emotion, acc in results["per_class_accuracy"].items():
        wandb_metrics[f"accuracy/{emotion}"] = acc

    wandb_logger.log(wandb_metrics)
    wandb_logger.finish()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
