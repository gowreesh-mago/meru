# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Zero-shot emotion classification on Emoset using CLIPBaseline or MERU.
Uses hand-crafted emotion prompts - no training required.

Usage:
    python scripts/zero_shot_emotion.py \
        --checkpoint path/to/checkpoint.pth \
        --data-root /path/to/emoset \
        --split test \
        --model-type clip
"""

import argparse
from pathlib import Path

import torch
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.cuda import amp
from torch.utils.data import DataLoader, Subset

# Register Emoset dataset
import meru.emotion.dataset_integration  # noqa: F401
from emoset.Emoset import EmoSet
from meru.emotion.emotion_classes import EMOTION_CLASS_NAMES
from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder
from meru.models import CLIPBaseline, MERU
from meru.tokenizer import Tokenizer
from meru.utils.wandb_logger import WandbLogger

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

# fmt: off
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="Path to pretrained CLIP or MERU checkpoint.")
parser.add_argument("--data-root", required=True, help="Path to Emoset dataset root.")
parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to evaluate.")
parser.add_argument("--model-type", default="clip", choices=["clip", "meru"], help="Model type to load.")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation.")
parser.add_argument("--output-dir", default="output/zero_shot", help="Directory to save results.")
parser.add_argument("--num-samples", type=int, default=None, help="Limit dataset to N samples (useful for quick testing).")
# fmt: on


def build_model(model_type, checkpoint_path, device):
    """Build CLIP or MERU model from checkpoint."""
    logger.info(f"Building {model_type.upper()} model...")

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


def encode_text(model, texts, device):
    """Encode text prompts using the model."""
    tokenizer = Tokenizer()
    tokens = tokenizer(texts)

    with torch.no_grad():
        text_features = model.encode_text(tokens, project=True)

    return text_features


def zero_shot_predict(model, images, text_features, num_classes, device):
    """
    Make zero-shot predictions.

    Args:
        model: CLIP or MERU model
        images: Batch of images
        text_features: Precomputed text features (num_prompts, embed_dim)
        num_classes: Number of emotion classes
        device: torch device

    Returns:
        logits: (batch_size, num_classes)
    """
    with torch.no_grad():
        image_features = model.encode_image(images, project=True)

        # Compute similarity with all prompts
        if isinstance(model, MERU):
            # Use Lorentzian distance for MERU
            import meru.lorentz as L
            similarity = -L.pairwise_dist(
                image_features.unsqueeze(1),  # (B, 1, D)
                text_features.unsqueeze(0),   # (1, N, D)
                model.curv.exp()
            )
        else:
            # Use cosine similarity for CLIP
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features_norm.T

        # Average similarity across prompts for each class
        prompts_per_class = len(EMOTION_PROMPTS)
        similarity = similarity.view(images.size(0), num_classes, prompts_per_class)
        logits = similarity.mean(dim=-1)  # Average over prompts

    return logits


def evaluate_zero_shot(model, dataloader, device, model_type):
    """Evaluate model in zero-shot setting."""
    num_classes = len(EMOTION_NAMES)

    # Generate and encode text prompts
    logger.info("Generating and encoding text prompts...")
    all_prompts = generate_prompts(EMOTION_NAMES, EMOTION_PROMPTS)
    text_features = encode_text(model, all_prompts, device)
    logger.info(f"Encoded {len(all_prompts)} prompts ({num_classes} classes × {len(EMOTION_PROMPTS)} templates)")

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []

    logger.info(f"Evaluating on {len(dataloader.dataset)} samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["emotion_label_idx"]

            with amp.autocast(enabled=True):
                logits = zero_shot_predict(model, images, text_features, num_classes, device)

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

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
    wandb_config = {
        "model_type": f"{args.model_type.upper()} Zero-Shot",
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_prompts": len(EMOTION_PROMPTS),
    }
    wandb_logger = WandbLogger(config=wandb_config)

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
    import json
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
