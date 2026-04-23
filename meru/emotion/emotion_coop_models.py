# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Emotion classification models combining CoOp's prompt learning with MERU models.

This module provides thin wrappers that integrate:
- CoOp's PromptLearner and TextEncoder (from /CoOp/trainers/coop.py)
- MERU's CLIPBaseline and MERU models (from meru/models.py)
- Entailment loss for MERU variant (Emotion → Image hierarchy)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add CoOp to path
coop_path = Path(__file__).parent.parent.parent / "CoOp"
if str(coop_path) not in sys.path:
    sys.path.insert(0, str(coop_path))

from trainers.coop import PromptLearner
from trainers.coop import TextEncoder as _CoOpTextEncoder  # noqa: E402


class TextEncoder(_CoOpTextEncoder):
    """
    Fixed TextEncoder that properly handles text_projection and device movement.

    The original CoOp TextEncoder caches text_projection at init, which breaks
    gradient flow when it's a transposed view that gets moved to different devices.

    This version stores the underlying MERU CLIP model (nn.Module) so that device
    movement works properly, and accesses text_projection dynamically.
    """

    def __init__(self, clip_model):
        # Don't call super().__init__() to avoid caching text_projection
        nn.Module.__init__(self)

        # Store underlying MERU CLIP model (nn.Module) for proper device movement
        # clip_model is PseudoCLIPModel (not nn.Module), so we extract the real model
        # Handle both _meru_clip (CLIPCoOpEmotion) and _meru (MERUCoOpEmotion)
        if hasattr(clip_model, '_meru_clip'):
            self._underlying_model = clip_model._meru_clip
        elif hasattr(clip_model, '_meru'):
            self._underlying_model = clip_model._meru
        else:
            raise AttributeError("clip_model must have _meru_clip or _meru attribute")

        # Store references to components (these are already nn.Module/Parameter/Tensor)
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype
        # DO NOT cache text_projection here!

        # Store the causal attention mask from the text encoder so the transformer
        # runs with the same mask used during MERU pre-training.
        if hasattr(clip_model, '_meru_clip'):
            self.register_buffer("attn_mask", clip_model._meru_clip.textual.attn_mask)
        elif hasattr(clip_model, '_meru'):
            self.register_buffer("attn_mask", clip_model._meru.textual.attn_mask)

    def forward(self, prompts, tokenized_prompts):
        # MERU's _TransformerBlock uses batch_first=True — inputs stay in NLD format.
        # The original CoOp TextEncoder permuted to LND for OpenAI's sequence-first
        # transformer, but that scrambles the sequence/class dims here and kills
        # the gradient path from EOT output back to ctx positions.
        x = prompts + self.positional_embedding.type(self.dtype)
        seq_len = x.shape[1]
        attn_mask = self.attn_mask[:seq_len, :seq_len]
        x = self.transformer(x, attn_mask)
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # Access text_projection dynamically from the underlying CLIP model
        # This ensures it's always on the correct device after .to() calls
        text_projection = self._underlying_model.textual_proj.weight.T
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ text_projection

        return x

from meru import lorentz as L  # noqa: E402

# ---------------------------------------------------------------------------
# EmotionCLIP training wrapper
# ---------------------------------------------------------------------------

_EMOTIONCLIP_TRAINABLE_KEYS = ("prefix", "prompt", "ln")


class EmotionCLIPEmotion(nn.Module):
    """
    EmotionCLIP wrapped for supervised emotion classification training.

    Uses EmotionCLIP's hybrid fine-tuning recipe: only LayerNorm params
    (any name containing "ln"), prefix embeddings ("prefix"), and soft prompt
    embeddings ("prompt") are trainable — everything else is frozen.

    Args:
        emotionclip_model: The loaded EmotionCLIP CLIP instance (float16, cuda).
        tokenizer_fn: EmotionCLIP's tokenizer callable (clip.tokenize).
        emotion_names: List of emotion class names (length = num_classes).
    """

    PROMPT_TEMPLATE = "This picture conveys a sense of {}"
    # context_length=57 = 77 - 20 (prompt_num); this activates Prompt_block
    # in CLIP.encode_text (see EmotionCLIP.py:62).
    CONTEXT_LENGTH = 57

    def __init__(self, emotionclip_model, tokenizer_fn, emotion_names: list[str]):
        super().__init__()
        self.model = emotionclip_model
        self.emotion_names = emotion_names

        # Apply the same freeze recipe used during EmotionCLIP pre-training:
        # train only prefix / prompt / ln params, freeze everything else.
        for name, param in self.model.named_parameters():
            param.requires_grad = any(k in name for k in _EMOTIONCLIP_TRAINABLE_KEYS)

        # logit_scale has requires_grad=False already in EmotionCLIP's Config,
        # but assert it explicitly to uphold the frozen-param invariant.
        self.model.logit_scale.requires_grad = False

        # Precompute fixed text tokens at init — shape (num_classes, CONTEXT_LENGTH).
        text_list = [self.PROMPT_TEMPLATE.format(e) for e in emotion_names]
        text_tokens = tokenizer_fn(text_list, context_length=self.CONTEXT_LENGTH)
        self.register_buffer("text_tokens", text_tokens)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None):
        """
        Args:
            images: (B, 3, 224, 224) — in EmoSet's ImageNet normalization; the
                EmotionCLIP preprocessor must be applied upstream (via _EmoSetRaw).
            labels: (B,) integer class indices.

        Returns:
            dict with keys: logits, preds, loss (if labels given).
        """
        device = self.text_tokens.device
        images = images.to(device=device, dtype=self.model.dtype)

        logits, _ = self.model(images, self.text_tokens)
        logits = logits.float()  # upcast from fp16 for numerically stable CE loss

        preds = logits.argmax(dim=-1)
        output = {"logits": logits, "preds": preds}

        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels.to(device))

        return output


class TransformerWrapper(nn.Module):
    """Wrapper to make ModuleList of resblocks callable as a single transformer.

    IMPORTANT: MERU's _TransformerBlock uses batch_first=True, so inputs must
    be in NLD format (batch, seq, dim). Do NOT permute to LND before calling.
    """

    def __init__(self, resblocks):
        super().__init__()
        self.resblocks = resblocks

    def forward(self, x, attn_mask=None):
        for block in self.resblocks:
            x = block(x, attn_mask)
        return x


class CLIPCoOpEmotion(nn.Module):
    """
    CLIP with learnable emotion prompts (CoOp-style) for emotion classification.

    This model freezes the CLIP visual and text encoders, and only learns
    context vectors for emotion prompts. Operates in Euclidean space with
    cosine similarity.

    Args:
        clip_model: MERU's CLIPBaseline model instance
        emotion_names: List of emotion class names (e.g., ["amusement", "anger", ...])
        n_ctx: Number of learnable context tokens (default: 16)
        ctx_init: Optional initialization words for context (e.g., "a photo of")
        class_token_position: Position of class token: "end", "middle", or "front"
    """

    def __init__(
        self,
        clip_model,
        emotion_names: list[str],
        n_ctx: int = 16,
        ctx_init: str = "",
        class_token_position: str = "end",
        csc: bool = False,
    ):
        super().__init__()

        # Create a pseudo CLIP model structure that CoOp's components expect
        # CoOp PromptLearner needs: token_embedding, ln_final, visual, dtype
        class PseudoCLIPModel:
            def __init__(self, meru_clip):
                self._meru_clip = meru_clip  # Store model reference
                self.token_embedding = meru_clip.textual.token_embed
                self.ln_final = meru_clip.textual.ln_final
                self.visual = meru_clip.visual
                self.dtype = torch.float32  # MERU uses float32 by default
                self.positional_embedding = meru_clip.textual.posit_embed
                self.transformer = TransformerWrapper(meru_clip.textual.resblocks)

            @property
            def text_projection(self):
                # Return transposed projection - always gets current device
                return self._meru_clip.textual_proj.weight.T

        # Create wrapper for visual encoder to add input_resolution attribute
        class VisualWrapper:
            def __init__(self, visual):
                self._visual = visual
                self.input_resolution = 224  # MERU ViT-Base uses 224x224 images

            def __getattr__(self, name):
                return getattr(self._visual, name)

        pseudo_clip = PseudoCLIPModel(clip_model)
        pseudo_clip.visual = VisualWrapper(pseudo_clip.visual)

        # Create config for PromptLearner
        cfg = SimpleNamespace(
            TRAINER=SimpleNamespace(
                COOP=SimpleNamespace(
                    N_CTX=n_ctx,
                    CSC=csc,  # Class-specific context
                    CTX_INIT=ctx_init,
                    CLASS_TOKEN_POSITION=class_token_position,
                )
            ),
            INPUT=SimpleNamespace(SIZE=[224, 224]),  # Image size
        )

        # Freeze CLIP encoders BEFORE creating CoOp components
        # This ensures only prompt tokens remain trainable
        for param in clip_model.visual.parameters():
            param.requires_grad = False
        for param in clip_model.visual_proj.parameters():
            param.requires_grad = False
        for param in clip_model.textual.parameters():
            param.requires_grad = False
        for param in clip_model.textual_proj.parameters():
            param.requires_grad = False

        # Initialize CoOp components
        self.prompt_learner = PromptLearner(cfg, emotion_names, pseudo_clip)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(pseudo_clip)

        # Store CLIP components
        self.image_encoder = clip_model.visual
        self.visual_proj = clip_model.visual_proj
        self.logit_scale = clip_model.logit_scale
        # logit_scale is not in the optimizer; freeze it so it is excluded from
        # clip_grad_norm_(model.parameters()) and optimizer.zero_grad() gaps.
        # Leaving it unfrozen causes its still-scaled grad (≈65536×true_grad)
        # to dominate the total norm, making clip_coeff≈0 and zeroing ctx.grad.
        self.logit_scale.requires_grad = False

        # Only prompt_learner parameters are trainable
        self.dtype = torch.float32

        # Store reference to the CLIP model so we can access text_projection property
        self._clip_model_ref = clip_model

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature space."""
        image_feats = self.image_encoder(images.type(self.dtype))
        image_feats = self.visual_proj(image_feats)
        return image_feats

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None):
        """
        Forward pass for emotion classification.

        Args:
            images: Batch of images (B, 3, 224, 224)
            labels: Batch of emotion labels (B,) - only used for computing loss

        Returns:
            Dictionary with keys:
            - logits: Classification logits (B, num_emotions)
            - loss: Cross-entropy loss (if labels provided)
        """
        # Get image features
        image_features = self.encode_image(images)

        # Get text features from learnable prompts
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts.to(prompts.device)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # DEBUG: Check if text features are collapsing (all similar)
        # Compute pairwise similarity between text features
        text_similarity = text_features @ text_features.t()
        # If all text features are very similar, similarities will be close to 1
        # Store for debugging (off-diagonal values should be < 1)
        output_debug = {
            "text_similarity_mean": text_similarity.mean().item(),
            "text_similarity_std": text_similarity.std().item(),
        }

        # Compute similarity logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        # Compute predictions once here to avoid recomputation in training script
        preds = logits.argmax(dim=-1)

        output = {"logits": logits, "preds": preds, "debug": output_debug}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        return output


class MERUCoOpEmotion(nn.Module):
    """
    MERU with learnable emotion prompts (CoOp-style) and entailment loss.

    This model extends CLIPCoOpEmotion to operate in hyperbolic space using
    MERU's Lorentzian geometry. It enforces the hierarchy: Emotion text → Image
    through entailment loss.

    Args:
        meru_model: MERU model instance
        emotion_names: List of emotion class names
        n_ctx: Number of learnable context tokens (default: 16)
        ctx_init: Optional initialization words for context
        class_token_position: Position of class token
        entail_weight: Weight for entailment loss (default: 0.2)
    """

    def __init__(
        self,
        meru_model,
        emotion_names: list[str],
        n_ctx: int = 16,
        ctx_init: str = "",
        class_token_position: str = "end",
        entail_weight: float = 0.2,
        csc: bool = False,
    ):
        super().__init__()

        # Create pseudo CLIP model for CoOp components
        class PseudoCLIPModel:
            def __init__(self, meru):
                self._meru = meru  # Store model reference
                self.token_embedding = meru.textual.token_embed
                self.ln_final = meru.textual.ln_final
                self.visual = meru.visual
                self.dtype = torch.float32
                self.positional_embedding = meru.textual.posit_embed
                self.transformer = TransformerWrapper(meru.textual.resblocks)

            @property
            def text_projection(self):
                # Return transposed projection - always gets current device
                return self._meru.textual_proj.weight.T

        # Create wrapper for visual encoder to add input_resolution attribute
        class VisualWrapper:
            def __init__(self, visual):
                self._visual = visual
                self.input_resolution = 224  # MERU ViT-Base uses 224x224 images

            def __getattr__(self, name):
                return getattr(self._visual, name)

        pseudo_clip = PseudoCLIPModel(meru_model)
        pseudo_clip.visual = VisualWrapper(pseudo_clip.visual)

        # Create config for PromptLearner
        cfg = SimpleNamespace(
            TRAINER=SimpleNamespace(
                COOP=SimpleNamespace(
                    N_CTX=n_ctx,
                    CSC=csc,
                    CTX_INIT=ctx_init,
                    CLASS_TOKEN_POSITION=class_token_position,
                )
            ),
            INPUT=SimpleNamespace(SIZE=[224, 224]),
        )

        # Freeze MERU encoders BEFORE creating CoOp components
        # This ensures only prompt tokens and hyperbolic params remain trainable
        for param in meru_model.visual.parameters():
            param.requires_grad = False
        for param in meru_model.visual_proj.parameters():
            param.requires_grad = False
        for param in meru_model.textual.parameters():
            param.requires_grad = False
        for param in meru_model.textual_proj.parameters():
            param.requires_grad = False

        # Initialize CoOp components
        self.prompt_learner = PromptLearner(cfg, emotion_names, pseudo_clip)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(pseudo_clip)

        # Store MERU model and components
        self.meru = meru_model
        self.entail_weight = entail_weight

        # logit_scale is inherited from CLIPBaseline but is never used in
        # MERUCoOpEmotion.forward (we use Lorentzian distance, not cosine sim).
        # It is NOT in the optimizer, so optimizer.zero_grad() won't zero it and
        # scaler.unscale_() won't unscale it. Freeze it to uphold the invariant:
        # every non-optimizer param must have requires_grad=False.
        self.meru.logit_scale.requires_grad = False

        # Keep MERU's hyperbolic parameters trainable
        # (curv, visual_alpha, textual_alpha are already Parameters in meru_model)

        self.dtype = torch.float32

    def encode_image(self, images: torch.Tensor, project_to_hyperbolic: bool = True):
        """Encode images, optionally projecting to hyperbolic space."""
        # Get Euclidean features
        image_feats = self.meru.visual(images.type(self.dtype))
        image_feats = self.meru.visual_proj(image_feats)

        if project_to_hyperbolic:
            # Scale and project to hyperboloid
            image_feats = image_feats * self.meru.visual_alpha.exp()
            with torch.autocast(images.device.type, dtype=torch.float32):
                image_feats = L.exp_map0(image_feats, self.meru.curv.exp())

        return image_feats

    def encode_text(self, prompts, tokenized_prompts, project_to_hyperbolic: bool = True):
        """Encode text prompts, optionally projecting to hyperbolic space."""
        # Get Euclidean features
        text_feats = self.text_encoder(prompts, tokenized_prompts)

        if project_to_hyperbolic:
            # Scale and project to hyperboloid
            text_feats = text_feats * self.meru.textual_alpha.exp()
            with torch.autocast(prompts.device.type, dtype=torch.float32):
                text_feats = L.exp_map0(text_feats, self.meru.curv.exp())

        return text_feats

    def compute_entailment_loss(
        self, text_feats: torch.Tensor, image_feats: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entailment loss: Emotion text → Image.

        For each image, its corresponding emotion text should form an entailment
        cone that contains the image point on the hyperboloid.

        Args:
            text_feats: Emotion text features on hyperboloid (num_emotions, embed_dim)
            image_feats: Image features on hyperboloid (batch_size, embed_dim)
            labels: Emotion labels for each image (batch_size,)

        Returns:
            Entailment loss (scalar)
        """
        # Get text features corresponding to each image's emotion
        text_for_images = text_feats[labels]  # (batch_size, embed_dim)

        # Compute angle between text and image
        curv = self.meru.curv.exp()
        angle = L.oxy_angle(text_for_images, image_feats, curv)

        # Compute half-aperture of entailment cone
        aperture = L.half_aperture(text_for_images, curv)

        # Loss: penalize images outside the entailment cone
        # If angle < aperture: inside cone (good), loss = 0
        # If angle > aperture: outside cone (bad), loss = angle - aperture
        entailment_loss = torch.clamp(angle - aperture, min=0).mean()

        return entailment_loss

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None):
        """
        Forward pass for emotion classification with entailment loss.

        Args:
            images: Batch of images (B, 3, 224, 224)
            labels: Batch of emotion labels (B,)

        Returns:
            Dictionary with keys:
            - logits: Classification logits (B, num_emotions)
            - loss: Total loss (if labels provided)
            - contrastive_loss: Cross-entropy classification loss
            - entailment_loss: Emotion → Image entailment loss            - metrics: Detailed metrics for tracking (norms, violations)
        """
        # Clamp hyperbolic params to valid ranges at the top of every forward pass
        # (training AND validation). Mirrors vanilla MERU (models.py:283-289).
        # Must happen here — not just after optimizer step — so that the current
        # forward uses numerically valid parameters, and validation is also covered.
        self.meru.curv.data.clamp_(**self.meru._curv_minmax)
        self.meru.visual_alpha.data.clamp_(max=0.0)
        self.meru.textual_alpha.data.clamp_(max=0.0)

        # Get image features on hyperboloid
        image_features = self.encode_image(images, project_to_hyperbolic=True)

        # Get text features from learnable prompts on hyperboloid
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts.to(prompts.device)
        text_features = self.encode_text(
            prompts, tokenized_prompts, project_to_hyperbolic=True
        )

        # Compute norms for tracking
        image_norms = torch.norm(image_features, dim=-1)  # (B,)
        text_norms = torch.norm(text_features, dim=-1)  # (num_classes,)

        # Compute similarity using Lorentzian distance, scaled by logit_scale
        # temperature — same as vanilla MERU (models.py:321-325). logit_scale is
        # frozen here but still provides the pretrained temperature calibration.
        curv = self.meru.curv.exp()
        distances = L.pairwise_dist(image_features, text_features, curv)
        logit_scale = self.meru.logit_scale.exp()
        logits = -distances * logit_scale

        # Metrics dict for detailed tracking
        metrics = {
            "image_norm_mean": image_norms.mean().item(),
            "image_norm_std": image_norms.std().item(),
            "image_norm_min": image_norms.min().item(),
            "image_norm_max": image_norms.max().item(),
            "text_norm_mean": text_norms.mean().item(),
            "text_norm_std": text_norms.std().item(),
            "text_norm_min": text_norms.min().item(),
            "text_norm_max": text_norms.max().item(),
            "distance_mean": distances.mean().item(),
            "distance_std": distances.std().item(),
        }

        # Compute predictions once here to avoid recomputation in training script
        preds = logits.argmax(dim=-1)

        output = {"logits": logits, "preds": preds, "metrics": metrics}

        if labels is not None:
            # Contrastive loss
            contrastive_loss = F.cross_entropy(logits, labels)

            batch_size = labels.shape[0]

            # --- Entailment violation analysis ---
            # For correct-label pairs: check if image is inside the entailment cone
            text_for_images = text_features[labels]  # (B, embed_dim)
            angle = L.oxy_angle(text_for_images, image_features, curv)
            aperture = L.half_aperture(text_for_images, curv)
            # Violation: image is outside the cone (angle > aperture)
            entailment_violations = (angle > aperture).float()
            num_entailment_violations = entailment_violations.sum().item()

            # --- Contrastive/distance violation analysis ---
            # Check if the predicted class (by distance) is wrong
            contrastive_correct = (preds == labels).float()
            num_contrastive_violations = (1.0 - contrastive_correct).sum().item()

            # --- Analyze misclassifications: are they due to entailment or distance? ---
            # For misclassified samples:
            #   - "entailment_caused": the correct class had entailment violation
            #   - "distance_caused": the correct class was NOT closest by distance
            misclassified_mask = (preds != labels)
            num_misclassified = misclassified_mask.sum().item()

            if num_misclassified > 0:
                # Among misclassified: did the correct class have entailment violation?
                entail_viol_for_misclassified = entailment_violations[misclassified_mask]
                num_entail_caused = entail_viol_for_misclassified.sum().item()

                # Among misclassified: check if predicted class is closer than correct
                # (always true by definition, but we can see margin)
                correct_distances = distances[torch.arange(batch_size, device=distances.device), labels]
                pred_distances = distances[torch.arange(batch_size, device=distances.device), preds]
                distance_margin = (correct_distances - pred_distances)[misclassified_mask]
                avg_distance_margin = distance_margin.mean().item() if num_misclassified > 0 else 0.0

                # Check entailment for predicted (wrong) class
                text_for_preds = text_features[preds]
                angle_pred = L.oxy_angle(text_for_preds, image_features, curv)
                aperture_pred = L.half_aperture(text_for_preds, curv)
                pred_inside_cone = (angle_pred <= aperture_pred).float()
                # Misclassified AND predicted class is inside cone (entailment says pred is valid)
                pred_entail_valid_for_misclassified = pred_inside_cone[misclassified_mask]
                num_pred_entail_valid = pred_entail_valid_for_misclassified.sum().item()
            else:
                num_entail_caused = 0.0
                avg_distance_margin = 0.0
                num_pred_entail_valid = 0.0

            # Entailment loss: Emotion → Image
            entailment_loss = self.compute_entailment_loss(
                text_features, image_features, labels
            )

            # Total loss
            total_loss = contrastive_loss + self.entail_weight * entailment_loss

            # Update metrics with violation info
            metrics.update({
                "entailment_violations": num_entailment_violations,
                "entailment_violation_rate": num_entailment_violations / batch_size,
                "contrastive_violations": num_contrastive_violations,
                "contrastive_violation_rate": num_contrastive_violations / batch_size,
                "num_misclassified": num_misclassified,
                "misclassified_with_entail_violation": num_entail_caused,
                "misclassified_pred_entail_valid": num_pred_entail_valid,
                "misclassified_avg_distance_margin": avg_distance_margin,
                "angle_mean": angle.mean().item(),
                "angle_std": angle.std().item(),
                "aperture_mean": aperture.mean().item(),
                "aperture_std": aperture.std().item(),
            })

            output.update(
                {
                    "loss": total_loss,
                    "contrastive_loss": contrastive_loss,
                    "entailment_loss": entailment_loss,
                }
            )

        return output
