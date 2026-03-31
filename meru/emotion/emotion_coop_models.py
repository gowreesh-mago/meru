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

from trainers.coop import PromptLearner, TextEncoder  # noqa: E402

from meru import lorentz as L  # noqa: E402


class TransformerWrapper(nn.Module):
    """Wrapper to make ModuleList of resblocks callable as a single transformer."""

    def __init__(self, resblocks):
        super().__init__()
        self.resblocks = resblocks

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
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
                    CSC=False,  # Not class-specific context
                    CTX_INIT=ctx_init,
                    CLASS_TOKEN_POSITION=class_token_position,
                )
            ),
            INPUT=SimpleNamespace(SIZE=[224, 224]),  # Image size
        )

        # Initialize CoOp components
        self.prompt_learner = PromptLearner(cfg, emotion_names, pseudo_clip)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(pseudo_clip)

        # Store CLIP components
        self.image_encoder = clip_model.visual
        self.visual_proj = clip_model.visual_proj
        self.logit_scale = clip_model.logit_scale

        # Freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.visual_proj.parameters():
            param.requires_grad = False

        # Only prompt_learner parameters are trainable
        self.dtype = torch.float32

    def to(self, *args, **kwargs):
        """Override to() to also move TextEncoder's text_projection tensor."""
        # Move the model
        self = super().to(*args, **kwargs)

        # TextEncoder's text_projection is a plain tensor, not a parameter/buffer
        # So we need to move it manually
        if hasattr(self.text_encoder, 'text_projection') and isinstance(self.text_encoder.text_projection, torch.Tensor):
            self.text_encoder.text_projection = self.text_encoder.text_projection.to(*args, **kwargs)

        return self

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

        # Compute similarity logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        output = {"logits": logits}

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
                    CSC=False,
                    CTX_INIT=ctx_init,
                    CLASS_TOKEN_POSITION=class_token_position,
                )
            ),
            INPUT=SimpleNamespace(SIZE=[224, 224]),
        )

        # Initialize CoOp components
        self.prompt_learner = PromptLearner(cfg, emotion_names, pseudo_clip)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(pseudo_clip)

        # Store MERU model and components
        self.meru = meru_model
        self.entail_weight = entail_weight

        # Freeze encoders
        for param in self.meru.visual.parameters():
            param.requires_grad = False
        for param in self.meru.textual.parameters():
            param.requires_grad = False

        # Keep MERU's hyperbolic parameters trainable
        # (curv, visual_alpha, textual_alpha are already Parameters in meru_model)

        self.dtype = torch.float32

    def to(self, *args, **kwargs):
        """Override to() to also move TextEncoder's text_projection tensor."""
        # Move the model
        self = super().to(*args, **kwargs)

        # TextEncoder's text_projection is a plain tensor, not a parameter/buffer
        # So we need to move it manually
        if hasattr(self.text_encoder, 'text_projection') and isinstance(self.text_encoder.text_projection, torch.Tensor):
            self.text_encoder.text_projection = self.text_encoder.text_projection.to(*args, **kwargs)

        return self

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
            - entailment_loss: Emotion → Image entailment loss
        """
        # Get image features on hyperboloid
        image_features = self.encode_image(images, project_to_hyperbolic=True)

        # Get text features from learnable prompts on hyperboloid
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts.to(prompts.device)
        text_features = self.encode_text(
            prompts, tokenized_prompts, project_to_hyperbolic=True
        )

        # Compute similarity using Lorentzian distance
        curv = self.meru.curv.exp()
        distances = L.pairwise_dist(image_features, text_features, curv)
        logits = -distances  # Negative distance (closer = higher logit)

        output = {"logits": logits}

        if labels is not None:
            # Contrastive loss
            contrastive_loss = F.cross_entropy(logits, labels)

            # Entailment loss: Emotion → Image
            entailment_loss = self.compute_entailment_loss(
                text_features, image_features, labels
            )

            # Total loss
            total_loss = contrastive_loss + self.entail_weight * entailment_loss

            output.update(
                {
                    "loss": total_loss,
                    "contrastive_loss": contrastive_loss,
                    "entailment_loss": entailment_loss,
                }
            )

        return output
