# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
MERU + CoOp for emotion classification on Emoset.
Learns emotion-specific context prompts in hyperbolic space with entailment loss.
"""

from torch.optim import AdamW

from meru.config import LazyCall as L
from meru.emotion.emotion_coop_models import MERUCoOpEmotion
from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder
from meru.models import MERU
from meru.optim import LinearWarmupCosineDecayLR

# Emotion class names
EMOTION_NAMES = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
]

# Model: MERU + Learnable emotion prompts + Entailment loss
meru_base_model = L(MERU)(
    visual=L(build_timm_vit)(
        arch="vit_base_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512",
        vocab_size=49408,
        context_length=77,
    ),
    embed_dim=512,
    curv_init=1.0,  # Initial negative curvature
    learn_curv=True,  # Learn curvature parameter
    entail_weight=0.0,  # Not used (we apply entailment in MERUCoOpEmotion)
)

model = L(MERUCoOpEmotion)(
    meru_model="${..meru_base_model}",
    emotion_names=EMOTION_NAMES,
    n_ctx=16,  # Number of learnable context tokens
    ctx_init="this picture conveys a sense of",  # Initialize with meaningful text (better than random)
    class_token_position="end",
    entail_weight=0.2,  # Weight for Emotion → Image entailment loss
    csc=True,  # Class-specific context - each emotion gets its own learnable context
)

# Dataset: Emoset
dataset = dict(
    name="emoset",
    data_root="datasets/emoset",
    num_emotion_classes=8,
    batch_size=32,
)

# Optimizer: AdamW with different LRs for prompt vs hyperbolic params
optim = dict(
    optimizer=L(AdamW)(
        # Separate param groups defined in training script:
        # - prompt_learner.parameters(): lr=2e-3, weight_decay=5e-4
        # - [curv, visual_alpha, textual_alpha]: lr=5e-4, weight_decay=0.0
        lr=2e-3,  # Default LR for prompts
        betas=(0.9, 0.98),
        weight_decay=5e-4,
    ),
    lr_scheduler=L(LinearWarmupCosineDecayLR)(
        # Note: total_steps and warmup_steps will be computed by training script
        # as: total_steps = num_epochs * steps_per_epoch
        #     warmup_steps = total_steps // 10  (10% warmup)
        total_steps=0,  # Placeholder, computed at runtime
        warmup_steps=0,  # Placeholder, computed at runtime
    ),
)

# Training parameters
train = dict(
    num_epochs=100,  # More epochs for hyperbolic training
    batch_size=32,
    num_workers=4,
    amp=True,  # Automatic mixed precision (critical for stability)
    gradient_clip_max_norm=1.0,  # Gradient clipping for stability
    seed=0,
    checkpoint_period=5,
    eval_period=1,
    early_stopping_patience=15,
    pretrained_checkpoint="",  # Path to pretrained MERU checkpoint
    steps_per_epoch=0,  # Will be computed from dataset size
)
