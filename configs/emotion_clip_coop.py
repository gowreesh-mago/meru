# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
CLIP + CoOp for emotion classification on Emoset.
Learns emotion-specific context prompts while freezing CLIP encoders.
"""

import torch
from torch.optim import SGD

from meru.config import LazyCall as L
from meru.emotion.emotion_coop_models import CLIPCoOpEmotion
from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder
from meru.models import CLIPBaseline

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

# Model: CLIPBaseline + Learnable emotion prompts
clip_base_model = L(CLIPBaseline)(
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
)

model = L(CLIPCoOpEmotion)(
    clip_model="${..clip_base_model}",
    emotion_names=EMOTION_NAMES,
    n_ctx=16,  # Number of learnable context tokens
    ctx_init="",  # Empty = random initialization
    class_token_position="end",
)

# Dataset: Emoset (registered via meru.emotion.dataset_integration)
dataset = dict(
    name="emoset",
    data_root="datasets/emoset",
    num_emotion_classes=8,
    batch_size=32,
)

# Optimizer: SGD for prompt learning (following CoOp paper)
optim = dict(
    optimizer=L(SGD)(
        lr=0.002,
        momentum=0.9,
        weight_decay=5e-4,
    ),
    lr_scheduler=L(torch.optim.lr_scheduler.CosineAnnealingLR)(
        T_max="${...train.num_epochs}",
        eta_min=1e-6,
    ),
)

# Training parameters
train = dict(
    num_epochs=50,
    batch_size=32,
    num_workers=4,
    amp=True,  # Automatic mixed precision
    seed=0,
    checkpoint_period=5,  # Save checkpoint every 5 epochs
    eval_period=1,  # Evaluate every epoch
    early_stopping_patience=10,
    pretrained_checkpoint="",  # Path to pretrained CLIPBaseline checkpoint
)
