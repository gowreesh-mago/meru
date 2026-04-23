# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
OpenAI CLIP + CoOp for emotion classification on EmoSet.

Loads OpenAI's pretrained CLIP ViT-B/32 (via ``clip.load``) and learns only
the CoOp soft-prompt ``ctx`` tokens. The backbone (visual + text transformer
+ logit_scale) is fully frozen.
"""

from torch.optim import AdamW

from meru.config import LazyCall as L
from meru.emotion.emotion_coop_models import CLIPCoOpOpenAI
from meru.optim import LinearWarmupCosineDecayLR

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

model = L(CLIPCoOpOpenAI)(
    emotion_names=EMOTION_NAMES,
    clip_model_name="ViT-B/32",
    n_ctx=16,
    ctx_init="this picture conveys a sense of",
    class_token_position="end",
    # Unified context — CSC overfits on 8-class EmoSet (see CoOp paper).
    csc=False,
)

dataset = dict(
    name="emoset",
    data_root="datasets/emoset",
    num_emotion_classes=8,
    batch_size=32,
)

optim = dict(
    optimizer=L(AdamW)(
        lr=0.002,
        betas=(0.9, 0.98),
        weight_decay=5e-4,
    ),
    lr_scheduler=L(LinearWarmupCosineDecayLR)(
        # total_steps / warmup_steps computed at runtime by train_emotion.py
        total_steps=0,
        warmup_steps=0,
    ),
)

train = dict(
    num_epochs=50,
    batch_size=32,
    num_workers=4,
    amp=True,
    gradient_clip_max_norm=1.0,
    seed=0,
    checkpoint_period=5,
    eval_period=1,
    early_stopping_patience=10,
    pretrained_checkpoint="",
)
