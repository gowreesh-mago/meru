"""
EmotionCLIP-V2 + prompt tuning for emotion classification on EmoSet.

Loads the pretrained EmotionCLIP-V2 checkpoint and fine-tunes only the
20-token prompt_block, keeping all other weights frozen.

Usage:
    python scripts/train_emotion.py \
        --config configs/emotion_emotionclipv2.py \
        --output-dir output/emotionclipv2_prompt
"""

from torch.optim import AdamW

from meru.config import LazyCall as L
from meru.emotion.emotionclipv2_model import EmotionCLIPV2Emotion
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

model = L(EmotionCLIPV2Emotion)(
    emotion_names=EMOTION_NAMES,
    clip_model_name="ViT-B/32",
    ctx_template="this picture conveys a sense of",
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
        total_steps=0,   # computed at runtime: num_epochs * steps_per_epoch
        warmup_steps=0,  # computed at runtime: total_steps // 10
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
