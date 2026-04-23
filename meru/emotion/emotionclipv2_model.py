"""
Thin wrapper around EmotionCLIP-V2 for the train_emotion.py pipeline.

Only the 20-token prompt_block is trained; everything else stays frozen.
The forward signature matches CLIPCoOpEmotion so train_emotion.py needs
no structural changes — only a new import and a wandb label fix.
"""

import sys
from pathlib import Path

import clip as _clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

_EMOTIONCLIP_DIR = Path(__file__).resolve().parent.parent.parent.parent / "EmotionCLIP-V2"

if str(_EMOTIONCLIP_DIR) not in sys.path:
    sys.path.insert(0, str(_EMOTIONCLIP_DIR))

from EmotionCLIP import CLIP, Config  # noqa: E402


class EmotionCLIPV2Emotion(nn.Module):
    """
    EmotionCLIP-V2 architecture initialised from standard OpenAI CLIP weights.

    The EmotionCLIP-V2 keys are a strict superset of OpenAI CLIP keys:
    the base CLIP params (visual encoder, text transformer, embeddings,
    logit_scale) are loaded 1-to-1 from clip.load(clip_model_name).
    The extra EmotionCLIP-V2 params (visual/text prefix embeddings) are
    left at their random initialisation and frozen.

    Trainable: prompt_block.prompt_embedding only (20 × 512, converted to
    float32 so GradScaler.unscale_() can handle its gradients).
    """

    def __init__(
        self,
        emotion_names: list,
        clip_model_name: str = "ViT-B/32",
        ctx_template: str = "this picture conveys a sense of",
    ):
        super().__init__()

        config = Config()
        self._model = CLIP(config)

        # Load standard OpenAI CLIP weights into the EmotionCLIP-V2 architecture.
        # strict=False: the extra prefix/prompt params stay randomly initialised.
        clip_model, _ = _clip.load(clip_model_name, device="cpu")
        missing, unexpected = self._model.load_state_dict(
            clip_model.state_dict(), strict=False
        )
        del clip_model

        # Log what was and wasn't loaded so it's easy to verify.
        if missing:
            logger.info(f"EmotionCLIPV2: params NOT in CLIP state dict (random init, will be frozen): {missing}")
        if unexpected:
            logger.warning(f"EmotionCLIPV2: unexpected keys from CLIP (ignored): {unexpected}")

        self.dtype = config.dtype
        self.n_ctx = config.prompt_num  # 20 — for wandb logging compatibility
        self.emotion_names = emotion_names

        # Freeze everything, then unfreeze only the prompt_block.
        # Convert prompt_block to float32: GradScaler.unscale_() refuses to
        # unscale FP16 gradients, so trainable parameters must be float32.
        # Prompt_block.forward casts back to input dtype via .to(dtype).
        for param in self._model.parameters():
            param.requires_grad = False
        for param in self._model.prompt_block.parameters():
            param.data = param.data.float()
            param.requires_grad = True

        # Pre-tokenise emotion text prompts.
        # context_length=57 so encode_text's prompt_block check (n == 77-20) fires.
        texts = [f"{ctx_template} {name}" for name in emotion_names]
        tokens = _clip.tokenize(texts, context_length=57)
        self.register_buffer("_text_tokens", tokens)

    @property
    def prompt_learner(self):
        """Alias so train_emotion.py's optimizer setup works unchanged."""
        return self._model.prompt_block

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        images = images.to(self.dtype)

        image_features = self._model.encode_image(images, use_emotion=True)   # [B, D]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = self._model.encode_text(self._text_tokens, use_emotion=True)  # [N, D]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute logits in float32 for numerical stability
        logit_scale = self._model.logit_scale.exp()
        logits = logit_scale * image_features.float() @ text_features.float().t()  # [B, N]

        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)

        return {"loss": loss, "logits": logits, "preds": preds}
