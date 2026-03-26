# Emotion classification models and utilities for Emoset dataset

from meru.emotion.dataset_integration import register_emoset
from meru.emotion.emotion_coop_models import CLIPCoOpEmotion, MERUCoOpEmotion
from meru.emotion.emotion_classes import (
    EMOTION_CLASS_NAMES,
    NUM_EMOTION_CLASSES,
    EMOTION_DESCRIPTIONS,
    get_emotion_class_names,
    get_num_emotion_classes,
    get_emotion_descriptions,
)

__all__ = [
    "register_emoset",
    "CLIPCoOpEmotion",
    "MERUCoOpEmotion",
    "EMOTION_CLASS_NAMES",
    "NUM_EMOTION_CLASSES",
    "EMOTION_DESCRIPTIONS",
    "get_emotion_class_names",
    "get_num_emotion_classes",
    "get_emotion_descriptions",
]
