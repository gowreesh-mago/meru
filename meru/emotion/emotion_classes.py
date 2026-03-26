# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Emotion class definitions for emotion classification on Emoset dataset.
"""

from __future__ import annotations

# Emotion class names for 8-class classification
EMOTION_CLASS_NAMES = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
]

# Number of emotion classes
NUM_EMOTION_CLASSES = 8

# Emotion class descriptions (for prompt templates)
EMOTION_DESCRIPTIONS = {
    "amusement": "a feeling of finding something funny or entertaining",
    "awe": "a feeling of wonder and amazement",
    "contentment": "a feeling of peaceful happiness and satisfaction",
    "excitement": "a feeling of enthusiasm and eagerness",
    "anger": "a strong feeling of displeasure or hostility",
    "disgust": "a feeling of strong disapproval or revulsion",
    "fear": "a feeling of being afraid or worried",
    "sadness": "a feeling of sorrow or unhappiness",
}


def get_emotion_class_names():
    """Get the list of emotion class names."""
    return EMOTION_CLASS_NAMES


def get_num_emotion_classes():
    """Get the number of emotion classes."""
    return NUM_EMOTION_CLASSES


def get_emotion_descriptions():
    """Get dictionary of emotion descriptions."""
    return EMOTION_DESCRIPTIONS
