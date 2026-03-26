# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Integration of Emoset dataset with MERU's evaluation framework.
Registers Emoset in DatasetCatalog and CLASS_NAMES.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add emoset to path
emoset_path = Path(__file__).parent.parent.parent / "emoset"
if str(emoset_path) not in sys.path:
    sys.path.insert(0, str(emoset_path))

from Emoset import EmoSet  # noqa: E402

from meru.evaluation.catalog import DatasetCatalog  # noqa: E402
from meru.evaluation.class_names import CLASS_NAMES  # noqa: E402
from meru.emotion.emotion_classes import EMOTION_CLASS_NAMES, NUM_EMOTION_CLASSES  # noqa: E402


def register_emoset():
    """
    Register Emoset dataset with MERU's DatasetCatalog.

    This function adds Emoset to:
    - DatasetCatalog.CONSTRUCTORS: Dataset constructor
    - DatasetCatalog.SPLITS: Train/val/test split names
    - DatasetCatalog.NUM_CLASSES: Number of emotion classes (8)
    - CLASS_NAMES: Emotion class names
    """
    if "emoset" not in DatasetCatalog.CONSTRUCTORS:
        # Register constructor - EmoSet expects data_root, num_emotion_classes, phase
        # We'll create a wrapper to match DatasetCatalog's interface (root, split, transform)
        def emoset_constructor(root, split, transform):
            # Map split names: train -> train, val -> val, test -> test
            phase_map = {"train": "train", "val": "val", "test": "test"}
            phase = phase_map.get(split, split)

            return EmoSet(
                data_root=root,
                num_emotion_classes=NUM_EMOTION_CLASSES,
                phase=phase,
            )

        DatasetCatalog.CONSTRUCTORS["emoset"] = emoset_constructor

        # Register splits - Emoset has train, val, test
        DatasetCatalog.SPLITS["emoset"] = ["train", "val", "test"]

        # Register number of classes
        DatasetCatalog.NUM_CLASSES["emoset"] = NUM_EMOTION_CLASSES

        # Register class names
        CLASS_NAMES["emoset"] = EMOTION_CLASS_NAMES

    return EMOTION_CLASS_NAMES


# Auto-register on import
register_emoset()
