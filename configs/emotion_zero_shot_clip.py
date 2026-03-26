# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Zero-shot emotion classification on Emoset using CLIPBaseline or MERU.
No training required - uses hand-crafted emotion prompts.
"""

from meru.config import LazyCall as L
from meru.evaluation.classification import ZeroShotClassificationEvaluator


# Zero-shot evaluator with emotion-specific prompts
evaluator = L(ZeroShotClassificationEvaluator)(
    datasets_and_prompts={
        "emoset": [
            "a photo of {}",
            "an image showing {}",
            "a picture depicting {}",
            "this evokes {}",
            "this makes me feel {}",
            "an image that conveys {}",
            "a photo expressing {}",
        ],
    },
    data_dir="datasets/emoset",  # Path to emoset data
    image_size=224,
    num_workers=4,
)

# Note: To run evaluation, load a pretrained model checkpoint and pass it to evaluator
# Example usage:
#   from meru.models import CLIPBaseline, MERU
#   model = CLIPBaseline(...)  # or MERU(...)
#   model.load_state_dict(torch.load("checkpoint.pth"))
#   results = evaluator(model)
