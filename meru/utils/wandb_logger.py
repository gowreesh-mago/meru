# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Weights & Biases logging integration.
Provides optional wandb logging that can be enabled via environment variables.
"""

import os
from typing import Any, Dict, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbLogger:
    """
    Wrapper for Weights & Biases logging.

    Usage:
        wandb_logger = WandbLogger(config=config_dict, name="my_experiment")
        wandb_logger.log({"train/loss": 0.5, "epoch": 1})
        wandb_logger.finish()

    Environment variables:
        WANDB_PROJECT: Project name (default: "emotion-experiments")
        WANDB_ENTITY: Team/username (optional)
        WANDB_NAME: Run name (optional, auto-generated if not set)
        WANDB_DIR: Directory for wandb files (default: "./wandb")
        WANDB_MODE: "online" (default), "offline", or "disabled"
        WANDB_DISABLED: Set to "true" to disable wandb completely
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
    ):
        """
        Initialize wandb logger.

        Args:
            config: Configuration dictionary to log
            name: Run name (overrides WANDB_NAME env var)
            tags: List of tags for the run
            notes: Optional notes about the run
            group: Group name for organizing runs
        """
        self.enabled = False
        self.run = None

        # Check if wandb is available and not disabled
        if not WANDB_AVAILABLE:
            print("⚠️  wandb not installed. Install with: pip install wandb")
            return

        # Check if explicitly disabled
        if os.getenv("WANDB_DISABLED", "").lower() == "true":
            print("ℹ️  wandb disabled via WANDB_DISABLED=true")
            return

        # Check wandb mode
        mode = os.getenv("WANDB_MODE", "online")
        if mode == "disabled":
            print("ℹ️  wandb disabled via WANDB_MODE=disabled")
            return

        # Get configuration from environment
        project = os.getenv("WANDB_PROJECT", "emotion-experiments")
        entity = os.getenv("WANDB_ENTITY", None)
        run_name = name or os.getenv("WANDB_NAME", None)
        wandb_dir = os.getenv("WANDB_DIR", "./wandb")

        # Create wandb directory
        os.makedirs(wandb_dir, exist_ok=True)

        try:
            # Initialize wandb
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config,
                tags=tags,
                notes=notes,
                group=group,
                dir=wandb_dir,
                mode=mode,
            )
            self.enabled = True
            print(f"✓ wandb initialized: {project}" + (f"/{entity}" if entity else ""))
            if run_name:
                print(f"  Run name: {run_name}")
            print(f"  Mode: {mode}")
            print(f"  URL: {self.run.url if hasattr(self.run, 'url') else 'N/A'}")
        except Exception as e:
            print(f"⚠️  Failed to initialize wandb: {e}")
            self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (e.g., epoch, iteration)
            commit: Whether to commit the metrics immediately
        """
        if not self.enabled or self.run is None:
            return

        try:
            if step is not None:
                wandb.log(metrics, step=step, commit=commit)
            else:
                wandb.log(metrics, commit=commit)
        except Exception as e:
            print(f"⚠️  Failed to log to wandb: {e}")

    def log_artifact(self, artifact_path: str, artifact_type: str, name: Optional[str] = None):
        """
        Log an artifact (file/directory) to wandb.

        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (e.g., "model", "dataset", "config")
            name: Optional name for the artifact
        """
        if not self.enabled or self.run is None:
            return

        try:
            artifact = wandb.Artifact(
                name=name or os.path.basename(artifact_path),
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)
        except Exception as e:
            print(f"⚠️  Failed to log artifact to wandb: {e}")

    def watch(self, model, log: str = "all", log_freq: int = 1000):
        """
        Watch a model's gradients and parameters.

        Args:
            model: PyTorch model to watch
            log: What to log - "gradients", "parameters", "all", or None
            log_freq: Logging frequency in steps
        """
        if not self.enabled or self.run is None:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            print(f"⚠️  Failed to watch model with wandb: {e}")

    def finish(self):
        """Finish the wandb run."""
        if not self.enabled or self.run is None:
            return

        try:
            wandb.finish()
            print("✓ wandb run finished")
        except Exception as e:
            print(f"⚠️  Failed to finish wandb run: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
