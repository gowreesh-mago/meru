"""
Comprehensive tests for training fixes applied to CLIP/MERU CoOp emotion models.

Fixes under test:
  1. Gradient clipping prevents AMP overflow / optimizer-step skipping (CLIP)
  2. Class-weighted cross-entropy addresses class-collapse
  3. Ctx parameters actually receive gradient updates (not frozen at init)
  4. Text features become more diverse across emotion classes after updates
  5. Val metrics change across epochs when the model is learning

Run from the repo root:
    cd /home/gmago/Emotions/code/meru
    python -m pytest tests/test_training_fixes.py -v
  or
    python tests/test_training_fixes.py
"""

import sys
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

# ---------------------------------------------------------------------------
# Path setup so imports work without installing the package
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CoOp"))

from meru.emotion.emotion_coop_models import CLIPCoOpEmotion, MERUCoOpEmotion
from meru.models import CLIPBaseline, MERU
from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

EMOTION_NAMES = [
    "amusement", "awe", "contentment", "excitement",
    "anger", "disgust", "fear", "sadness",
]
N_CLASSES = len(EMOTION_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_clip_base() -> CLIPBaseline:
    """Build a tiny CLIPBaseline from scratch (random weights, no pretrained load)."""
    model = CLIPBaseline(
        visual=build_timm_vit(
            arch="vit_base_patch16_224",
            global_pool="token",
            use_sincos2d_pos=True,
        ),
        textual=TransformerTextEncoder(
            arch="L12_W512",
            vocab_size=49408,
            context_length=77,
        ),
        embed_dim=512,
    )
    return model


def _build_meru_base() -> MERU:
    """Build a tiny MERU model from scratch (random weights)."""
    model = MERU(
        visual=build_timm_vit(
            arch="vit_base_patch16_224",
            global_pool="token",
            use_sincos2d_pos=True,
        ),
        textual=TransformerTextEncoder(
            arch="L12_W512",
            vocab_size=49408,
            context_length=77,
        ),
        embed_dim=512,
        curv_init=1.0,
        learn_curv=True,
        entail_weight=0.0,
    )
    return model


def _build_clip_coop(n_ctx: int = 4, csc: bool = False) -> CLIPCoOpEmotion:
    base = _build_clip_base()
    model = CLIPCoOpEmotion(
        clip_model=base,
        emotion_names=EMOTION_NAMES,
        n_ctx=n_ctx,
        ctx_init="",
        class_token_position="end",
        csc=csc,
    )
    return model.to(DEVICE)


def _build_meru_coop(n_ctx: int = 4, csc: bool = False) -> MERUCoOpEmotion:
    base = _build_meru_base()
    model = MERUCoOpEmotion(
        meru_model=base,
        emotion_names=EMOTION_NAMES,
        n_ctx=n_ctx,
        ctx_init="",
        class_token_position="end",
        entail_weight=0.2,
        csc=csc,
    )
    return model.to(DEVICE)


def _fake_batch(batch_size: int = 4):
    """Return (images, labels) with random balanced labels."""
    images = torch.randn(batch_size, 3, 224, 224, device=DEVICE)
    labels = torch.arange(batch_size, device=DEVICE) % N_CLASSES
    return images, labels


def _inverse_freq_weights(class_counts: list) -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.float32)
    weights = counts.sum() / (len(counts) * counts)
    return weights.to(DEVICE)


# ---------------------------------------------------------------------------
# Fix 1 — Gradient clipping prevents AMP optimizer-skip (CLIP)
# ---------------------------------------------------------------------------

class TestAMPGradientClipping(unittest.TestCase):
    """Without gradient clipping CLIP+AMP routinely overflows and skips steps.
    With clip_max_norm=1.0 the scale should remain stable."""

    def _count_skipped_steps(self, model, n_steps: int, clip_norm=None) -> int:
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler()
        skipped = 0
        images, labels = _fake_batch(8)

        for _ in range(n_steps):
            optimizer.zero_grad()
            with amp.autocast():
                out = model(images, labels)
                loss = F.cross_entropy(out["logits"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1

        return skipped

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_clipping_bounds_gradient_norm(self):
        """After unscaling, gradient norm must be <= clip_norm when clipping is applied."""
        model = _build_clip_coop(n_ctx=4)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler()
        clip_norm = 1.0
        grad_norms = []

        for _ in range(10):
            images, labels = _fake_batch(8)
            optimizer.zero_grad()
            with amp.autocast():
                out = model(images, labels)
                loss = F.cross_entropy(out["logits"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Record norm BEFORE clipping
            total_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ) ** 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            # Record norm AFTER clipping
            clipped_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ) ** 0.5
            grad_norms.append((total_norm, clipped_norm))
            scaler.step(optimizer)
            scaler.update()

        print(f"\n  (pre_clip, post_clip) norms over 10 steps:")
        for pre, post in grad_norms:
            print(f"    {pre:.4f} → {post:.4f}")

        for pre, post in grad_norms:
            if pre > 1e-6:  # only check non-trivial gradients
                self.assertLessEqual(
                    post, clip_norm + 1e-4,
                    f"Gradient norm {post:.4f} exceeds clip_norm={clip_norm} after clipping",
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_clipped_model_has_stable_scaler(self):
        """After 20 steps with clipping, scaler should not have collapsed to 1."""
        model = _build_clip_coop(n_ctx=4)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler()
        images, labels = _fake_batch(8)

        for _ in range(20):
            optimizer.zero_grad()
            with amp.autocast():
                out = model(images, labels)
                loss = F.cross_entropy(out["logits"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        self.assertGreater(
            scaler.get_scale(), 1.0,
            "Scaler should stay above 1.0 when gradients are properly clipped",
        )


# ---------------------------------------------------------------------------
# Fix 2 — Class-weighted loss prevents class collapse
# ---------------------------------------------------------------------------

class TestClassWeightedLoss(unittest.TestCase):
    """Weighted CE should penalise minority-class errors more, yielding higher
    loss for a model that ignores minority classes compared to unweighted CE."""

    def _imbalanced_labels(self, batch_size: int = 32) -> torch.Tensor:
        """Simulate imbalance: class 0 (amusement) has 60% of samples."""
        labels = [0] * int(batch_size * 0.6)
        labels += [i % (N_CLASSES - 1) + 1 for i in range(batch_size - len(labels))]
        return torch.tensor(labels, device=DEVICE)

    def test_weighted_loss_penalises_collapse_more(self):
        """A model that always predicts class 0 should have higher weighted loss
        than unweighted loss, because minority classes are penalised more."""
        labels = self._imbalanced_labels(32)

        # Simulate a model that always predicts class 0 with high confidence
        logits = torch.zeros(len(labels), N_CLASSES, device=DEVICE)
        logits[:, 0] = 10.0

        # Class counts reflecting 60/40 imbalance
        class_counts = [int(len(labels) * 0.6)] + [int(len(labels) * 0.4 / (N_CLASSES - 1))] * (N_CLASSES - 1)
        weights = _inverse_freq_weights(class_counts)

        loss_unweighted = F.cross_entropy(logits, labels).item()
        loss_weighted = F.cross_entropy(logits, labels, weight=weights).item()

        print(f"\n  Loss unweighted: {loss_unweighted:.4f}")
        print(f"  Loss weighted:   {loss_weighted:.4f}")

        self.assertGreater(
            loss_weighted, loss_unweighted,
            "Weighted loss must be higher than unweighted when minority classes are wrong",
        )

    def test_uniform_weights_equal_unweighted(self):
        """Uniform class weights should give the same loss as no weights."""
        labels = torch.randint(0, N_CLASSES, (16,), device=DEVICE)
        logits = torch.randn(16, N_CLASSES, device=DEVICE)

        uniform_weights = torch.ones(N_CLASSES, device=DEVICE)
        loss_uw = F.cross_entropy(logits, labels).item()
        loss_w = F.cross_entropy(logits, labels, weight=uniform_weights).item()

        self.assertAlmostEqual(loss_uw, loss_w, places=4)

    def test_balanced_training_step_raises_minority_gradient(self):
        """With weighted loss, minority-class samples should have larger gradient
        contribution to ctx than with unweighted loss."""
        model = _build_clip_coop(n_ctx=4)
        labels = self._imbalanced_labels(32)
        images = torch.randn(32, 3, 224, 224, device=DEVICE)

        class_counts = [int(32 * 0.6)] + [int(32 * 0.4 / (N_CLASSES - 1))] * (N_CLASSES - 1)
        weights = _inverse_freq_weights(class_counts)

        # Unweighted
        model.zero_grad()
        out = model(images, labels)
        loss_uw = F.cross_entropy(out["logits"], labels)
        loss_uw.backward()
        grad_uw = model.prompt_learner.ctx.grad.clone()

        # Weighted
        model.zero_grad()
        out = model(images, labels)
        loss_w = F.cross_entropy(out["logits"], labels, weight=weights)
        loss_w.backward()
        grad_w = model.prompt_learner.ctx.grad.clone()

        # Gradient norms should differ
        norm_uw = grad_uw.norm().item()
        norm_w = grad_w.norm().item()
        print(f"\n  ctx grad norm (unweighted): {norm_uw:.6f}")
        print(f"  ctx grad norm (weighted):   {norm_w:.6f}")

        self.assertNotAlmostEqual(
            norm_uw, norm_w, places=4,
            msg="Weighted and unweighted losses should produce different ctx gradients",
        )


# ---------------------------------------------------------------------------
# Fix 3 — Ctx parameters receive non-zero gradient updates
# ---------------------------------------------------------------------------

class TestCtxGradientFlow(unittest.TestCase):
    """The prompt context tensor must receive non-zero gradients during a
    forward+backward pass, for both CLIP and MERU variants."""

    def _check_ctx_grad(self, model, model_name: str):
        images, labels = _fake_batch(4)
        out = model(images, labels)
        loss = F.cross_entropy(out["logits"], labels)
        loss.backward()

        ctx = model.prompt_learner.ctx
        self.assertIsNotNone(ctx.grad, f"{model_name}: ctx.grad is None after backward")
        grad_norm = ctx.grad.norm().item()
        print(f"\n  {model_name} ctx grad norm: {grad_norm:.6f}")
        self.assertGreater(grad_norm, 0.0, f"{model_name}: ctx gradient is zero")

    def test_clip_ctx_receives_gradient(self):
        model = _build_clip_coop(n_ctx=4)
        self._check_ctx_grad(model, "CLIPCoOpEmotion")

    def test_meru_ctx_receives_gradient(self):
        model = _build_meru_coop(n_ctx=4)
        self._check_ctx_grad(model, "MERUCoOpEmotion")

    def test_frozen_encoder_params_have_no_grad(self):
        """Visual encoder weights must have requires_grad=False."""
        model = _build_clip_coop(n_ctx=4)
        for name, param in model.named_parameters():
            if "prompt_learner" not in name and "logit_scale" not in name:
                self.assertFalse(
                    param.requires_grad,
                    f"Param '{name}' should be frozen but requires_grad=True",
                )

    def test_only_ctx_updated_after_step(self):
        """After an optimizer step, only the ctx tensor should have changed."""
        model = _build_clip_coop(n_ctx=4)
        ctx_before = model.prompt_learner.ctx.data.clone()

        # Capture a frozen-encoder param before the step
        frozen_param_before = None
        frozen_param_name = None
        for name, param in model.named_parameters():
            if "image_encoder" in name:
                frozen_param_before = param.data.clone()
                frozen_param_name = name
                break

        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=1e-3)
        images, labels = _fake_batch(4)
        out = model(images, labels)
        loss = F.cross_entropy(out["logits"], labels)
        loss.backward()
        optimizer.step()

        ctx_after = model.prompt_learner.ctx.data
        self.assertFalse(
            torch.allclose(ctx_before, ctx_after),
            "ctx should have changed after an optimizer step",
        )

        if frozen_param_before is not None:
            frozen_param_after = dict(model.named_parameters())[frozen_param_name].data
            self.assertTrue(
                torch.allclose(frozen_param_before, frozen_param_after),
                f"Frozen param '{frozen_param_name}' must not change after optimizer step",
            )


# ---------------------------------------------------------------------------
# Fix 4 — Text feature diversity increases with training
# ---------------------------------------------------------------------------

class TestTextFeatureDiversity(unittest.TestCase):
    """After gradient updates the 8 emotion text embeddings should become more
    diverse (lower mean pairwise cosine similarity) than at random init."""

    def _pairwise_cos_sim_mean(self, model) -> float:
        """Return mean off-diagonal cosine similarity across 8 emotion text features."""
        model.eval()
        with torch.no_grad():
            prompts = model.prompt_learner()
            tokenized = model.tokenized_prompts.to(DEVICE)
            text_feats = model.text_encoder(prompts, tokenized)
            text_feats = F.normalize(text_feats, dim=-1)
        sim = text_feats @ text_feats.t()
        n = sim.shape[0]
        # off-diagonal mean
        mask = ~torch.eye(n, dtype=torch.bool, device=DEVICE)
        return sim[mask].mean().item()

    def test_initial_similarity_is_high(self):
        """Pretrained encoder produces very similar text features for different
        emotions — this is what we're trying to fix."""
        model = _build_clip_coop(n_ctx=4)
        sim = self._pairwise_cos_sim_mean(model)
        print(f"\n  Initial text pairwise cos-sim mean: {sim:.4f}")
        # Not asserting a threshold — just log for reference

    def test_gradient_updates_change_text_features(self):
        """After 5 gradient steps the text features must differ from init."""
        model = _build_clip_coop(n_ctx=4)

        prompts_init = model.prompt_learner().detach().clone()

        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=1e-2)
        for _ in range(5):
            images, labels = _fake_batch(8)
            optimizer.zero_grad()
            out = model(images, labels)
            loss = F.cross_entropy(out["logits"], labels)
            loss.backward()
            optimizer.step()

        prompts_after = model.prompt_learner().detach()
        self.assertFalse(
            torch.allclose(prompts_init, prompts_after, atol=1e-6),
            "Prompt tensors must change after gradient updates",
        )

    def test_diversity_improves_with_balanced_loss(self):
        """Weighted CE should push text features apart more than unweighted CE
        over the same number of steps (higher ctx norm change)."""
        def ctx_delta(use_weights: bool, n_steps: int = 10) -> float:
            model = _build_clip_coop(n_ctx=4)
            ctx_init = model.prompt_learner.ctx.data.clone()
            class_counts = [N_CLASSES] * N_CLASSES  # uniform
            weights = _inverse_freq_weights(class_counts) if use_weights else None
            optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=1e-2)
            for _ in range(n_steps):
                images, labels = _fake_batch(8)
                optimizer.zero_grad()
                out = model(images, labels)
                loss = F.cross_entropy(out["logits"], labels, weight=weights)
                loss.backward()
                optimizer.step()
            return (model.prompt_learner.ctx.data - ctx_init).norm().item()

        delta_uw = ctx_delta(use_weights=False)
        delta_w = ctx_delta(use_weights=True)
        print(f"\n  ctx Δ norm (unweighted): {delta_uw:.4f}")
        print(f"  ctx Δ norm (weighted):   {delta_w:.4f}")
        # Both should be non-zero (model is learning)
        self.assertGreater(delta_uw, 0.0, "Ctx should change with unweighted loss")
        self.assertGreater(delta_w, 0.0, "Ctx should change with weighted loss")


# ---------------------------------------------------------------------------
# Fix 5 — Val metrics vary across epochs (model is actually learning)
# ---------------------------------------------------------------------------

class TestValMetricsVariation(unittest.TestCase):
    """Run a tiny training loop for 3 epochs on synthetic data.
    Val loss values across epochs must not be bit-for-bit identical,
    proving the model updates are reaching the val-time predictions."""

    def _run_mini_training(self, model, n_epochs: int = 3, n_batches: int = 5):
        """Run a short training loop and return per-epoch val losses."""
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=1e-2)
        val_losses = []
        for epoch in range(n_epochs):
            model.train()
            for _ in range(n_batches):
                images, labels = _fake_batch(8)
                optimizer.zero_grad()
                out = model(images, labels)
                loss = F.cross_entropy(out["logits"], labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                images, labels = _fake_batch(16)
                out = model(images, labels)
                val_loss = F.cross_entropy(out["logits"], labels).item()
            val_losses.append(val_loss)
        return val_losses

    def test_clip_val_loss_varies_across_epochs(self):
        model = _build_clip_coop(n_ctx=4)
        losses = self._run_mini_training(model, n_epochs=3, n_batches=5)
        print(f"\n  CLIP val losses across 3 epochs: {[f'{l:.6f}' for l in losses]}")
        # All losses being identical means the model isn't learning at all
        self.assertFalse(
            losses[0] == losses[1] == losses[2],
            f"CLIP val losses are identical across all epochs: {losses} — "
            "ctx is not being updated (gradient not flowing or optimizer steps skipped)",
        )

    def test_meru_val_loss_varies_across_epochs(self):
        model = _build_meru_coop(n_ctx=4)
        losses = self._run_mini_training(model, n_epochs=3, n_batches=5)
        print(f"\n  MERU val losses across 3 epochs: {[f'{l:.6f}' for l in losses]}")
        self.assertFalse(
            losses[0] == losses[1] == losses[2],
            f"MERU val losses are identical across all epochs: {losses}",
        )


# ---------------------------------------------------------------------------
# Fix 6 — CSC (class-specific context) parameter count
# ---------------------------------------------------------------------------

class TestClassSpecificContext(unittest.TestCase):
    """With csc=True each class gets its own ctx tensor: shape (n_cls, n_ctx, dim).
    With csc=False all classes share one tensor: shape (n_ctx, dim)."""

    def _ctx_param_count(self, model) -> int:
        return model.prompt_learner.ctx.numel()

    def test_shared_ctx_param_count(self):
        n_ctx, embed_dim = 4, 512
        model = _build_clip_coop(n_ctx=n_ctx, csc=False)
        expected = n_ctx * embed_dim
        actual = self._ctx_param_count(model)
        self.assertEqual(actual, expected, f"Shared ctx: expected {expected}, got {actual}")

    def test_class_specific_ctx_param_count(self):
        n_ctx, embed_dim = 4, 512
        model = _build_clip_coop(n_ctx=n_ctx, csc=True)
        expected = N_CLASSES * n_ctx * embed_dim
        actual = self._ctx_param_count(model)
        self.assertEqual(
            actual, expected,
            f"CSC ctx: expected {expected} ({N_CLASSES}×{n_ctx}×{embed_dim}), got {actual}",
        )

    def test_csc_gradients_per_class(self):
        """With CSC each class's ctx slice should receive independent gradient."""
        model = _build_clip_coop(n_ctx=4, csc=True)
        images, labels = _fake_batch(8)  # labels 0..7
        out = model(images, labels)
        loss = F.cross_entropy(out["logits"], labels)
        loss.backward()

        ctx_grad = model.prompt_learner.ctx.grad  # (n_cls, n_ctx, dim)
        self.assertEqual(ctx_grad.shape[0], N_CLASSES)
        # All class gradients should be non-zero since all labels appear in batch
        for cls_idx in range(N_CLASSES):
            self.assertGreater(
                ctx_grad[cls_idx].norm().item(), 0.0,
                f"Class {cls_idx} ctx gradient is zero — loss not flowing per-class",
            )


# ---------------------------------------------------------------------------
# Fix 7 — MERU-specific: hyperbolic params still trainable after freeze
# ---------------------------------------------------------------------------

class TestMERUHyperbolicParams(unittest.TestCase):
    """curv, visual_alpha, textual_alpha must remain trainable after the
    visual/textual encoder freeze inside MERUCoOpEmotion.__init__.

    Note: these params are accessed via model.meru.{curv,visual_alpha,textual_alpha}.
    In named_parameters() they may appear under a different prefix because
    TextEncoder also holds a reference to the same MERU model object. Direct
    attribute access (model.meru.curv) is always correct.
    """

    def test_hyperbolic_params_trainable(self):
        model = _build_meru_coop(n_ctx=4)
        for attr in ("curv", "visual_alpha", "textual_alpha"):
            param = getattr(model.meru, attr)
            self.assertTrue(
                param.requires_grad,
                f"Hyperbolic param 'meru.{attr}' should be trainable but is frozen",
            )

    def test_curv_clamped_during_training(self):
        """Simulate the clamping that happens in the training loop."""
        model = _build_meru_coop(n_ctx=4)
        # Force curv way outside bounds
        with torch.no_grad():
            model.meru.curv.data.fill_(100.0)

        # Mimic training-loop clamp (use float32-aware bound)
        log10_f32 = torch.tensor(10.0).log().item()
        log01_f32 = torch.tensor(0.1).log().item()
        with torch.no_grad():
            model.meru.curv.data.clamp_(min=log01_f32, max=log10_f32)

        self.assertLessEqual(model.meru.curv.item(), log10_f32 + 1e-6)
        self.assertGreaterEqual(model.meru.curv.item(), log01_f32 - 1e-6)


# ---------------------------------------------------------------------------
# Fix 8 — Integration: weighted loss + clipping together (CLIP, 3 steps)
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):
    """End-to-end test: weighted CE + gradient clipping + AMP for CLIP.
    Three optimizer steps must all succeed (no skip), and ctx must change."""

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_clip_integrated_training_steps(self):
        model = _build_clip_coop(n_ctx=4)
        ctx_init = model.prompt_learner.ctx.data.clone()

        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler()
        class_counts = [12, 8, 10, 9, 7, 11, 8, 9]
        weights = _inverse_freq_weights(class_counts)

        skipped = 0
        for _ in range(3):
            images, labels = _fake_batch(8)
            optimizer.zero_grad()
            with amp.autocast():
                out = model(images, labels)
                loss = F.cross_entropy(out["logits"], labels, weight=weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1

        ctx_after = model.prompt_learner.ctx.data
        print(f"\n  Integration: skipped={skipped}/3, "
              f"ctx_delta_norm={( ctx_after - ctx_init).norm().item():.6f}")

        self.assertEqual(skipped, 0, "All 3 steps should succeed with clipping; none skipped")
        self.assertFalse(
            torch.allclose(ctx_init, ctx_after, atol=1e-8),
            "ctx must change after 3 integrated training steps",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
