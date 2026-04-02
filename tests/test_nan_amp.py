"""
Targeted tests to diagnose and pin the NaN / AMP-overflow issue seen in
sanity_20260401_163431/clip_coop.

Observed failure signature
--------------------------
- Epoch 1 Batch 0: loss = 2.53  (valid forward pass)
- Epoch 1 Batch 10: loss = nan  (text_sim_mean = nan  → F.normalize(text_features) = nan)
- AMP scaler scale collapses to ~0 in one epoch (65536 / 2^31 ≈ 3e-5)
- Skipped steps: 31/32

Two bugs under investigation
-----------------------------
Bug A — logit_scale pollutes clip_grad_norm_:
  CLIPCoOpEmotion stores self.logit_scale = clip_model.logit_scale, which has
  requires_grad=True but is NOT in the optimizer.  scaler.unscale_(optimizer)
  only unscales prompt_learner params.  clip_grad_norm_(model.parameters())
  then mixes the *still-scaled* logit_scale.grad (≈ 65536×true_grad) with the
  already-unscaled ctx.grad.  Result: total_norm ≈ 65536 → clip_coeff ≈ 0 →
  ctx.grad is multiplied by ≈ 0, giving a near-zero update. Additionally,
  optimizer.zero_grad() only zeros prompt_learner.parameters() so
  logit_scale.grad accumulates NaN across subsequent batches.

Bug B — fp16 overflow / underflow in the text encoder:
  Under AMP autocast the text transformer runs in fp16.  After the near-zero
  (but non-zero) ctx update from Bug A's one successful step, the AdamW
  denominator is dominated by eps (v ≈ 0) so the effective per-element step
  can be lr * m_hat / eps, which can be unexpectedly large in the wrong
  direction.  The resulting ctx values, combined with the fp16 text encoder,
  can produce inf attention logits → softmax NaN.  amp=False (fp32 throughout)
  eliminates the overflow and keeps text features finite.

Run from repo root:
    cd /home/gmago/Emotions/code/meru
    python -m pytest tests/test_nan_amp.py -v -s
"""

import sys
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.cuda import amp

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CoOp"))

from meru.emotion.emotion_coop_models import CLIPCoOpEmotion, MERUCoOpEmotion
from meru.models import CLIPBaseline, MERU
from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder

EMOTION_NAMES = [
    "amusement", "awe", "contentment", "excitement",
    "anger", "disgust", "fear", "sadness",
]
N_CLASSES = len(EMOTION_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_meru_base() -> MERU:
    return MERU(
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


def _build_meru_coop(n_ctx: int = 4, csc: bool = False) -> MERUCoOpEmotion:
    base = _build_meru_base()
    return MERUCoOpEmotion(
        meru_model=base,
        emotion_names=EMOTION_NAMES,
        n_ctx=n_ctx,
        ctx_init="",
        class_token_position="end",
        entail_weight=0.2,
        csc=csc,
    ).to(DEVICE)


def _text_features_finite_meru(model: MERUCoOpEmotion) -> bool:
    """Return True if MERU Euclidean and hyperbolic text features are all finite."""
    model.eval()
    with torch.no_grad():
        prompts = model.prompt_learner()
        tokenized = model.tokenized_prompts.to(DEVICE)
        text_feats_euc = model.text_encoder(prompts, tokenized)
        text_feats_hyp = model.encode_text(prompts, tokenized, project_to_hyperbolic=True)
    return (
        torch.isfinite(text_feats_euc).all().item()
        and torch.isfinite(text_feats_hyp).all().item()
    )


def _build_clip_base() -> CLIPBaseline:
    return CLIPBaseline(
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


def _build_clip_coop(n_ctx: int = 4, ctx_init: str = "", csc: bool = False,
                     freeze_logit_scale: bool = False) -> CLIPCoOpEmotion:
    base = _build_clip_base()
    model = CLIPCoOpEmotion(
        clip_model=base,
        emotion_names=EMOTION_NAMES,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        class_token_position="end",
        csc=csc,
    )
    if freeze_logit_scale:
        model.logit_scale.requires_grad = False
    return model.to(DEVICE)


def _text_features_are_finite(model: CLIPCoOpEmotion) -> bool:
    """Return True if text features and their normalised form are all finite."""
    model.eval()
    with torch.no_grad():
        prompts = model.prompt_learner()
        tokenized = model.tokenized_prompts.to(DEVICE)
        text_feats = model.text_encoder(prompts, tokenized)
        normed = F.normalize(text_feats, dim=-1)
    return torch.isfinite(text_feats).all().item() and torch.isfinite(normed).all().item()


def _run_amp_training_step(model, optimizer, scaler, images, labels,
                            clip_params=None):
    """
    Single AMP training step exactly as in train_emotion.py.
    clip_params: iterable of parameters for clip_grad_norm_. Defaults to
                 model.parameters() (the buggy behaviour). Pass
                 model.prompt_learner.parameters() to test the fix.
    Returns (scale_decreased, logit_scale_grad_norm).
    """
    if clip_params is None:
        clip_params = model.parameters()  # Bug: includes unscaled logit_scale.grad

    optimizer.zero_grad()
    with amp.autocast():
        out = model(images, labels)
        loss = F.cross_entropy(out["logits"], labels)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(list(clip_params), 1.0)

    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale_decreased = scaler.get_scale() < scale_before

    ls_grad_norm = (
        model.logit_scale.grad.norm().item()
        if model.logit_scale.grad is not None else 0.0
    )
    return scale_decreased, ls_grad_norm


# ---------------------------------------------------------------------------
# Bug A: logit_scale.grad is unfrozen and excluded from optimizer.zero_grad()
# ---------------------------------------------------------------------------

class TestLogitScaleGradFix(unittest.TestCase):
    """
    Regression tests for the logit_scale gradient pollution fix.

    Root cause (now fixed in emotion_coop_models.py):
      CLIPCoOpEmotion stored self.logit_scale = clip_model.logit_scale with
      requires_grad=True but NOT added to the optimizer.  This caused:
        1. scaler.unscale_(optimizer) did not unscale logit_scale.grad, leaving
           it at 65536×true_grad when clip_grad_norm_(model.parameters()) ran.
        2. The bloated logit_scale norm dominated total_norm → clip_coeff ≈ 0
           → ctx.grad multiplied by ~0 → near-zero ctx update (no learning).
        3. optimizer.zero_grad() did not zero logit_scale.grad, so NaN
           gradients from skipped steps accumulated across batches.

    Fix: self.logit_scale.requires_grad = False in CLIPCoOpEmotion.__init__.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_logit_scale_is_frozen(self):
        """Fix verified: logit_scale.requires_grad must be False."""
        model = _build_clip_coop(n_ctx=4)
        self.assertFalse(
            model.logit_scale.requires_grad,
            "logit_scale.requires_grad should be False — it is not in the "
            "optimizer and must not pollute clip_grad_norm_.",
        )
        print(f"\n  [OK] logit_scale.requires_grad = {model.logit_scale.requires_grad}")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_logit_scale_has_no_grad_after_backward(self):
        """With logit_scale frozen, backward must not set its grad."""
        model = _build_clip_coop(n_ctx=4)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler()
        images = torch.randn(4, 3, 224, 224, device=DEVICE)
        labels = torch.arange(4, device=DEVICE) % N_CLASSES

        optimizer.zero_grad()
        with amp.autocast():
            out = model(images, labels)
            loss = F.cross_entropy(out["logits"], labels)
        scaler.scale(loss).backward()

        self.assertIsNone(
            model.logit_scale.grad,
            "logit_scale.grad should be None after backward (param is frozen)",
        )
        print(f"\n  [OK] logit_scale.grad after backward: None (frozen, no grad computed)")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_clip_grad_norm_not_polluted_by_logit_scale(self):
        """
        With logit_scale frozen, clip_grad_norm_(model.parameters()) sees only
        the unscaled ctx gradient — total_norm should be small and clip_coeff
        should not be near zero.
        """
        model = _build_clip_coop(n_ctx=4)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler(init_scale=65536)
        images = torch.randn(4, 3, 224, 224, device=DEVICE)
        labels = torch.arange(4, device=DEVICE) % N_CLASSES

        optimizer.zero_grad()
        with amp.autocast():
            out = model(images, labels)
            loss = F.cross_entropy(out["logits"], labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        ctx_grad_norm = model.prompt_learner.ctx.grad.norm().item()
        # logit_scale.grad is None so total_norm == ctx_grad_norm
        total_norm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ) ** 0.5

        clip_coeff = 1.0 / (total_norm + 1e-6)

        print(
            f"\n  Scale factor:         {scaler.get_scale():.0f}\n"
            f"  ctx.grad norm:        {ctx_grad_norm:.6f}\n"
            f"  Total norm (all):     {total_norm:.6f}  (== ctx norm, logit_scale frozen)\n"
            f"  clip_coeff:           {clip_coeff:.4f}  (not near zero)"
        )

        # Without logit_scale pollution the total_norm must equal ctx_grad_norm
        self.assertAlmostEqual(
            total_norm, ctx_grad_norm, places=4,
            msg="total_norm should equal ctx_grad_norm when logit_scale is frozen",
        )
        # clip_coeff must not be near zero (was ~1.5e-5 with the bug)
        self.assertGreater(clip_coeff, 0.1,
                           "clip_coeff is near zero — logit_scale is still polluting the norm")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_ctx_gets_real_update_after_fix(self):
        """
        With logit_scale frozen and clip applied to optimizer params only,
        ctx must receive a non-trivial update after one AMP step.
        """
        model = _build_clip_coop(n_ctx=4)
        ctx_init = model.prompt_learner.ctx.data.clone()
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler(init_scale=65536)
        images = torch.randn(8, 3, 224, 224, device=DEVICE)
        labels = torch.arange(8, device=DEVICE) % N_CLASSES

        # Use optimizer param groups for clip (the fix in train_emotion.py)
        params_to_clip = [p for g in optimizer.param_groups for p in g["params"]]
        _run_amp_training_step(model, optimizer, scaler, images, labels,
                               clip_params=params_to_clip)
        delta = (model.prompt_learner.ctx.data - ctx_init).norm().item()

        print(f"\n  ctx delta norm after 1 fixed AMP step: {delta:.4f}")
        self.assertGreater(delta, 1e-4,
                           f"ctx delta {delta:.2e} is too small — update is still being zeroed")


# ---------------------------------------------------------------------------
# Bug B: AMP fp16 overflow makes text features NaN
# ---------------------------------------------------------------------------

class TestAMPTextFeatureNaN(unittest.TestCase):
    """
    Under AMP autocast the text transformer runs in fp16. With the near-zero
    ctx update from Bug A's one successful step, AdamW's state is initialised
    with v≈0. When the next step (if any) updates ctx, the effective step can
    be larger than expected (denominator ≈ eps). Combined with fp16 precision,
    this can drive text features to NaN.

    More directly: with AMP enabled and a large initial scale, the single
    successful step may apply a gradient that destabilises the fp16 text encoder.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_text_features_finite_before_training(self):
        """Baseline: text features must be finite before any training."""
        model = _build_clip_coop(n_ctx=4)
        finite = _text_features_are_finite(model)
        print(f"\n  Text features finite before training: {finite}")
        self.assertTrue(finite, "Text features should be finite before training")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_amp_training_may_produce_nan_text_features(self):
        """
        Simulate the training loop as in train_emotion.py (AMP enabled, clip
        all params, high init_scale).  After several steps, check whether text
        features become NaN. This test DOCUMENTS the observed behaviour and
        passes when the bug is present (NaN detected) or absent (all finite).
        """
        model = _build_clip_coop(n_ctx=4)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler(init_scale=65536)

        nan_found_at = None
        for step in range(15):
            images = torch.randn(8, 3, 224, 224, device=DEVICE)
            labels = torch.arange(8, device=DEVICE) % N_CLASSES
            _run_amp_training_step(model, optimizer, scaler, images, labels,
                                   clip_params=model.parameters())  # buggy
            if not _text_features_are_finite(model):
                nan_found_at = step
                break

        print(
            f"\n  Text feature NaN appeared at step: "
            f"{nan_found_at if nan_found_at is not None else 'never (all finite)'}"
        )
        print(f"  Final scaler scale: {scaler.get_scale():.0f}")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_amp_false_keeps_text_features_finite(self):
        """
        With amp=False (fp32 throughout), text features must remain finite
        across 15 training steps, even with the buggy clip_grad_norm_ call.
        """
        model = _build_clip_coop(n_ctx=4)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)

        for step in range(15):
            images = torch.randn(8, 3, 224, 224, device=DEVICE)
            labels = torch.arange(8, device=DEVICE) % N_CLASSES

            optimizer.zero_grad()
            # No autocast — fp32 throughout
            out = model(images, labels)
            loss = F.cross_entropy(out["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            finite = _text_features_are_finite(model)
            if not finite:
                self.fail(f"Text features became NaN at step {step} even with amp=False")

        print(f"\n  [PASS] Text features finite for all 15 steps with amp=False")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_fix_freeze_logit_scale_and_clip_prompt_only_keeps_features_finite(self):
        """
        Fix: freeze logit_scale + clip only prompt_learner params.
        Text features must stay finite across 15 AMP steps.
        """
        model = _build_clip_coop(n_ctx=4, freeze_logit_scale=True)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler(init_scale=65536)

        for step in range(15):
            images = torch.randn(8, 3, 224, 224, device=DEVICE)
            labels = torch.arange(8, device=DEVICE) % N_CLASSES
            _run_amp_training_step(
                model, optimizer, scaler, images, labels,
                clip_params=model.prompt_learner.parameters(),  # fixed
            )
            finite = _text_features_are_finite(model)
            if not finite:
                self.fail(
                    f"Text features NaN at step {step} even with fix applied "
                    "(freeze logit_scale + clip prompt only)"
                )

        print(
            f"\n  [PASS] Text features finite for all 15 AMP steps "
            f"with logit_scale frozen and clip applied to prompt_learner only"
        )


# ---------------------------------------------------------------------------
# Regression: scaler stays stable when bugs are fixed
# ---------------------------------------------------------------------------

class TestScalerStabilityWithFixes(unittest.TestCase):
    """After applying both fixes the AMP scaler should not collapse."""

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_scaler_stable_with_fixes(self):
        """
        With logit_scale frozen and clip applied to prompt_learner only, the
        AMP scaler should not collapse to 0 within 20 steps.
        """
        model = _build_clip_coop(n_ctx=4, freeze_logit_scale=True)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler(init_scale=65536)

        skipped = 0
        for _ in range(20):
            images = torch.randn(8, 3, 224, 224, device=DEVICE)
            labels = torch.arange(8, device=DEVICE) % N_CLASSES
            decreased, _ = _run_amp_training_step(
                model, optimizer, scaler, images, labels,
                clip_params=model.prompt_learner.parameters(),
            )
            if decreased:
                skipped += 1

        final_scale = scaler.get_scale()
        print(
            f"\n  Skipped steps: {skipped}/20\n"
            f"  Final scaler scale: {final_scale:.0f}"
        )
        self.assertGreater(
            final_scale, 1.0,
            f"Scaler collapsed to {final_scale} — gradient overflow still happening",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_ctx_actually_updates_with_fixes(self):
        """With fixes applied, ctx must change meaningfully after 5 steps."""
        model = _build_clip_coop(n_ctx=4, freeze_logit_scale=True)
        ctx_init = model.prompt_learner.ctx.data.clone()

        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler(init_scale=65536)

        for _ in range(5):
            images = torch.randn(8, 3, 224, 224, device=DEVICE)
            labels = torch.arange(8, device=DEVICE) % N_CLASSES
            _run_amp_training_step(
                model, optimizer, scaler, images, labels,
                clip_params=model.prompt_learner.parameters(),
            )

        ctx_delta = (model.prompt_learner.ctx.data - ctx_init).norm().item()
        print(f"\n  ctx delta norm after 5 fixed steps: {ctx_delta:.6f}")
        self.assertGreater(
            ctx_delta, 1e-5,
            "ctx should have a non-trivial update after 5 steps with fixes applied",
        )


# ---------------------------------------------------------------------------
# Summary: what amp=False changes
# ---------------------------------------------------------------------------

class TestAMPVsNoAMP(unittest.TestCase):
    """Compare loss behaviour over 5 steps with amp=True vs amp=False."""

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_loss_stays_finite_without_amp(self):
        """Training loss must be finite for every step when amp=False."""
        model = _build_clip_coop(n_ctx=4)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)

        for step in range(5):
            images = torch.randn(8, 3, 224, 224, device=DEVICE)
            labels = torch.arange(8, device=DEVICE) % N_CLASSES
            optimizer.zero_grad()
            out = model(images, labels)
            loss = F.cross_entropy(out["logits"], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            self.assertTrue(
                math.isfinite(loss.item()),
                f"Loss became NaN at step {step} with amp=False — unexpected",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_amp_true_buggy_may_nan_within_15_steps(self):
        """
        Buggy training (amp=True, clip all params) — document whether NaN
        appears within 15 steps. The test always passes; it's for diagnosis.
        """
        model = _build_clip_coop(n_ctx=4)
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=2e-3)
        scaler = amp.GradScaler(init_scale=65536)

        losses = []
        for step in range(15):
            images = torch.randn(8, 3, 224, 224, device=DEVICE)
            labels = torch.arange(8, device=DEVICE) % N_CLASSES
            optimizer.zero_grad()
            with amp.autocast():
                out = model(images, labels)
                loss = F.cross_entropy(out["logits"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # buggy
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        nan_steps = [i for i, l in enumerate(losses) if not math.isfinite(l)]
        print(
            f"\n  Losses (amp=True, buggy): {[f'{l:.3f}' for l in losses]}\n"
            f"  NaN appeared at steps: {nan_steps if nan_steps else 'none'}\n"
            f"  Final scaler scale: {scaler.get_scale():.0f}"
        )


# ---------------------------------------------------------------------------
# MERU: logit_scale frozen + AMP stability
# ---------------------------------------------------------------------------

class TestMERUAMPNoNaN(unittest.TestCase):
    """
    MERU inherits logit_scale from CLIPBaseline (requires_grad=True by default)
    but MERUCoOpEmotion.forward never uses it. It is NOT in the optimizer, so:
      - optimizer.zero_grad() does not zero logit_scale.grad
      - scaler.unscale_() does not unscale logit_scale.grad

    Although no gradient flows to logit_scale through the MERU forward path,
    its requires_grad=True status violates the invariant that every non-optimizer
    param must be frozen. Fix: self.meru.logit_scale.requires_grad = False.

    These tests verify:
      1. logit_scale is frozen after __init__
      2. Text features (Euclidean and hyperbolic) stay finite under AMP
      3. AMP scaler stays stable when correct optimizer params are clipped
      4. ctx receives a real update after AMP steps
    """

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_meru_logit_scale_is_frozen(self):
        """logit_scale (inherited from CLIPBaseline) must be frozen in MERUCoOpEmotion."""
        model = _build_meru_coop(n_ctx=4)
        self.assertFalse(
            model.meru.logit_scale.requires_grad,
            "meru.logit_scale.requires_grad should be False — it is not in the optimizer "
            "and must be frozen to uphold the all-non-optimizer-params-frozen invariant.",
        )
        print(f"\n  [OK] meru.logit_scale.requires_grad = {model.meru.logit_scale.requires_grad}")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_meru_text_features_finite_before_training(self):
        """Euclidean and hyperbolic text features must be finite at init."""
        model = _build_meru_coop(n_ctx=4)
        finite = _text_features_finite_meru(model)
        print(f"\n  MERU text features finite before training: {finite}")
        self.assertTrue(finite, "MERU text features should be finite before any training")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_meru_text_features_finite_with_amp(self):
        """Text features must stay finite across 15 AMP steps with correct optimizer setup."""
        model = _build_meru_coop(n_ctx=4)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.prompt_learner.parameters(), "lr": 2e-3, "weight_decay": 1e-4},
                {
                    "params": [model.meru.curv, model.meru.visual_alpha, model.meru.textual_alpha],
                    "lr": 5e-4,
                    "weight_decay": 0.0,
                },
            ],
            betas=(0.9, 0.999),
        )
        scaler = amp.GradScaler(init_scale=65536)

        for step in range(15):
            images = torch.randn(8, 3, 224, 224, device=DEVICE)
            labels = torch.arange(8, device=DEVICE) % N_CLASSES

            optimizer.zero_grad()
            with amp.autocast():
                out = model(images, labels)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            params_to_clip = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
            scaler.step(optimizer)
            scaler.update()

            self.assertTrue(
                math.isfinite(loss.item()),
                f"MERU loss became NaN/inf at step {step} with AMP",
            )
            self.assertTrue(
                _text_features_finite_meru(model),
                f"MERU text features became NaN at step {step} with AMP",
            )

        print(
            f"\n  [PASS] MERU text features and loss finite for 15 AMP steps. "
            f"Final scaler scale: {scaler.get_scale():.0f}"
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_meru_scaler_stable_with_correct_clipping(self):
        """AMP scaler must not collapse when clipping optimizer params only (not model.parameters())."""
        model = _build_meru_coop(n_ctx=4)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.prompt_learner.parameters(), "lr": 2e-3, "weight_decay": 1e-4},
                {
                    "params": [model.meru.curv, model.meru.visual_alpha, model.meru.textual_alpha],
                    "lr": 5e-4,
                    "weight_decay": 0.0,
                },
            ],
            betas=(0.9, 0.999),
        )
        scaler = amp.GradScaler(init_scale=65536)
        skipped = 0

        for _ in range(20):
            images = torch.randn(8, 3, 224, 224, device=DEVICE)
            labels = torch.arange(8, device=DEVICE) % N_CLASSES
            optimizer.zero_grad()
            with amp.autocast():
                out = model(images, labels)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            params_to_clip = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1

        final_scale = scaler.get_scale()
        print(
            f"\n  MERU skipped steps: {skipped}/20\n"
            f"  Final scaler scale: {final_scale:.0f}"
        )
        self.assertGreater(
            final_scale, 1.0,
            f"MERU scaler collapsed to {final_scale} — gradient overflow still happening",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for AMP")
    def test_meru_ctx_gets_real_update(self):
        """MERU ctx must receive a non-trivial update after 5 AMP steps."""
        model = _build_meru_coop(n_ctx=4)
        ctx_init = model.prompt_learner.ctx.data.clone()
        optimizer = torch.optim.AdamW(
            [
                {"params": model.prompt_learner.parameters(), "lr": 2e-3, "weight_decay": 1e-4},
                {
                    "params": [model.meru.curv, model.meru.visual_alpha, model.meru.textual_alpha],
                    "lr": 5e-4,
                    "weight_decay": 0.0,
                },
            ],
            betas=(0.9, 0.999),
        )
        scaler = amp.GradScaler(init_scale=65536)

        for _ in range(5):
            images = torch.randn(8, 3, 224, 224, device=DEVICE)
            labels = torch.arange(8, device=DEVICE) % N_CLASSES
            optimizer.zero_grad()
            with amp.autocast():
                out = model(images, labels)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            params_to_clip = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
            scaler.step(optimizer)
            scaler.update()

        ctx_delta = (model.prompt_learner.ctx.data - ctx_init).norm().item()
        print(f"\n  MERU ctx delta norm after 5 AMP steps: {ctx_delta:.6f}")
        self.assertGreater(
            ctx_delta, 1e-5,
            "MERU ctx should have a non-trivial update after 5 AMP steps",
        )


if __name__ == "__main__":
    print("\n=== Running NaN / AMP overflow diagnostic tests ===\n")
    unittest.main(verbosity=2)
