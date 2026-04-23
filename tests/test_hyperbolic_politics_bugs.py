"""
Tests verifying three hypotheses for why PoliticsHyperbolicClassifier fails to learn.

Bug 1 — CosineAnnealingLR T_max mismatch
  T_max=epochs (30) but scheduler.step() is called per-batch (~3560 steps/epoch).
  LR completes a full cosine cycle every 60 batches instead of over the full training
  run. Net effective LR is ~0.5x with wild oscillation; training makes near-zero
  progress per epoch.

Bug 2 — Prototypes pinned near the hyperboloid origin
  init: 0.02 * randn(N, 512) → tangent-space norm ≈ 0.45 → hyperbolic spatial norm
  ≈ 0.47. textual_alpha is clamped ≤ 0 throughout training, so exp(alpha) ≤ 1 —
  prototypes can only be scaled DOWN, never UP. The optimizer pushes alpha more
  negative, actively compressing prototypes toward the origin. Result: all 22
  prototypes stay near the hyperboloid "north pole" → near-uniform Lorentzian
  distances to all images → near-uniform logits (std ≈ 0.02) → classification
  signal is almost noise.

Bug 3 — Entailment gradient dominates CE gradient at initialisation
  Tiny protos produce moderate cone apertures (~25°). Random proto placement yields
  oxy_angle >> aperture → 100% entailment violations at init → large entailment
  gradient (≈ 0.61 norm). CE gradient is much weaker (≈ 0.12 norm) because
  near-uniform logits cause large gradient cancellation across the 20 classes.
  Unweighted entailment gradient is ~5× the CE gradient; with entail_weight=0.2 the
  two contributions are roughly equal. The optimizer spends its budget satisfying
  the geometric hierarchy rather than separating classes. By epoch 7 violations have
  dropped to ~0.9% but topic accuracy is still at random chance (10%).

Run from repo root:
    python -m pytest tests/test_hyperbolic_politics_bugs.py -v
"""

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import meru.lorentz as L

# ---------------------------------------------------------------------------
# Shared constants matching the actual model
# ---------------------------------------------------------------------------

NUM_INCL = 2
NUM_TOPIC = 20
D = 512
BATCH = 64
CURV = torch.tensor(1.0)
ENTAIL_WEIGHT = 0.2


def _make_protos(n: int, scale: float, seed: int = 42) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(n, D, generator=g) * scale


def _to_hyperboloid(tangent: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    """Same formula as PoliticsHyperbolicClassifier._to_hyperbolic."""
    return L.exp_map0(tangent * math.exp(alpha), CURV)


def _fake_img_hyp(batch: int = BATCH, seed: int = 7) -> torch.Tensor:
    """Random image points on the hyperboloid (unit-norm tangent then exp_map0)."""
    g = torch.Generator()
    g.manual_seed(seed)
    tangent = F.normalize(torch.randn(batch, D, generator=g), dim=-1)
    return L.exp_map0(tangent, CURV)


def _random_labels(n: int, num_classes: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(0, num_classes, (n,), generator=g)


# ============================================================================
# Bug 1 — CosineAnnealingLR T_max mismatch
# ============================================================================

class TestSchedulerTMaxBug(unittest.TestCase):
    """
    CosineAnnealingLR with T_max=epochs but stepped per-batch causes the LR to
    cycle through its full range every 2*epochs batches instead of over the whole
    training run.
    """

    def _make_scheduler(self, t_max: int, lr: float = 1e-3):
        param = nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=lr)
        # Call optimizer.step() once so PyTorch doesn't warn about ordering
        opt.zero_grad()
        (param * 0).sum().backward()
        opt.step()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
        return opt, sched

    def test_tmax_epochs_reaches_zero_in_epochs_steps(self):
        """With T_max=epochs=30, LR drops to ~0 after only 30 scheduler steps.

        In the buggy script, 30 steps = 30 batches ≈ 8 seconds of training —
        not one epoch. LR should still be near its maximum at this point.
        """
        epochs = 30
        _, sched = self._make_scheduler(t_max=epochs)

        for _ in range(epochs):
            sched.step()

        lr = sched.get_last_lr()[0]
        self.assertLess(lr, 1e-6,
            f"T_max=epochs: LR hits zero in {epochs} batches: lr={lr:.2e}")

    def test_tmax_epochs_causes_many_cycles_per_epoch(self):
        """T_max=30 means ~59 complete LR cycles per epoch (228k/64 ≈ 3562 batches)."""
        epochs = 30
        batches_per_epoch = 3562
        _, sched = self._make_scheduler(t_max=epochs)

        lrs = []
        for _ in range(batches_per_epoch):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        lrs_t = torch.tensor(lrs)
        median = lrs_t.median().item()
        # Count downward zero-crossings of median; each ≈ half a LR cycle
        crossings = ((lrs_t[:-1] > median) & (lrs_t[1:] <= median)).sum().item()
        self.assertGreater(crossings, 50,
            f"Expected >50 LR half-cycles per epoch with T_max=epochs, got {crossings}")

    def test_correct_tmax_lr_barely_moves_in_one_epoch(self):
        """With T_max = epochs * batches_per_epoch, LR stays near max for the
        first epoch (as intended for cosine annealing over the full run)."""
        epochs = 30
        batches_per_epoch = 3562
        lr_init = 1e-3
        _, sched = self._make_scheduler(t_max=epochs * batches_per_epoch, lr=lr_init)

        for _ in range(batches_per_epoch):
            sched.step()

        lr = sched.get_last_lr()[0]
        # After 1/30 of the schedule, LR should still be >97% of initial
        self.assertGreater(lr, 0.97 * lr_init,
            f"Correct T_max: after 1 epoch LR should be near max, got {lr:.6f}")

    def test_correct_tmax_lr_monotonically_decreases(self):
        """With the correct T_max, LR is monotonically non-increasing."""
        epochs = 30
        batches_per_epoch = 50  # small for test speed
        _, sched = self._make_scheduler(t_max=epochs * batches_per_epoch)

        lrs = []
        for _ in range(epochs * batches_per_epoch):
            sched.step()
            lrs.append(sched.get_last_lr()[0])

        for i in range(1, len(lrs)):
            self.assertGreaterEqual(lrs[i - 1] + 1e-12, lrs[i],
                f"LR increased at step {i}: {lrs[i-1]:.6f} → {lrs[i]:.6f}")


# ============================================================================
# Bug 2 — Prototypes pinned near the hyperboloid origin
# ============================================================================

class TestPrototypeNearOrigin(unittest.TestCase):
    """
    Prototype initialisation at 0.02*randn + textual_alpha clamped ≤ 0 keeps
    all prototypes near the hyperboloid north pole throughout training.
    """

    def test_tiny_init_small_hyperbolic_norms(self):
        """0.02*randn → hyperbolic spatial norm < 1.0 for all 20 topic prototypes."""
        topic_proto = _make_protos(NUM_TOPIC, scale=0.02)
        norms = _to_hyperboloid(topic_proto).norm(dim=-1)
        self.assertTrue(
            (norms < 1.0).all(),
            f"Tiny-init protos should all be near origin. "
            f"max norm={norms.max():.4f}, mean={norms.mean():.4f}"
        )

    def test_alpha_clamp_can_only_shrink_protos(self):
        """exp(alpha) with alpha ≤ 0 means protos can only be scaled DOWN."""
        topic_proto = _make_protos(NUM_TOPIC, scale=0.02)
        norm_at_zero = _to_hyperboloid(topic_proto, alpha=0.0).norm(dim=-1).mean()

        for alpha_val in [-0.1, -0.5, -1.0, -2.0, -5.0]:
            norm_neg = _to_hyperboloid(topic_proto, alpha=alpha_val).norm(dim=-1).mean()
            self.assertLessEqual(
                norm_neg.item(), norm_at_zero.item() + 1e-5,
                f"alpha={alpha_val}: norm should be ≤ norm at alpha=0, "
                f"got {norm_neg:.6f} vs {norm_at_zero:.6f}"
            )

    def test_near_uniform_logits_with_tiny_init(self):
        """With tiny init prototypes, logit std across classes is near zero (≈0.02)."""
        topic_proto = _make_protos(NUM_TOPIC, scale=0.02)
        topic_hyp = _to_hyperboloid(topic_proto)
        img_hyp = _fake_img_hyp()

        logits = -L.pairwise_dist(img_hyp, topic_hyp, CURV)
        logit_std = logits.std(dim=-1).mean().item()

        self.assertLess(logit_std, 0.05,
            f"Logit std should be near-zero with tiny protos, got {logit_std:.4f}")

    def test_near_maximum_entropy_with_tiny_init(self):
        """With tiny init, cross-entropy softmax entropy approaches log(20) ≈ 3.0."""
        topic_proto = _make_protos(NUM_TOPIC, scale=0.02)
        topic_hyp = _to_hyperboloid(topic_proto)
        img_hyp = _fake_img_hyp()

        logits = -L.pairwise_dist(img_hyp, topic_hyp, CURV)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
        max_entropy = math.log(NUM_TOPIC)

        self.assertGreater(entropy, 0.97 * max_entropy,
            f"Near-maximum entropy expected. H={entropy:.4f}, log(20)={max_entropy:.4f}")

    # ── Fix verification ─────────────────────────────────────────────────────

    def test_larger_init_gives_more_diverse_logits(self):
        """With 0.1*randn init (proposed fix), logit std is >4× larger (≈0.08 vs 0.02)."""
        tiny_proto = _make_protos(NUM_TOPIC, scale=0.02)
        large_proto = _make_protos(NUM_TOPIC, scale=0.1)
        img_hyp = _fake_img_hyp()

        logits_tiny = -L.pairwise_dist(img_hyp, _to_hyperboloid(tiny_proto), CURV)
        logits_large = -L.pairwise_dist(img_hyp, _to_hyperboloid(large_proto), CURV)
        std_tiny = logits_tiny.std(dim=-1).mean().item()
        std_large = logits_large.std(dim=-1).mean().item()

        self.assertGreater(std_large, std_tiny * 2,
            f"Larger init should give >2× logit std: tiny={std_tiny:.4f}, large={std_large:.4f}")
        self.assertGreater(std_large, 0.06,
            f"Larger init logit std should be meaningfully above 0: {std_large:.4f}")

    def test_unclamped_protos_can_expand(self):
        """Prototypes projected without alpha scaling have spatial norm > 2 (well separated
        from the hyperboloid origin), letting the CE gradient drive class separation."""
        topic_proto = _make_protos(NUM_TOPIC, scale=0.1)
        # Direct exp_map0 without alpha scaling (proposed fix)
        topic_hyp = L.exp_map0(topic_proto, CURV)
        norms = topic_hyp.norm(dim=-1)

        self.assertTrue(
            (norms > 2.0).all(),
            f"Larger init without alpha should place protos far from origin. "
            f"min norm={norms.min():.4f}"
        )


# ============================================================================
# Bug 3 — Entailment gradient dominates CE gradient at initialisation
# ============================================================================

class TestEntailmentDominatesCE(unittest.TestCase):
    """
    At tiny init, the oxy_angle between randomly placed prototypes far exceeds
    their ~25° cone aperture, giving 100% violations and a large entailment
    gradient (~0.61 norm). The CE gradient is much weaker (~0.12) due to
    near-uniform logits that cause large gradient cancellation across 20 classes.
    Unweighted entailment gradient is ~5× CE gradient. With entail_weight=0.2
    the two contributions are roughly equal in the optimizer, so the gradient
    budget is split between geometry and classification rather than focusing on
    classification.
    """

    def _entailment_data(self, scale: float, seed: int = 42):
        """Return prototypes and labels for entailment calculations."""
        g = torch.Generator()
        g.manual_seed(seed)
        incl_proto = torch.randn(NUM_INCL, D, generator=g) * scale
        topic_proto = torch.randn(NUM_TOPIC, D, generator=g) * scale
        incl_hyp = _to_hyperboloid(incl_proto)
        topic_hyp = _to_hyperboloid(topic_proto)
        topic_labels = _random_labels(BATCH, NUM_TOPIC, seed=0)
        incl_labels = _random_labels(BATCH, NUM_INCL, seed=1)
        return incl_hyp, topic_hyp, topic_labels, incl_labels

    def test_violations_100pct_at_tiny_init(self):
        """Random proto placement yields oxy_angle >> 25° aperture → 100% t2incl violations."""
        incl_hyp, topic_hyp, topic_labels, incl_labels = self._entailment_data(scale=0.02)
        t_b = topic_hyp[topic_labels]
        i_b = incl_hyp[incl_labels]
        violations = (L.oxy_angle(t_b, i_b, CURV) > L.half_aperture(t_b, CURV)).float().mean()
        self.assertGreater(violations.item(), 0.9,
            f"Expected ~100% t2incl violations at tiny init, got {violations.item():.3f}")

    def test_entailment_loss_large_at_tiny_init(self):
        """t2incl entailment loss is large at init (~2.0), not near-zero."""
        incl_hyp, topic_hyp, topic_labels, incl_labels = self._entailment_data(scale=0.02)
        t_b = topic_hyp[topic_labels]
        i_b = incl_hyp[incl_labels]
        loss = torch.clamp(
            L.oxy_angle(t_b, i_b, CURV) - L.half_aperture(t_b, CURV), min=0
        ).mean().item()
        self.assertGreater(loss, 1.0,
            f"Entailment loss should be large at init (~2.0), got {loss:.4f}")

    def test_tiny_protos_have_moderate_aperture(self):
        """Tiny protos produce ~25° apertures — not trivially wide, not trivially narrow.

        This moderate aperture combined with random ~90° oxy_angles causes 100% violations.
        """
        topic_proto = _make_protos(NUM_TOPIC, scale=0.02)
        topic_hyp = _to_hyperboloid(topic_proto)
        apertures_deg = L.half_aperture(topic_hyp, CURV).mul(180 / math.pi)

        self.assertTrue(
            (apertures_deg > 15).all() and (apertures_deg < 45).all(),
            f"Apertures should be ~25° (not trivially wide or narrow). "
            f"range=[{apertures_deg.min():.1f}°, {apertures_deg.max():.1f}°]"
        )

    def test_unweighted_entailment_grad_exceeds_ce_grad(self):
        """Unweighted entailment gradient (~0.61) is ~5× CE gradient (~0.12).

        With entail_weight=0.2, the weighted contributions to the total gradient are
        roughly equal — meaning the gradient budget is split rather than dominated by CE.
        """
        img_hyp = _fake_img_hyp().detach()
        topic_labels = _random_labels(BATCH, NUM_TOPIC)
        incl_labels = _random_labels(BATCH, NUM_INCL)

        # CE gradient
        tp_ce = nn.Parameter(_make_protos(NUM_TOPIC, scale=0.02, seed=1))
        logits = -L.pairwise_dist(img_hyp, _to_hyperboloid(tp_ce), CURV)
        F.cross_entropy(logits, topic_labels).backward()
        ce_grad_norm = tp_ce.grad.norm().item()

        # Entailment gradient (t2incl only)
        tp_ent = nn.Parameter(_make_protos(NUM_TOPIC, scale=0.02, seed=1))
        ip_ent = nn.Parameter(_make_protos(NUM_INCL, scale=0.02, seed=2))
        t_b = _to_hyperboloid(tp_ent)[topic_labels]
        i_b = _to_hyperboloid(ip_ent)[incl_labels]
        entail = torch.clamp(
            L.oxy_angle(t_b, i_b, CURV) - L.half_aperture(t_b, CURV), min=0
        ).mean()
        entail.backward()
        ent_grad_norm = tp_ent.grad.norm().item()

        self.assertGreater(ent_grad_norm, ce_grad_norm,
            f"Entailment grad ({ent_grad_norm:.4f}) should exceed CE grad ({ce_grad_norm:.4f})")
        # Specifically expect ~5× ratio based on actual measurements
        self.assertGreater(ent_grad_norm / ce_grad_norm, 3.0,
            f"Ratio entailment/CE should be ~5×, got {ent_grad_norm/ce_grad_norm:.2f}")

    def test_weighted_entailment_rivals_ce_contribution(self):
        """With entail_weight=0.2, weighted entailment gradient ≈ CE gradient.

        The optimizer does not prioritise classification — half of the gradient
        budget goes to satisfying geometric constraints.
        """
        img_hyp = _fake_img_hyp().detach()
        topic_labels = _random_labels(BATCH, NUM_TOPIC)
        incl_labels = _random_labels(BATCH, NUM_INCL)

        # CE gradient
        tp = nn.Parameter(_make_protos(NUM_TOPIC, scale=0.02, seed=1))
        F.cross_entropy(-L.pairwise_dist(img_hyp, _to_hyperboloid(tp), CURV),
                        topic_labels).backward()
        ce_norm = tp.grad.norm().item()

        # Weighted entailment gradient
        tp2 = nn.Parameter(_make_protos(NUM_TOPIC, scale=0.02, seed=1))
        ip2 = nn.Parameter(_make_protos(NUM_INCL, scale=0.02, seed=2))
        entail = torch.clamp(
            L.oxy_angle(_to_hyperboloid(tp2)[topic_labels],
                        _to_hyperboloid(ip2)[incl_labels], CURV)
            - L.half_aperture(_to_hyperboloid(tp2)[topic_labels], CURV),
            min=0
        ).mean()
        (ENTAIL_WEIGHT * entail).backward()
        weighted_ent_norm = tp2.grad.norm().item()

        # Weighted entailment should be within 2× of CE gradient
        ratio = weighted_ent_norm / ce_norm
        self.assertGreater(ratio, 0.5,
            f"Weighted entailment grad ({weighted_ent_norm:.4f}) should be "
            f"comparable to CE grad ({ce_norm:.4f}), ratio={ratio:.2f}")
        self.assertLess(ratio, 2.0,
            f"Weighted entailment grad should not dominate 2×, ratio={ratio:.2f}")

    # ── Fix verification ─────────────────────────────────────────────────────

    def test_larger_init_more_diverse_logits_for_classification(self):
        """With 0.1*randn, logit std is 4× larger — CE gradient has clearer direction."""
        img_hyp = _fake_img_hyp()
        std_tiny = (-L.pairwise_dist(img_hyp, _to_hyperboloid(_make_protos(NUM_TOPIC, 0.02)), CURV)
                    ).std(dim=-1).mean().item()
        std_large = (-L.pairwise_dist(img_hyp, _to_hyperboloid(_make_protos(NUM_TOPIC, 0.1)), CURV)
                     ).std(dim=-1).mean().item()

        self.assertGreater(std_large / std_tiny, 3.0,
            f"Larger init should give >3× logit std: tiny={std_tiny:.4f}, large={std_large:.4f}")

    def test_removing_alpha_from_protos_allows_free_growth(self):
        """Projecting protos via exp_map0 without alpha lets the optimizer grow them freely.

        With alpha clamped ≤ 0, scale factor exp(alpha) can only shrink protos.
        Without alpha, the raw proto parameters can grow to any scale via gradient descent.
        """
        # With alpha=-1 (common after training), protos are compressed to 37% of their size
        proto = _make_protos(NUM_TOPIC, scale=0.1)
        scale_with_alpha = math.exp(-1.0)
        norm_with_alpha = _to_hyperboloid(proto, alpha=-1.0).norm(dim=-1).mean()
        norm_without_alpha = L.exp_map0(proto, CURV).norm(dim=-1).mean()

        self.assertGreater(norm_without_alpha.item(), norm_with_alpha.item(),
            f"No-alpha protos have larger norms: {norm_without_alpha:.4f} "
            f"vs with-alpha: {norm_with_alpha:.4f}")
        # No-alpha protos are at their natural scale (not artificially compressed)
        self.assertGreater(norm_without_alpha.item(), 2.0,
            f"No-alpha protos should be well off the origin: {norm_without_alpha:.4f}")


if __name__ == "__main__":
    unittest.main()
