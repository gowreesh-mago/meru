# Coding conventions

- Do not use `try`/`catch` or `try`/`except`.
- Prefer explicit validation and guard clauses.
- Fail fast with clear error returns or raised errors at the boundary only when necessary.
- Keep control flow linear; avoid exception-based branching.
- When rewriting existing code, remove `try`/`catch` unless I explicitly ask for it.
- Do not add imports inside functions; keep imports at the top of the file.
- Follow basic pylint principles where practical: keep modules organized, avoid unused imports, prefer readable names, and keep functions small and explicit.
- self verify your own edits before finishing
# Deep learning conventions

- When using PyTorch, NumPy, JAX, or any other library with randomness, prefer explicit random number generators with controllable seeding.
- Make randomness reproducible by threading seeds or generator objects through the call sites instead of relying on hidden global state.

# Project: Emotion Recognition with CoOp + MERU

## Repository layout

- `code/meru/` — primary working directory; contains all emotion-specific code
- `code/meru/meru/emotion/emotion_coop_models.py` — the single file that marries CoOp prompt learning with CLIP/MERU encoders; this is the only file to touch when changing model architecture
- `code/meru/scripts/train_emotion.py` — training entry point; handles dataset building, optimizer, AMP scaler, logging
- `code/meru/configs/emotion_clip_coop.py` / `emotion_meru_coop.py` / `emotion_emotionclipv2.py` — config dicts consumed by the training script. `emotion_clip_coop.py` now targets `CLIPCoOpOpenAI` (OpenAI CLIP backbone); the MERU-backbone `CLIPCoOpEmotion` path is still present in code but no longer has a default config
- `code/meru/tests/test_training_fixes.py` — 21-test suite covering gradient flow, AMP, class weights, MERU hyperbolic params, frozen-param invariant for both models
- `code/meru/tests/test_nan_amp.py` — 17-test suite regressing NaN/AMP overflow bugs for both CLIP and MERU
- `code/meru/tests/debug_gradient_flow.py` / `debug_ctx_gradient.py` — diagnostic scripts showing the fixed gradient path; run manually to inspect hook-by-hook gradient norms
- `CoOp/` and `code/meru/` are both on `sys.path`; CoOp code is an unmodified clone — never edit it directly
- `outputs/experiments/<run_name>/` — experiment outputs; each model variant gets its own subdirectory with `logs/` and `checkpoints/`

## Emotion+CoOp training vs vanilla MERU / CLIP / CoOp

Understanding what changes and what stays the same across the four systems is critical when debugging or extending the code.

### Vanilla CLIP pre-training
- **Goal**: learn aligned visual and text representations via contrastive loss on image–caption pairs
- **Trainable**: all visual encoder params, all text encoder params, `logit_scale`
- **Loss**: symmetric InfoNCE over the full batch
- **Text input**: raw captions tokenised to 77 tokens; `TextEncoder` permutes NLD→LND before OpenAI's sequence-first transformer
- **No PromptLearner**: class embeddings come directly from tokenised text

### Vanilla MERU pre-training
- **Goal**: same as CLIP but in Lorentzian (hyperbolic) space; adds entailment hierarchy (text entails image)
- **Trainable**: all visual/text encoder params + `curv`, `visual_alpha`, `textual_alpha`, `logit_scale`
- **Loss**: contrastive + entailment cone loss
- **Text input**: same pipeline as CLIP; MERU's `_TransformerBlock` uses `batch_first=True` so inputs stay in NLD (never permuted)
- **Similarity**: Lorentzian distance `−L.pairwise_dist(image, text, curv)` replaces cosine similarity

### Vanilla CoOp (on top of frozen CLIP)
- **Goal**: adapt CLIP to downstream tasks by learning a soft prompt prefix
- **Trainable**: only `PromptLearner.ctx` (a `n_ctx × embed_dim` tensor)
- **Frozen**: everything else — visual encoder, text transformer, `logit_scale`
- **Text input**: `[SOS | ctx | class_tokens | EOS]` → passes through CoOp's `TextEncoder` which permutes NLD→LND (correct for OpenAI CLIP's transformer)
- **Key invariant**: `logit_scale` in CLIP has `requires_grad=True` in the base model but must be frozen when not in the optimizer

### Emotion+CoOp with OpenAI backbone (CLIPCoOpOpenAI — added 2026-04-23)
- **Goal**: same task as `CLIPCoOpEmotion` but with OpenAI CLIP's pretrained weights instead of MERU's CLIPBaseline
- **Backbone**: `clip.load("ViT-B/32")` — auto-downloads to `~/.cache/clip`; **no pretrained checkpoint path needed**
- **Trainable**: only `prompt_learner.ctx`
- **Frozen**: full OpenAI CLIP (visual + text transformer + token_embedding + positional_embedding + text_projection + `logit_scale`)
- **TextEncoder**: uses vanilla CoOp's `TextEncoder` *unchanged* — correct because OpenAI CLIP's transformer is sequence-first (needs NLD↔LND permute). **Do not** reuse the custom no-permute `TextEncoder` in `emotion_coop_models.py` here; that one exists only for the MERU batch_first path.
- **Dtype**: `clip_model.float()` after load, so the whole model is fp32 and GradScaler.unscale_ works on ctx grads natively (simpler than EmotionCLIPV2's "freeze in fp16, cast trainables to fp32" pattern, but equivalent in outcome)
- **Config default**: `csc=False` (per CoOp paper — CSC overfits on small-class datasets like 8-class EmoSet)

### Emotion+CoOp (this project — CLIPCoOpEmotion / MERUCoOpEmotion)
- **Goal**: adapt CLIP or MERU to 8-class emotion classification using CoOp prompt learning
- **Trainable**:
  - CLIP variant: only `prompt_learner.ctx`
  - MERU variant: `prompt_learner.ctx` + `meru.curv` + `meru.visual_alpha` + `meru.textual_alpha`
- **Frozen**: visual encoder, text transformer/projection, **and `logit_scale`** (explicitly frozen in `__init__` — see bug #2 and #4 below)
- **Text input**: uses a custom `TextEncoder` that wraps MERU's transformer — **no NLD→LND permute** (MERU uses `batch_first=True`); passes the causal `attn_mask` from the pretrained MERU model
- **Similarity**: cosine (CLIP variant) or Lorentzian distance (MERU variant)
- **Loss**: standard cross-entropy; MERU variant additionally adds entailment cone loss
- **Optimizer**: `prompt_learner.parameters()` only for CLIP; two param groups for MERU (ctx at full LR, hyperbolic params at 0.25× LR)

### Key differences from vanilla CoOp
| | Vanilla CoOp | Emotion+CoOp (this project) |
|---|---|---|
| Base model | OpenAI CLIP | MERU's CLIPBaseline or MERU |
| TextEncoder permute | NLD→LND→NLD (correct for OpenAI) | **No permute** (MERU is batch_first) |
| `logit_scale` freeze | Not always explicit | **Explicitly frozen in `__init__`** |
| Grad clipping scope | `model.parameters()` | **Optimizer params only** |
| Loss | Standard CE | Standard CE |
| MERU hyperbolic params | N/A | Trainable, separate LR group |
| Entailment loss | N/A | Added for MERU variant |

## Architecture: how the pieces fit together

- **PromptLearner** (from CoOp): holds the learnable `ctx` tensor (shape `n_ctx × embed_dim`); concatenates `[SOS | ctx | class_tokens | EOS]` into full prompt embeddings
- **Two TextEncoders coexist in `emotion_coop_models.py`** — pick the right one per backbone:
  - Custom `TextEncoder` (no permute) — for MERU's `_TransformerBlock` (`batch_first=True`, NLD). Used by `CLIPCoOpEmotion` and `MERUCoOpEmotion`.
  - Vanilla CoOp `TextEncoder` (NLD↔LND permute) — for OpenAI CLIP's sequence-first transformer. Used by `CLIPCoOpOpenAI`. Imported directly from `CoOp/trainers/coop.py` — do not modify.
- **CLIPCoOpOpenAI / CLIPCoOpEmotion / MERUCoOpEmotion**: top-level wrappers; CLIPCoOpEmotion and MERUCoOpEmotion share the custom TextEncoder; CLIPCoOpOpenAI uses the vanilla one.
- **These two TextEncoder formats are incompatible**: never permute before passing to MERU's transformer blocks; always permute before passing to OpenAI CLIP's transformer.

## Frozen-param invariant

**Every param in the model must be either in the optimizer OR have `requires_grad=False`. No third state.**

For both `CLIPCoOpEmotion` and `MERUCoOpEmotion`, the only trainable params must be those explicitly placed in the optimizer. This is enforced in `__init__` by:
1. Freezing all visual/text encoder submodule params with a loop
2. Explicitly setting `requires_grad=False` on any scalar param (like `logit_scale`) that lives outside a submodule and is not in the optimizer

Violation consequences:
- `optimizer.zero_grad()` will not zero the stray param's gradient → it accumulates across batches
- `scaler.unscale_(optimizer)` will not unscale it → its gradient stays at AMP scale (e.g. 65536×)
- `clip_grad_norm_(optimizer_params, ...)` is safe only because we explicitly pass optimizer params; using `model.parameters()` would mix scaled and unscaled grads

**Checked by**:
- `tests/test_training_fixes.py::TestCtxGradientFlow::test_frozen_encoder_params_have_no_grad` (CLIP)
- `tests/test_training_fixes.py::TestCtxGradientFlow::test_meru_frozen_encoder_params_have_no_grad` (MERU)
- `tests/test_nan_amp.py::TestLogitScaleGradFix::test_logit_scale_is_frozen` (CLIP)
- `tests/test_nan_amp.py::TestMERUAMPNoNaN::test_meru_logit_scale_is_frozen` (MERU)

## Known bugs found and fixed (2026-04-01)

### 1. Illegal NLD→LND permutation in TextEncoder (critical)
- **Symptom**: val loss/acc completely frozen every epoch; ctx.grad = 0.0; model only predicts classes {0, 2}
- **Root cause**: `TextEncoder.forward` had `x = x.permute(1, 0, 2)` before the transformer. MERU's blocks use `batch_first=True`, so attention ran across the 8-class dimension instead of the 77-token sequence dimension. EOT output was mathematically independent of ctx positions 1–4 → zero gradient to ctx.
- **Fix**: Remove both permutes. Pass `attn_mask` (causal mask) stored from the underlying MERU model through `TransformerWrapper.forward`. See `TextEncoder.__init__` and `TextEncoder.forward` in `emotion_coop_models.py`.
- **Affects both models**: CLIPCoOpEmotion and MERUCoOpEmotion use the same TextEncoder class.

### 2. logit_scale unfrozen causes NaN ctx gradients under AMP (critical, CLIP)
- **Symptom**: Epoch 1 Batch 0 loss=2.53, Batch 10 loss=nan; AMP scaler collapsed to scale≈0; 31/32 optimizer steps skipped; ctx.grad=nan; text features NaN at validation
- **Root cause**: Two compounding bugs:
  1. `self.logit_scale = clip_model.logit_scale` stored logit_scale **with `requires_grad=True`** but it was never added to the optimizer. `optimizer.zero_grad()` only zeros optimizer params, so logit_scale.grad accumulates across batches (never zeroed). With pretrained CLIP weights, fp16 gradient overflow occurs at scale=65536; logit_scale.grad stays at AMP scale (~65536×true_grad) while ctx.grad is correctly unscaled after `scaler.unscale_(optimizer)`.
  2. `clip_grad_norm_(model.parameters(), clip)` mixes the still-scaled logit_scale.grad (≈65536×) with the unscaled ctx.grad. Total norm ≈ 65536 → clip_coeff ≈ 0 → **ctx.grad zeroed every step**. After enough steps, accumulated NaN in logit_scale.grad propagates into ctx.grad.
- **Fix in `emotion_coop_models.py`**: Add `self.logit_scale.requires_grad = False` immediately after storing logit_scale in `CLIPCoOpEmotion.__init__`.
- **Fix in `train_emotion.py`**: Clip only optimizer parameters, not all model parameters:
  ```python
  scaler.unscale_(optimizer)
  params_to_clip = [p for group in optimizer.param_groups for p in group["params"]]
  torch.nn.utils.clip_grad_norm_(params_to_clip, gradient_clip)
  ```
- **Key insight**: `scaler.unscale_()` only unscales gradients for params IN the optimizer. Any trainable param outside the optimizer retains its scaled gradient. Mixing scaled and unscaled grads in `clip_grad_norm_` silently zeros all gradients.
- **Tests**: `tests/test_nan_amp.py::TestLogitScaleGradFix` (4 tests)

### 3. AMP GradScaler skipping optimizer steps (CLIP config)
- **Symptom**: CLIP training logged no gradient updates; loss never changed
- **Root cause**: No gradient clipping → gradients overflow to inf in float16 → scaler halves its scale and skips the step entirely
- **Fix**: Add `gradient_clip_max_norm: 1.0` to the config dict; always call `scaler.unscale_(optimizer)` before the optional clip, then call `scaler.step(optimizer)` and `scaler.update()`. Log skipped steps per epoch.

### 5. MERU entailment loss discarded in training loop (critical)
- **Symptom**: `entail_weight=0.2` config has zero effect; `curv`, `visual_alpha`, `textual_alpha` only receive gradients from classification loss; hyperbolic cone geometry is never trained. Train/val loss curves are incomparable for MERU (val uses entailment, train did not).
- **Root cause**: `train_emotion.py:396` had `loss = F.cross_entropy(output["logits"], labels)`, overriding `output["loss"]` which already includes the entailment term (`contrastive + entail_weight * entailment`) computed in `MERUCoOpEmotion.forward`.
- **Fix**: Replace with `loss = output["loss"]` — works for both CLIP (which returns plain CE in `output["loss"]`) and MERU (which returns CE + entailment).

### 6. Missing logit_scale temperature scaling in MERUCoOpEmotion (high)
- **Symptom**: Softmax receives raw negative Lorentzian distances instead of temperature-scaled logits; wrong gradient magnitude and poorly calibrated class probabilities compared to vanilla MERU.
- **Root cause**: `MERUCoOpEmotion.forward` computed `logits = -distances` without multiplying by `logit_scale.exp()`. Vanilla MERU (`models.py:322-326`) always scales by `logit_scale.exp()`.
- **Fix**: `logit_scale = self.meru.logit_scale.exp(); logits = -distances * logit_scale` in `MERUCoOpEmotion.forward`. `logit_scale` is frozen so this is a constant temperature factor, not a new trainable.

### 7. Hyperbolic parameter clamping happens after optimizer step, not before forward (high)
- **Symptom**: Each forward pass uses whatever unclamped values the optimizer produced in the previous step; `exp_map0` can receive out-of-range alphas and blow up feature norms.
- **Root cause**: Clamping in `train_emotion.py` ran between `unscale_()` and `scaler.step()`. The optimizer step then applied gradients to the already-clamped values, potentially pushing them back out of range. The *next* forward used those unclamped post-step values.
- **Fix**: Move clamping into `MERUCoOpEmotion.forward()` at the very top, using `self.meru._curv_minmax` (the model's own bounds). Remove the training-loop clamping block from `train_emotion.py`. This matches vanilla MERU's pattern (`models.py:283-289`).

### 8. Curvature and alphas not clamped during validation (high)
- **Symptom**: If the last training step pushed `curv` or alphas out of valid range, the entire validation epoch runs with unclamped values, corrupting val metrics.
- **Root cause**: The clamping block in `train_emotion.py` was inside the training loop only.
- **Fix**: Resolved by the same fix as bug #7 — clamping inside `MERUCoOpEmotion.forward()` covers both training and validation forward passes.

### 4. MERU's inherited logit_scale violates frozen-param invariant
- **Symptom**: Latent — no NaN (logit_scale unused in MERUCoOpEmotion.forward), but `model.meru.logit_scale.requires_grad=True` with no optimizer entry
- **Root cause**: `MERU` inherits from `CLIPBaseline` which has `self.logit_scale = nn.Parameter(...)`. `MERUCoOpEmotion` stores `self.meru = meru_model` but only freezes submodule params (visual, textual, etc.); direct scalar Parameters like `logit_scale` are not in any freeze loop.
- **Fix**: Add `self.meru.logit_scale.requires_grad = False` in `MERUCoOpEmotion.__init__`, after storing `self.meru`.
- **Why it matters even though no grad flows**: violates the invariant; if `clip_grad_norm_(model.parameters(), ...)` were ever used instead of the optimizer-scoped version, it would trigger the same bug as bug #2.
- **Tests**: `tests/test_nan_amp.py::TestMERUAMPNoNaN` (5 tests), `tests/test_training_fixes.py::test_meru_frozen_encoder_params_have_no_grad`

## AMP + gradient clipping invariants

These must hold simultaneously or training silently breaks:

1. **Every trainable param is either in the optimizer OR frozen** — no third state. A param with `requires_grad=True` that is not in any optimizer param_group will not be zeroed by `optimizer.zero_grad()` and will not be unscaled by `scaler.unscale_()`. Its gradient accumulates at AMP scale across batches.
2. **`scaler.unscale_(optimizer)` before any `clip_grad_norm_`** — always, even if gradient_clip is None (unscaling is cheap and required for correct grad inspection).
3. **`clip_grad_norm_` must receive exactly the same set of params as the optimizer** — use `[p for group in optimizer.param_groups for p in group["params"]]`, not `model.parameters()`.
4. **Log `scaler.get_scale()` and skipped-step count every epoch** — if scale collapses below 1024 or skipped steps > 10%, stop and debug. Silent skip-storms are not recoverable by just continuing training.

## Debugging gradient flow

When ctx.grad is zero or training metrics are frozen, check in this order:

1. Confirm `ctx.requires_grad = True` and `ctx.is_leaf = True`
2. Register hooks on `prompts`, `after_pos_embed`, `after_transformer`, `after_eot_index` to find where norm drops to zero
3. Check that `prompts[:, 1:n_ctx+1, :]` slice has non-zero gradient (not just the full tensor)
4. If gradient reaches `prompts` but ctx.grad=0, the bug is inside `PromptLearner.forward` or the tensor operations between ctx and prompts
5. Use `tests/debug_gradient_flow.py` and `tests/debug_ctx_gradient.py` as starting templates (both updated to use the fixed no-permute forward; they show ctx.grad norm ≈ 1.36 when working correctly)

## Training diagnostics to add to logs

Log these every epoch to catch silent failures early:
- AMP scaler scale value and number of skipped steps
- ctx gradient norm
- Prediction class distribution (detect collapse to a subset of classes)
- Text feature cosine similarity mean and std (should not be near 1.0)

## MERU-specific notes

- MERU has three additional trainable scalars: `curv` (curvature), `visual_alpha`, `textual_alpha`
- These are trainable by default in `MERUCoOpEmotion` and placed in a separate optimizer param group at 0.25× learning rate
- MERU's optimizer has two param groups: `prompt_learner.parameters()` and `[meru.curv, meru.visual_alpha, meru.textual_alpha]`; `scaler.unscale_()` and `clip_grad_norm_` handle both groups correctly via the optimizer-scoped pattern
- MERU's config already includes `gradient_clip_max_norm: 1.0`; the AMP fix in the training script applies to both models via the same code path
- When accessing MERU params by name in tests or scripts, use `getattr(model.meru, attr)` for direct attribute access; `named_parameters()` lists them under both `meru.*` and `text_encoder._underlying_model.*` (same object, two paths)
- `MERU` inherits `logit_scale` from `CLIPBaseline` — it is **explicitly frozen** in `MERUCoOpEmotion.__init__` even though MERU's forward uses Lorentzian distance, not cosine similarity

## Backbone choice: OpenAI CLIP vs MERU CLIPBaseline (investigation 2026-04-23)

Observation: on EmoSet, `emotionclipv2_train.slurm` (OpenAI CLIP ViT-B/32 + 20-token prefix) beat `02_train_clip_coop.sh` (MERU CLIPBaseline + CoOp ctx) by a large margin, even though "both use prompt tuning."

Root causes, in order of impact:

1. **Pretraining corpus size dominates**: OpenAI CLIP was trained on ~400M image–text pairs; MERU's `CLIPBaseline` checkpoint comes from RedCaps (~12M pairs). For any downstream vision task including emotion recognition, the backbone gap swamps all prompt-tuning differences. This was the biggest driver.
2. **CSC overfits on small-class datasets**: `configs/emotion_clip_coop.py` originally set `csc=True` with `n_ctx=16`, giving 8×16×512 ≈ 65K trainable ctx params on 8 classes — CoOp paper explicitly warns against this on datasets with few classes. `CLIPCoOpOpenAI` defaults to `csc=False`.
3. **Architecture mismatch was not the cause** — both paths use the correct TextEncoder for their transformer; the permute bug was already fixed.
4. Input normalization is identical (both go through EmoSet's ImageNet-norm transforms). Not the issue.

**Takeaway when adding new emotion-task baselines**: start from OpenAI CLIP unless there's a specific reason to use MERU's pretrained weights. `CLIPCoOpOpenAI` is the reference implementation.

## CoOp quirks that bite

- **`ctx_init` silently overrides `n_ctx`**: `CoOp/trainers/coop.py:73-80` sets `n_ctx = len(ctx_init.split(" "))` when `ctx_init` is non-empty, ignoring the configured `n_ctx`. Example: `ctx_init="this picture conveys a sense of"` gives n_ctx=6 regardless of config. To train the full configured n_ctx with random init, pass `ctx_init=""`. The wrapper classes store the *final* value via `self.n_ctx = self.prompt_learner.n_ctx` so wandb logs the truth.
- **PromptLearner registers `token_prefix` / `token_suffix` as buffers in `clip_model.dtype`** — if the backbone is fp16 and you only convert `ctx` to fp32, the `torch.cat([prefix, ctx, suffix])` in `PromptLearner.forward` will upcast and may break downstream transformer dtype assumptions. Two safe patterns:
  - Cast the whole backbone to fp32 via `clip_model.float()` before passing it to `PromptLearner` (used by `CLIPCoOpOpenAI`).
  - Keep the backbone fp16 and handle dtype casts explicitly in a custom prompt module (used by `EmotionCLIPV2Emotion` with its own `Prompt_block`).
  Do not mix: a fp16 PromptLearner with a fp32 ctx param is a dtype trap.
- **`clip.load()` auto-downloads** to `~/.cache/clip` on first call. Compute nodes without internet must pre-download on the login node (or set `CLIP_HOME`). This is why `scripts/emotion_experiments/02_train_clip_coop.sh` no longer requires `PRETRAINED_CLIP`.
