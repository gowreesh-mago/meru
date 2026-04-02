"""
Debug script: trace gradient flow through the fixed CLIPCoOpEmotion TextEncoder.

This script uses the actual TextEncoder.forward (no manual permutes) to confirm
that ctx.grad is non-zero after the NLD→LND permutation bug was fixed.

Run: python tests/debug_gradient_flow.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "CoOp"))

import torch
import torch.nn.functional as F
from meru.emotion.emotion_coop_models import CLIPCoOpEmotion
from meru.models import CLIPBaseline
from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder

EMOTION_NAMES = ["amusement","awe","contentment","excitement","anger","disgust","fear","sadness"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    base = CLIPBaseline(
        visual=build_timm_vit(arch="vit_base_patch16_224", global_pool="token", use_sincos2d_pos=True),
        textual=TransformerTextEncoder(arch="L12_W512", vocab_size=49408, context_length=77),
        embed_dim=512,
    )
    model = CLIPCoOpEmotion(base, EMOTION_NAMES, n_ctx=4, ctx_init="", class_token_position="end", csc=False)
    return model.to(DEVICE)

def check(name, tensor):
    if tensor is None:
        print(f"  {name}: None")
    else:
        rg = tensor.requires_grad
        has_grad = tensor.grad_fn is not None
        print(f"  {name}: requires_grad={rg}, has_grad_fn={has_grad}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

print("=" * 70)
print("Building model...")
model = build_model()
print(f"ctx.requires_grad = {model.prompt_learner.ctx.requires_grad}")
print(f"ctx.shape = {model.prompt_learner.ctx.shape}")
print(f"logit_scale.requires_grad = {model.logit_scale.requires_grad}  (should be False)")
print()

# ---- Step 1: prompts from prompt_learner ----
print("Step 1: prompt_learner()")
prompts = model.prompt_learner()
check("prompts", prompts)

# ---- Step 2: text encoder forward via the FIXED TextEncoder.forward ----
print("\nStep 2: text encoder forward (no permutes — NLD throughout)")
tokenized_prompts = model.tokenized_prompts.to(DEVICE)
text_features = model.text_encoder(prompts, tokenized_prompts)
check("text_features (post-projection)", text_features)

# ---- Step 3: normalize and compute loss ----
print("\nStep 3: logits and loss")
images = torch.randn(8, 3, 224, 224, device=DEVICE)
labels = torch.arange(8, device=DEVICE) % len(EMOTION_NAMES)

image_feats = model.encode_image(images)
check("image_features (before norm)", image_feats)

img_norm = F.normalize(image_feats, dim=-1)
txt_norm = F.normalize(text_features, dim=-1)
check("image_features (normalized)", img_norm)
check("text_features (normalized)", txt_norm)

logit_scale = model.logit_scale.exp()
print(f"  logit_scale = {logit_scale.item():.4f}, requires_grad={model.logit_scale.requires_grad}")

logits = logit_scale * img_norm @ txt_norm.t()
check("logits", logits)
print(f"  logits.min={logits.min().item():.4f}, max={logits.max().item():.4f}")

loss = F.cross_entropy(logits, labels)
check("loss", loss)
print(f"  loss = {loss.item():.4f}")

# ---- Step 4: backward and check gradients ----
print("\nStep 4: backward pass")
loss.backward()

ctx_grad = model.prompt_learner.ctx.grad
if ctx_grad is None:
    print("  ctx.grad = None  ← BUG: GRADIENT NOT COMPUTED AT ALL")
else:
    norm = ctx_grad.norm().item()
    print(f"  ctx.grad norm = {norm:.8f}")
    if norm == 0.0:
        print("  ← BUG: GRADIENT IS ZERO (permutation bug active?)")
    else:
        print("  ← OK: ctx receives non-zero gradient")

logit_scale_grad = model.logit_scale.grad
if logit_scale_grad is not None:
    print(f"  logit_scale.grad = {logit_scale_grad.item():.8f}  ← BUG: should be None (frozen)")
else:
    print("  logit_scale.grad = None  ← OK (frozen)")

# ---- Step 5: gradient hooks on intermediate tensors via fixed forward ----
print("\nStep 5: gradient hooks tracing the fixed forward path")
model.zero_grad()

hook_results = {}
def make_hook(name):
    def hook(grad):
        hook_results[name] = grad.norm().item() if grad is not None else None
    return hook

prompts2 = model.prompt_learner()
prompts2.register_hook(make_hook("prompts"))

te = model.text_encoder
x_a = prompts2 + te.positional_embedding.type(te.dtype)
x_a.register_hook(make_hook("after_pos_embed"))

# Fixed: no permute — pass NLD directly to transformer with attn_mask
seq_len = x_a.shape[1]
attn_mask = te.attn_mask[:seq_len, :seq_len]
x_b = te.transformer(x_a, attn_mask)
x_b.register_hook(make_hook("after_transformer"))

x_c = te.ln_final(x_b).type(te.dtype)
x_c.register_hook(make_hook("after_ln_final"))

eot2 = model.tokenized_prompts.to(DEVICE).argmax(dim=-1)
x_d = x_c[torch.arange(x_c.shape[0]), eot2]
x_d.register_hook(make_hook("after_eot_index"))

x_e = x_d @ te._underlying_model.textual_proj.weight.T
x_e.register_hook(make_hook("after_text_proj"))

txt_n = F.normalize(x_e, dim=-1)
txt_n.register_hook(make_hook("text_features_norm"))

img_n = F.normalize(model.encode_image(images), dim=-1)
ls2 = model.logit_scale.exp()
logits2 = ls2 * img_n @ txt_n.t()
loss2 = F.cross_entropy(logits2, labels)
loss2.backward()

print("  Gradient norms at each stage (None = hook not called = no grad):")
for stage, norm in hook_results.items():
    status = f"{norm:.8f}" if norm is not None else "NOT CALLED"
    ok = "OK" if (norm is not None and norm > 0) else "ZERO/NONE"
    print(f"    {stage:30s}: {status}  [{ok}]")

final_ctx_norm = model.prompt_learner.ctx.grad.norm().item()
print(f"\n  ctx.grad norm after hooks: {final_ctx_norm:.8f}", end="")
if final_ctx_norm > 0:
    print("  ← OK: gradient flows to ctx")
else:
    print("  ← BUG: gradient is zero")
