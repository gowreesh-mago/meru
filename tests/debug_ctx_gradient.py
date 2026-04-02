"""
Trace gradient flow inside PromptLearner using the fixed TextEncoder.forward.

Previously prompts received grad but ctx.grad was zero due to the NLD→LND
permutation bug. This script verifies the fix: ctx must receive a non-zero
gradient through the full forward path.

Run: python tests/debug_ctx_gradient.py
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

base = CLIPBaseline(
    visual=build_timm_vit(arch="vit_base_patch16_224", global_pool="token", use_sincos2d_pos=True),
    textual=TransformerTextEncoder(arch="L12_W512", vocab_size=49408, context_length=77),
    embed_dim=512,
)
model = CLIPCoOpEmotion(base, EMOTION_NAMES, n_ctx=4, ctx_init="", class_token_position="end", csc=False)
model = model.to(DEVICE)

pl = model.prompt_learner

print("=" * 60)
print("Probing PromptLearner internals")
print(f"ctx.requires_grad = {pl.ctx.requires_grad}")
print(f"ctx.shape = {pl.ctx.shape}")
print(f"token_prefix.requires_grad = {pl.token_prefix.requires_grad}")
print(f"token_suffix.requires_grad = {pl.token_suffix.requires_grad}")
print(f"logit_scale.requires_grad = {model.logit_scale.requires_grad}  (should be False)")

# ---- Manually replicate PromptLearner.forward() with hooks ----
hooks = {}
def make_hook(name):
    def hook(grad):
        hooks[name] = (grad.norm().item(), grad.shape, grad[:, 1:5, :].norm().item() if grad.dim()==3 else None)
    return hook

ctx = pl.ctx                                     # leaf (4, 512)
ctx_u = ctx.unsqueeze(0)                         # (1, 4, 512)
ctx_u.register_hook(make_hook("ctx_unsqueezed"))

ctx_e = ctx_u.expand(8, -1, -1)                 # (8, 4, 512)
ctx_e.register_hook(make_hook("ctx_expanded"))

prefix = pl.token_prefix                         # (8, 1, 512) no grad
suffix = pl.token_suffix                         # (8, *, 512) no grad

prompts = torch.cat([prefix, ctx_e, suffix], dim=1)  # (8, 77, 512)
prompts.register_hook(make_hook("prompts"))

print(f"\nprompts.grad_fn = {prompts.grad_fn}")
print(f"ctx_e.grad_fn   = {ctx_e.grad_fn}")
print(f"ctx_u.grad_fn   = {ctx_u.grad_fn}")
print(f"ctx.is_leaf     = {ctx.is_leaf}")

# Now run through text encoder using the FIXED forward (no permutes)
te = model.text_encoder
tok = model.tokenized_prompts.to(DEVICE)

x = prompts + te.positional_embedding.type(te.dtype)
seq_len = x.shape[1]
attn_mask = te.attn_mask[:seq_len, :seq_len]
x = te.transformer(x, attn_mask)   # NLD throughout — no permute
x = te.ln_final(x).type(te.dtype)
eot = tok.argmax(dim=-1)
x = x[torch.arange(x.shape[0]), eot]
x = x @ te._underlying_model.textual_proj.weight.T

text_feats = F.normalize(x, dim=-1)

images = torch.randn(8, 3, 224, 224, device=DEVICE)
labels = torch.arange(8, device=DEVICE) % len(EMOTION_NAMES)
img_feats = F.normalize(model.encode_image(images), dim=-1)

logit_scale = model.logit_scale.exp()
logits = logit_scale * img_feats @ text_feats.t()
loss = F.cross_entropy(logits, labels)
print(f"\nloss = {loss.item():.4f}, loss.requires_grad = {loss.requires_grad}")
loss.backward()

print("\nGradient norms at each stage:")
for name, (norm, shape, ctx_slice_norm) in hooks.items():
    extra = f", ctx_positions[1:5] norm={ctx_slice_norm:.6f}" if ctx_slice_norm is not None else ""
    ok = "OK" if norm > 0 else "ZERO ← BUG"
    print(f"  {name:20s}: norm={norm:.6f}, shape={tuple(shape)}{extra}  [{ok}]")

ctx_grad = pl.ctx.grad
if ctx_grad is None:
    print(f"\nctx.grad = None  ← BUG")
else:
    norm = ctx_grad.norm().item()
    print(f"\nctx.grad norm = {norm:.8f}", end="")
    if norm > 0:
        print("  ← OK: non-zero gradient reaches ctx")
    else:
        print("  ← BUG: ctx.grad is all zeros")

# ---- Sanity check: simple differentiability of the pattern ----
print("\n" + "=" * 60)
print("Sanity check: does expand+cat->backward reach a leaf param?")
ctx2 = torch.randn(4, 512, requires_grad=True, device=DEVICE)
pre2 = torch.zeros(8, 1, 512, device=DEVICE)
suf2 = torch.zeros(8, 72, 512, device=DEVICE)
ctx2_e = ctx2.unsqueeze(0).expand(8, -1, -1)
p2 = torch.cat([pre2, ctx2_e, suf2], dim=1)  # (8, 77, 512)
loss2 = p2.sum()
loss2.backward()
print(f"ctx2.grad norm = {ctx2.grad.norm().item():.6f}  (expect non-zero)")
