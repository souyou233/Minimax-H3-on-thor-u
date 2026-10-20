# Shim module so ComfyUI's `--use-sage-attention` (attention_sage -> sageattn) actually
# routes to SpargeAttn block-sparse attention on NVIDIA Thor (cu130). Drop this file in the
# ComfyUI cwd (sys.path[0]); a locally built `spas_sage_attn` package is required for the speedup.
# Per-generation toggle: /workspace/sparge_topk with a float in (0,1] => SpargeAttn at that top-k;
# empty/0/off => exact SDPA (baseline); missing file => _DEFAULT_TOPK. Any error => SDPA fallback.
import torch
import torch.nn.functional as F

_TOGGLE = "/workspace/sparge_topk"
_DEFAULT_TOPK = 0.3   # default top-k when the toggle file is absent. File overrides; 0/off => baseline.
_sp = None
_sp_failed = False


def _get():
    global _sp, _sp_failed
    if _sp is not None:
        return _sp
    if _sp_failed:
        return None
    try:
        from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda as f
        _sp = f
    except Exception as e:
        _sp_failed = True
        print("[sageattn-shim] spas_sage_attn import failed, using SDPA:", repr(e)[:160], flush=True)
    return _sp


def _topk():
    try:
        with open(_TOGGLE) as fh:
            s = fh.read().strip()
    except Exception:
        return _DEFAULT_TOPK
    if s == "" or s.lower() in ("0", "off", "none", "baseline"):
        return None
    try:
        t = float(s)
    except ValueError:
        return _DEFAULT_TOPK
    return t if 0.0 < t <= 1.0 else None


def _sdpa(q, k, v, tensor_layout):
    if tensor_layout == "HND":
        return F.scaled_dot_product_attention(q, k, v)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


def sageattn(q, k, v, is_causal=False, tensor_layout="HND", sm_scale=None,
             smooth_k=False, attn_mask=None, **kw):
    topk = _topk()
    if topk is None or attn_mask is not None or is_causal:
        return _sdpa(q, k, v, tensor_layout)
    fn = _get()
    if fn is None:
        return _sdpa(q, k, v, tensor_layout)
    try:
        return fn(q.contiguous(), k.contiguous(), v.contiguous(), topk=topk,
                  is_causal=False, tensor_layout=tensor_layout, scale=sm_scale,
                  smooth_k=True, output_dtype=q.dtype)  # sparge needs km => smooth_k must be True
    except Exception as e:
        print("[sageattn-shim] sparge failed -> sdpa:", repr(e)[:220], flush=True)
        return _sdpa(q, k, v, tensor_layout)
