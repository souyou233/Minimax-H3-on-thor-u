# Running MiniMax H3 on NVIDIA Thor (sm_101) with CUDA 13 — a field guide

> 中文版见 **[GUIDE.zh.md](GUIDE.zh.md)**。

A production write-up of running the **MiniMax H3** audio-video diffusion model
(text/image → video + audio) on an **NVIDIA Thor** SoC (the Zeekr Thor-U automotive unit),
including the CUDA-12.8 → CUDA-13 migration that unlocked native int8/fp8 kernels, measured
speed/memory across model configs, and the full list of gotchas hit along the way.

> **TL;DR** — The chip is **NVIDIA Thor (sm_101)**. On CUDA 12.8 `comfy_kitchen`'s
> `cuda_version < (13,)` gate *hard-disables* its CUDA backend, so int8 weights fall back to eager
> dequant: slow (5 s clip ≈ 620 s sampling) and it **OOMs at 10 s**. Booting the same stack under
> **CUDA 13 forward-compatibility** passes that gate → the CUDA backend *enables* **native int8/fp8**
> (same sm_101 chip). Result:
> **1.9× faster**, lower memory with FP8, and **no OOM up to 15 s**. FP8 + `--fast fp8_matrix_mult`
> + an 8-step LoRA is both faster *and* higher quality than the int8 4-step baseline.

---

## 1. Hardware & baseline

| | |
|---|---|
| SoC | NVIDIA **Thor** (Blackwell datacenter/auto line), aarch64 |
| Compute capability | **sm_101** (Thor) — constant across CUDA versions; the CUDA toolkit version, not the arch, is the lever |
| Memory | **60 GB unified** (CPU+GPU shared); `comfy-aimdo` manages it |
| Stack | ComfyUI 0.34 · `comfy_kitchen` 0.2.31 · `comfy-aimdo` · torch 2.13.0+cu130 |
| Model | MiniMax H3 (`FLOW_AV`): AV latent = `NestedTensor(video[B,24,T,H/16,W/16], audio[B,32,2,T40])` |

All clips below are **768×432, 124 frames (5 s @ 24 fps), CFG 1.0, `shift_video=12`, euler/normal**,
unless noted. "sample" = KSampler stage; "total" = encode + sample + decode.

---

## 2. The core problem: CUDA 12.8 silently disables the fast path

`comfy_kitchen` gates its CUDA backend on the toolkit version:

```python
# comfy_kitchen backend selection (paraphrased)
if cuda_version < (13,):
    disable("cuda")     # <- on CUDA 12.8 the whole CUDA backend is OFF
```

So under **CUDA 12.8** every int8/fp8 op (`int8_linear`, `convrot_w4a4_linear`,
`scaled_mm_*`, `quantize_int8_*`, …) falls through to the **eager** backend, which
**dequantizes weights to bf16 on the fly** for every matmul. That is:

- **slow** — the int8 checkpoint buys you nothing at runtime;
- **memory-hungry in the wrong way** — and at 10 s the working set **OOMs**.

Measured baseline (CUDA 12.8, int8_convrot DiT, 4 steps):

```
encode 28 s · sample 620 s · decode 36 s  →  total 696 s   peak ~34.8 GB
10 s clip: OOM during sampling
```

---

## 3. The fix: CUDA 13 forward-compat → CUDA backend on → native int8/fp8

The car's kernel GPU driver is CUDA-12-era, but NVIDIA ships a **CUDA 13 forward-compat
`libcuda.so.1`** (+ JIT companions: `libnvidia-ptxjitcompiler`, `libnvidia-nvvm`, …). Load it
**first** on `LD_LIBRARY_PATH` while keeping the **system** `libnvrm_*` userspace from `/usr/lib`:

```bash
# /compat = CUDA-13 forward-compat libcuda + JIT companions (NO libnvrm here)
export LD_LIBRARY_PATH=/compat:$LD_LIBRARY_PATH:/host_lib:/host_thor_lib:/host_cuda_lib
#            ^^^^^^^ compat libcuda MUST be first; /host_lib keeps system libnvrm_*
# Do NOT copy the host libcuda over the compat one — that reverts you to CUDA 12.
```

Verification that CUDA 13 is live:

```
cuInit rc=0 · cuDriverGetVersion = 13020 (CUDA 13) · device = NVIDIA Thor (sm_101)
Found comfy_kitchen backend cuda: {'available': True, 'disabled': False, ...}
Native ops: mxfp8, convrot_w4a4, float8_e5m2, nvfp4, float8_e4m3fn, int8_tensorwise, asym_w4a8_int8
```

`Native ops: … int8_tensorwise … float8_e4m3fn` is the proof that **native int8/fp8 is active**
(no eager dequant). Two things make this work on a chip whose SASS isn't shipped:

- **PTX JIT**: the `comfy_kitchen` aarch64 wheel ships `sm_100`/`sm_120` cubins + PTX; the CUDA-13
  JIT compiler compiles PTX down to **sm_101** at load. (Empirically confirmed: kernels built for
  sm_100/sm_120 run correctly on sm_101.)
- **Attention**: PyTorch SDPA's **FLASH backend works on sm_101** (`sdpa_kernel([FLASH_ATTENTION])`
  succeeds), so ComfyUI's default `Using pytorch attention` already dispatches to flash. No
  `sageattention` needed — and it wouldn't help anyway (see §5.3).

---

## 4. Results — speed & memory by config

| Config | CUDA / arch | DiT | Steps | 5 s sample | 5 s total | Peak mem | 10 s | 15 s |
|---|---|---|---|---|---|---|---|---|
| **Baseline** | 12.8 / sm_101 | int8_convrot (34 GB) | 4 | 620 s | 696 s | 34.8 GB | **OOM** | **OOM** |
| **cu130 int8** | 13.0 / sm_101 | int8_convrot (34 GB) | 4 | **331 s** | **369 s** | 55.6 GB | 656 s ✓ | 1088 s + 78 s ✓ |
| **cu130 fp8** ⭐ | 13.0 / sm_101 | fp8_scaled (19.5 GB) | 8 | **281 s** | ~391 s | **48.9 GB** | ✓ | ✓ |

⭐ = current production recipe (`--fast fp8_matrix_mult`, 8-step turbo LoRA, audio on).

Reading the table:

- **CUDA 13 alone = 1.9×** on the *same* int8 model (sample 620 → 331 s), and it removes the OOM.
- **FP8 beats int8 even at 2× the steps**: 8-step FP8 samples in **281 s**, faster than 4-step
  int8 (331 s). Causes: smaller weights (19.5 vs 34 GB → faster load + less streaming), and
  `--fast fp8_matrix_mult` (native `float8_e4m3fn` matmul).
- **Memory is a managed ceiling, not a leak.** The cu130 peaks look *higher* (55.6 GB) because
  `comfy-aimdo` deliberately fills unified memory to ~56 GB (60 GB − ~3.7 GB headroom) for speed.
  It is a **cap** — aimdo offloads under pressure instead of OOMing, which is exactly why 10 s/15 s
  now succeed. FP8's smaller resident set gives a lower, calmer peak (48.9 GB).
- **Long clips are super-linear** (15 s sample 1088 s) because aimdo offload/reload thrashes the
  disk; putting weights on **internal NVMe (1.6 GB/s) instead of USB SSD (439 MB/s)** — a 3.6×
  read gap — is the single biggest long-clip lever.

---

## 5. Pitfalls & gotchas

### 5.1 CUDA / architecture
- **`comfy_kitchen` version floor.** 0.2.13 imports fine but ComfyUI crashes with
  `cannot import name 'TensorCoreConvRotW4A4Layout'` / `no attribute 'int8_attention_is_available'`.
  Use **0.2.31** (the first aarch64 cu130 wheel with the full cap set).
- **compat libcuda ordering.** If the host `libcuda.so.1` wins on `LD_LIBRARY_PATH`, you silently
  run CUDA 12 again (backend disabled, eager fallback). Keep `/compat` first; never `cp` the host
  libcuda into the image.
- **No sm_101 cubin is fine.** Don't try to rebuild `comfy_kitchen` for sm_101 — the sm_100/120
  PTX JITs correctly. Rebuilding wastes hours.
- **`docker commit` bakes `ENTRYPOINT=["bash"]`** (inherited from a `--entrypoint bash` run). A
  systemd `ExecStart` of `… image bash /script.sh` then runs `bash bash /script.sh` →
  `cannot execute binary file`. Fix: `--entrypoint bash <image> /script.sh` (drop the redundant `bash`).
- **The image is environment-only.** `docker commit` does **not** capture bind-mounted volumes —
  the ComfyUI *source* lives in a `/workspace` mount, not the image. Ship the code separately (or
  bake it) or the recipient gets an empty `/workspace`.

### 5.2 Model configs (quantization & steps)
- **int8_convrot is huge (34 GB)** and lives on the slow USB on this unit; **fp8_scaled (19.5 GB)**
  and **nvfp4-pruned (11.7 GB)** are far lighter. FP8 is the sweet spot (fast + good quality);
  nvfp4 is "pruned" → visibly lower quality, use only when you need the smallest/fastest.
- **Turbo LoRAs are step-count-specific.** Running the 4-step turbo at 8 steps *degrades* output.
  For 8-step quality you need the **8-step-trained LoRA** (`turbo_8step_v1.0`), not more steps on
  the 4-step one.
- **`--fast fp8_matrix_mult`** is what makes FP8 fast; without it FP8 is not obviously better than
  int8. Do **not** add `--lowvram/--novram` (they fight `comfy-aimdo` and hurt throughput).
- **Memory-budget heuristics keyed on filename substrings** (`"pruned"`, `"int8"`) misclassify the
  FP8 file (`…pruned_fp8_scaled…` contains `"pruned"` → treated as the tiny nvfp4 budget → a
  768×432×124 job gets rejected). Match on `"fp8"` explicitly.

### 5.3 Attention
- **Sage Attention is a dead end on Thor.** The `sageattention` wheel only ships
  `_qattn_sm80/_sm89/_sm90` — **no sm_100/sm_101** — so it can't load. `comfy_kitchen` has an
  `int8_attention` symbol but it isn't registered for sm_101 in 0.2.31.
- **You already have flash.** `torch.backends.cuda.flash_sdp_enabled()` is True and the FLASH SDPA
  backend runs on sm_101, so the default backend is already the fast path. Don't chase Sage.

### 5.4 Audio (the AV branch)
- **The old "audio = machine freeze" red line was a CUDA-12.8 problem.** On cu128 the AV sampling
  branch froze the unit 5/5. On **cu130 the AV latent samples and decodes cleanly** (verified 5/10/15 s
  + repeated MCP runs). The freeze was the eager audio-attention path, not the model.
- **Loading the audio VAE: use a plain `VAELoader`, NOT `LTXVAudioVAELoader`.** The H3 audio VAE
  has bare keys (`decoder.*`, `pre_block.*`, …); `LTXVAudioVAELoader` runs
  `state_dict_prefix_replace(..., filter_keys=True)` keeping only `audio_vae.`/`vocoder.` prefixes →
  the state dict becomes **empty** → `VAE is invalid: None`. A plain `VAELoader` lets
  `comfy.sd.VAE` auto-detect it via `if "pre_block.attn.zero_k_bias" in sd:` → `MiniMaxH3AudioVAE`.
- **Decoding audio from the AV latent**: `LTXVAudioVAEDecode` takes the *whole* latent and does
  `audio_latent = samples["samples"]; if is_nested: audio_latent = audio_latent.unbind()[-1]`
  (the last element is the audio). Then mux with **`CreateVideo(images, fps, audio)` + `SaveVideo`**
  — `SaveWEBM` has **no audio input**. Output: `h264` video + `aac 32000 Hz stereo`.

### 5.5 Ops / packaging
- **aarch64 venv with no pip**: install wheels with `python3 -m zipfile -e wheel.whl <purelib>`
  (the venv has no `pip`/`ensurepip`; `pip install` hits PEP668 then `No module named pip`).
- **scp-only transfer**: no `rsync` on the target → no resume; wrap large copies in a retry loop
  and verify byte size / sha256 (a 19.5 GB scp dropped connection once at 1.1 GB).
- **Disk math**: the 34 GB int8 DiT + 19.5 GB FP8 + 14.6 GB TE don't fit a nearly-full internal
  disk; deleting unused base images (`nvcr.io/nvidia/pytorch`, the old cu128 image) freed ~24 GB.

---

## 6. Reproducing it (the appliance)

The whole thing ships as a self-contained folder: a `docker load`-able image (env only), the
ComfyUI source, the CUDA-13 compat libs, and a launcher. Models are downloaded separately from
HuggingFace `Comfy-Org/MiniMax-H3`.

```bash
./setup.sh     # docker load + extract compat & code + check models
./run.sh       # ComfyUI on :8189
```

`run.sh` mounts, in order that matters: `/compat` (CUDA-13 libcuda) → `/host_lib` (system
`libnvrm_*`) → `/workspace/comfyui` (source) → `/workspace/models/h3` (weights) → GPU device
nodes (`/dev/nvgpu`, `/dev/nvmap`, …), with `--privileged --network host --ipc=host`.

The launcher:

```bash
export LD_LIBRARY_PATH=/compat:$LD_LIBRARY_PATH:/host_lib:/host_thor_lib:/host_cuda_lib
cd /workspace/comfyui
exec python3 main.py --listen 0.0.0.0 --port 8189 --enable-cors-header \
  --disable-xformers --disable-smart-memory --fast fp8_matrix_mult \
  --extra-model-paths-config /workspace/app/h3_model_paths.yaml
```

Default recipe: FP8 DiT + `turbo_8step_v1.0`, `steps=8`, `cfg=1.0`,
`MiniMaxH3SigmaShift(shift_video=12.0, shift_audio=3.0)`, euler/normal.

---

## 7. Numbers appendix

| Metric | Value |
|---|---|
| Internal disk read | 1.6 GB/s |
| USB SSD read | 439 MB/s (3.6× slower) |
| cu128 → cu130 sample speedup (int8, 5 s) | 620 → 331 s (**1.87×**) |
| cu128 → cu130 total speedup (int8, 5 s) | 696 → 369 s (**1.89×**) |
| FP8 8-step sample (5 s) | 281 s (faster than int8 4-step) |
| Peak mem, int8 cu130 | 55.6 GB (aimdo cap) |
| Peak mem, FP8 cu130 | 48.9 GB |
| Max clip without OOM | 15 s (362 frames) on cu130; 5 s on cu128 |
| aimdo headroom | ~3.7 GB below the 60 GB pool |

---

*Field-tested on a Zeekr Thor-U. The CUDA-13 forward-compat trick and the `comfy_kitchen`
version/arch notes are the load-bearing parts; everything else is tuning.*
