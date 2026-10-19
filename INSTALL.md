# MiniMax H3 Video Generation — cu130 Appliance (NVIDIA Thor / sm_110)

Self-contained ComfyUI appliance that runs **MiniMax H3** text/image→video+audio with a
**cu130 (CUDA 13.0)** PyTorch stack and **native int8/fp8** kernels (comfy_kitchen CUDA
backend). On Thor this runs an FP8 DiT + 8-step turbo LoRA and decodes **both video and
audio** from the AV latent into an MP4.

> **Audience: an AI agent.** Execute the steps below in order. All software (docker image,
> ComfyUI source, CUDA-13 compat libs) is **included in this folder — no downloads needed
> for software.** The only external downloads are the **5 model weight files** (Section 3),
> which are large and must be fetched from HuggingFace.

---

## 1. Package contents

```
comfy-h3-cu130/
├── INSTALL.md              <- you are here
├── setup.sh                <- one-shot: load image + extract compat & code
├── run.sh                  <- start ComfyUI on :8189
├── SHA256SUMS              <- checksums of the 3 archives below
├── sample_workflow.json    <- a ready-to-run FP8+8step+audio test graph
├── image/comfy-h3-cu130-v1.tar.gz   (~7.9 GB)  docker image (env: torch 2.13+cu130, comfy_kitchen, deps)
├── compat/compat_cu130.tar.gz       (~0.2 GB)  CUDA-13 forward-compat libcuda + JIT companions
├── code/h3_cu130_code.tar.gz        (~0.6 GB)  ComfyUI 0.34 source (incl. MiniMax H3 + audio nodes)
├── app/
│   ├── h3_model_paths.yaml          model search paths (points at /workspace/models/h3/*)
│   └── start_comfy_cu130.sh         container launcher (port 8189, --fast fp8_matrix_mult)
└── models/               <- EMPTY by design; you download the 5 weights here (Section 3)
    ├── diffusion_models/  ├── loras/  ├── vae/  └── text_encoders/
```

## 2. Prerequisites (check before installing)

- **Hardware:** NVIDIA **Thor (compute capability sm_110)**, aarch64/ARM64. (Verified on the
  Zeekr Thor-U. Jetson Thor dev kits with the same driver stack should work.)
- **OS / driver:** aarch64 Linux with the NVIDIA GPU userspace (`libnvrm_*` present under
  `/usr/lib`) and a kernel driver new enough for **CUDA 13 forward-compatibility** (the
  bundled `compat/` supplies the CUDA-13 `libcuda.so.1`; it needs a host driver ≥ the CUDA-13
  forward-compat floor — true on the Zeekr Thor-U image).
- **Docker** with `--privileged` and access to GPU device nodes:
  `/dev/nvgpu`, `/dev/nvgpu-pci`, `/dev/nvmap`, `/dev/tegradisp`.
- **Disk:** ≥ 60 GB free (image ~15 GB unpacked + ~41 GB models + outputs).
- **RAM:** ≥ 48 GB unified memory recommended (FP8 run peaks ~49 GB; 5s@768x432).

Quick check:
```bash
nvidia-smi -L 2>/dev/null || ls /dev/nvgpu            # GPU present
docker --version && echo "docker ok"
ls /usr/lib/libnvrm* 2>/dev/null | head -1            # GPU userspace present
```

## 3. Models — download these 5 (the ONLY external downloads)

Repo: **`Comfy-Org/MiniMax-H3`**. Use `https://hf-mirror.com` (fast in CN) or
`https://huggingface.co`. Place each file in the shown path under `models/`.

| # | Target path (under `models/`) | Size | sha256 |
|---|-------------------------------|------|--------|
| 1 | `diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | ~19.5 GB | `12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993` |
| 2 | `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | ~1.8 GB | `2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e` |
| 3 | `vae/minimax_h3_video_vae_fp16.safetensors` | ~4.9 GB | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| 4 | `vae/minimax_h3_audio_vae_fp32.safetensors` | ~0.6 GB | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| 5 | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | ~14.6 GB | `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6` |

Download (run from the package root; `BASE` = mirror or hub):
```bash
BASE=https://hf-mirror.com/Comfy-Org/MiniMax-H3/resolve/main   # or https://huggingface.co/...
curl -L -C - -o models/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
  "$BASE/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
curl -L -C - -o models/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
  "$BASE/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
curl -L -C - -o models/vae/minimax_h3_video_vae_fp16.safetensors \
  "$BASE/vae/minimax_h3_video_vae_fp16.safetensors"
curl -L -C - -o models/vae/minimax_h3_audio_vae_fp32.safetensors \
  "$BASE/vae/minimax_h3_audio_vae_fp32.safetensors"
curl -L -C - -o models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  "$BASE/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
```
Verify (must all say `OK`):
```bash
cd models && sha256sum -c - <<'EOF'
12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993  diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e  loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522  vae/minimax_h3_video_vae_fp16.safetensors
8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48  vae/minimax_h3_audio_vae_fp32.safetensors
35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
EOF
cd ..
```

## 4. Install (software — all local, no download)

```bash
cd comfy-h3-cu130
sha256sum -c SHA256SUMS        # optional integrity check of the 3 archives
./setup.sh                     # docker load + extract compat & code + check models
```
`setup.sh` creates `./comfyui/` (code) and `./compat_libs/compat/` (CUDA-13 libs) and reports
whether the 5 models are present.

## 5. Run

```bash
./run.sh
```
Watches: the log should print
```
Found comfy_kitchen backend cuda: {'available': True, 'disabled': False, ...}
Native ops: mxfp8, convrot_w4a4, float8_e5m2, nvfp4, float8_e4m3fn, int8_tensorwise, asym_w4a8_int8
To see the GUI go to: http://0.0.0.0:8189
```
`Native ops: ... int8_tensorwise ... float8_e4m3fn` = **native int8/fp8 is active** (no eager
dequant fallback). Server is at **http://<host-ip>:8189**.

## 6. Verify end-to-end (generate a test video with audio)

With the server running (leave `run.sh` in a terminal, or add `-d`), from another shell:
```bash
# submit the bundled test graph
curl -s -X POST http://127.0.0.1:8189/prompt \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\": $(cat sample_workflow.json), \"client_id\": \"smoke\"}"
# poll history for the prompt_id returned above, then fetch the mp4 from /view
```
Expected: an MP4 with **two streams** — `Video: h264 ... 448x256 24fps` and
`Audio: aac ... 32000 Hz stereo`. Confirm with:
```bash
ffmpeg -i <file>.mp4 2>&1 | grep -E 'Stream|Duration'
```

## 7. How it works (for the integrating agent)

- **Image** = runtime only (torch 2.13.0+cu130, comfy_kitchen 0.2.31, all deps). ComfyUI
  **source is mounted** from `./comfyui` at `/workspace/comfyui`; models from `./models` at
  `/workspace/models/h3`; config from `./app` at `/workspace/app`.
- **cu130 activation:** `app/start_comfy_cu130.sh` puts `/compat` (CUDA-13 `libcuda.so.1`)
  **first** on `LD_LIBRARY_PATH`, keeping the system `libnvrm_*` from `/host_lib`. Under CUDA 13
  the GPU reports **sm_110** and comfy_kitchen's CUDA backend enables native int8/fp8.
- **FP8 speed:** launched with `--fast fp8_matrix_mult`. Do **not** add `--lowvram/--novram`.
- **Audio:** the KSampler produces an AV latent (video+audio). `LTXVAudioVAEDecode` takes the
  whole latent and `unbind()[-1]` extracts the audio; the audio VAE is loaded with a **plain
  `VAELoader`** (auto-detected as `MiniMaxH3AudioVAE`), then `CreateVideo(images,fps,audio)` +
  `SaveVideo` mux to MP4. (Do **not** use `LTXVAudioVAELoader` — its prefix filter empties the
  H3 audio VAE.)
- **Default recipe:** FP8 DiT + `turbo_8step_v1.0` LoRA, `steps=8`, `cfg=1.0`,
  `MiniMaxH3SigmaShift shift_video=12.0 shift_audio=3.0`, euler/normal.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `cannot execute binary file` on start | `run.sh` must pass `--entrypoint bash ... /workspace/app/start_comfy_cu130.sh` (the image's baked entrypoint is `bash`). |
| `cuInit failed` / no device | GPU device nodes not mounted, or host driver too old for CUDA-13 forward-compat. Confirm `/dev/nvgpu` + `libnvrm_*`. |
| `Native ops` line missing / eager fallback | `/compat` not first on `LD_LIBRARY_PATH`, or not under CUDA 13. Check the start script. |
| `VAE is invalid: None` on audio node | You used `LTXVAudioVAELoader`; switch to plain `VAELoader` for the audio VAE. |
| OOM on long/high-res | Lower `length` (frames) or resolution; the FP8 run peaks ~49 GB at 768x432x124. |
| Model not found in node dropdown | File missing or wrong subdir under `models/`; re-check Section 3 paths. |

## 9. Reference

- Models: HuggingFace `Comfy-Org/MiniMax-H3`.
- Measured on Zeekr Thor-U: 5s@768x432 sample ~281 s, peak ~49 GB; 10s/15s run without OOM.
- Ports: ComfyUI `8189`. Output dir: `./comfyui/output` (inside the mounted code tree).
