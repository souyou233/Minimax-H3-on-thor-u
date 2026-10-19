#!/bin/bash
# One-shot setup for MiniMax H3 ComfyUI (cu130 / Thor sm_110). Run from the package root.
# Loads the docker image, extracts compat libs + ComfyUI code, checks models are present.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"

echo "[1/4] docker load image (comfy-h3-cu130:v1) ..."
docker load -i image/comfy-h3-cu130-v1.tar.gz

echo "[2/4] extract CUDA-13 forward-compat libs ..."
rm -rf compat_libs && mkdir -p compat_libs && tar xzf compat/compat_cu130.tar.gz -C compat_libs

echo "[3/4] extract ComfyUI code ..."
rm -rf comfyui && tar xzf code/h3_cu130_code.tar.gz

echo "[4/4] check models ..."
miss=0
for m in \
  diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
  loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors ; do
  [ -f "models/$m" ] || { echo "  MISSING: models/$m"; miss=1; }
done
if [ "$miss" = 1 ]; then
  echo "  !! Download the 5 models into models/<subdir>/ first (see INSTALL.md, section 'Models')."
else
  echo "  all 5 models present."
fi
echo "Setup complete. Start the server with:  ./run.sh"
