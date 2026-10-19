#!/bin/bash
# MiniMax H3 ComfyUI launcher for cu130 (CUDA 13 / Thor sm_110).
# /compat = CUDA-13 forward-compat libcuda + JIT companions (MUST be first on LD_LIBRARY_PATH).
# System GPU userspace (libnvrm_*) comes from /host_lib. Do NOT copy host libcuda (would override compat).
export LD_LIBRARY_PATH=/compat:$LD_LIBRARY_PATH:/host_lib:/host_thor_lib:/host_cuda_lib
cd /workspace/comfyui
exec python3 main.py --listen 0.0.0.0 --port 8189 --enable-cors-header \
  --disable-xformers --disable-smart-memory --fast fp8_matrix_mult \
  --extra-model-paths-config /workspace/app/h3_model_paths.yaml
