#!/bin/bash
# MiniMax H3 ComfyUI launcher for cu130 (CUDA 13 / Thor sm_110).
# /compat = CUDA-13 forward-compat libcuda + JIT companions (MUST be first on LD_LIBRARY_PATH).
# System GPU userspace (libnvrm_*) comes from /host_lib. Do NOT copy host libcuda (would override compat).
export LD_LIBRARY_PATH=/compat:$LD_LIBRARY_PATH:/host_lib:/host_thor_lib:/host_cuda_lib
cd /workspace/comfyui
# Optional ~1.8x block-sparse attention (SpargeAttn): add `--use-sage-attention` to the args below
# and drop app/sageattention.py (shim) + a locally built spas_sage_attn/ into /workspace/comfyui.
# Off by default in this template; our production deployment runs it on at topk=0.3. See INSTALL.
exec python3 main.py --listen 0.0.0.0 --port 8189 --enable-cors-header \
  --disable-xformers --disable-smart-memory --fast fp8_matrix_mult \
  --extra-model-paths-config /workspace/app/h3_model_paths.yaml
