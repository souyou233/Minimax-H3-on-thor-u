#!/bin/bash
# Launch MiniMax H3 ComfyUI (cu130) on http://<host>:8189 . Run from the package root.
# Requires: NVIDIA Thor (sm_110) host with docker + GPU device nodes, and ./setup.sh already run.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -d "$HERE/comfyui" ] || { echo "Run ./setup.sh first."; exit 1; }
COMPAT="$HERE/compat_libs/compat"
[ -d "$COMPAT" ] || { echo "compat not extracted; run ./setup.sh"; exit 1; }

MOUNTS="-v $COMPAT:/compat:ro -v /usr/lib:/host_lib:ro"
# cu128 driver-side lib dirs (present on the Zeekr Thor-U image); skipped if absent.
[ -d /usr/local/cuda-12.8/thor/lib64 ] && MOUNTS="$MOUNTS -v /usr/local/cuda-12.8/thor/lib64:/host_thor_lib:ro"
[ -d /usr/local/cuda-12.8/lib64 ]      && MOUNTS="$MOUNTS -v /usr/local/cuda-12.8/lib64:/host_cuda_lib:ro"

exec docker run --rm --name comfy-h3-cu130 --privileged --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  $MOUNTS \
  -v "$HERE/comfyui":/workspace/comfyui \
  -v "$HERE/models":/workspace/models/h3 \
  -v "$HERE/app":/workspace/app:ro \
  -v /dev/nvgpu:/dev/nvgpu -v /dev/nvgpu-pci:/dev/nvgpu-pci \
  -v /dev/nvmap:/dev/nvmap -v /dev/tegradisp:/dev/tegradisp \
  --entrypoint bash comfy-h3-cu130:v1 /workspace/app/start_comfy_cu130.sh
