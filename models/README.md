# Models — place these 5 files here before running

Download from HuggingFace repo **Comfy-Org/MiniMax-H3** (mirror: https://hf-mirror.com).
Put each file in the matching subfolder below. Verify size and sha256 (see INSTALL.md).

```
models/
├── diffusion_models/
│   └── minimax_h3_fl2va_pruned_fp8_scaled.safetensors        (~19.5 GB)  FP8 DiT
├── loras/
│   └── minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors (~1.8 GB) 8-step turbo LoRA
├── vae/
│   ├── minimax_h3_video_vae_fp16.safetensors                 (~4.9 GB)  video VAE
│   └── minimax_h3_audio_vae_fp32.safetensors                 (~0.6 GB)  audio VAE
└── text_encoders/
    └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors          (~14.6 GB) text encoder
```

Exact download commands are in INSTALL.md (section "Models").
