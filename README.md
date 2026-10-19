# MiniMax H3 视频生成 · cu130 一体机包（NVIDIA Thor / sm_110）

一个自包含的 ComfyUI 一体机：在 **cu130（CUDA 13.0）** 上跑 **MiniMax H3** 文/图→视频+音频，
用 **FP8 DiT + 8步 turbo LoRA**，原生 int8/fp8 内核，输出**带音频的 MP4**。

> 详细安装见 **[INSTALL.md](INSTALL.md)**（给 AI 自动执行）；完整技术复盘、cu 版本与不同模型配置的速度/显存对比、踩坑清单见 **[GUIDE.zh.md](GUIDE.zh.md)**（中文）/ [GUIDE.md](GUIDE.md)（English）。本文件是给人看的速览。

## 包内容
| 路径 | 说明 |
|---|---|
| `image/comfy-h3-cu130-v1.tar.gz` | Docker 镜像（torch 2.13+cu130、comfy_kitchen、全部依赖） |
| `code/h3_cu130_code.tar.gz` | ComfyUI 0.34 源码（含 MiniMax H3 + 音频节点） |
| `compat/compat_cu130.tar.gz` | CUDA-13 前向兼容 libcuda + JIT |
| `app/` | 模型路径配置 + 容器启动脚本 |
| `setup.sh` / `run.sh` | 一键安装 / 启动 |
| `sample_workflow.json` | 现成的 FP8+8步+音频 测试工作流 |
| `models/` | **空**——需自行下载 5 个模型（见下） |
| `SHA256SUMS` | 3 个归档的校验和 |

## 快速开始（3 步）
```bash
# 0) 前置：NVIDIA Thor(sm_110) + docker + GPU 设备节点 + ≥60G 磁盘 + ≥48G 内存
# 1) 下载 5 个模型到 models/ 下（命令与 sha256 见 INSTALL.md §3）
# 2) 安装（软件零下载）
./setup.sh
# 3) 启动，浏览器开 http://<本机IP>:8189
./run.sh
```

## 需要的 5 个模型（源：HuggingFace `Comfy-Org/MiniMax-H3`）
| 放到 `models/` 下 | 大小 |
|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | ~19.5G |
| `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | ~1.8G |
| `vae/minimax_h3_video_vae_fp16.safetensors` | ~4.9G |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | ~0.6G |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | ~14.6G |

## 验证
启动日志出现 `Native ops: ... int8_tensorwise ... float8_e4m3fn` = 原生 int8/fp8 已生效；
`To see the GUI ... :8189` = 就绪。提交 `sample_workflow.json` 应产出**双流 MP4**
（h264 视频 + aac 32kHz 音频）。

## 关键提示
- 启动已带 `--fast fp8_matrix_mult`，**不要**加 `--lowvram/--novram`。
- 音频 VAE 用**普通 `VAELoader`** 加载（不是 `LTXVAudioVAELoader`）。
- 实测（Zeekr Thor-U）：5s@768×432 采样 ~281s、峰值 ~49G；10s/15s 不 OOM。
