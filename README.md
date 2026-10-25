<div align="center">

# MiniMax H3 on NVIDIA Thor · CUDA 13
# 在 NVIDIA Thor 上跑 MiniMax H3 · CUDA 13

**Self-contained ComfyUI appliance + field guide** · **自包含 ComfyUI 一体机 + 技术复盘**

Runs **MiniMax H3** text/image → **video + audio** on an NVIDIA Thor (sm_101) with native int8/fp8.
在 NVIDIA Thor（sm_101）上以原生 int8/fp8 跑 **MiniMax H3** 文/图 → **视频 + 音频**。

</div>

---

## TL;DR · 一句话

| 🇨🇳 中文 | 🇬🇧 English |
|---|---|
| 芯片是 **NVIDIA Thor（sm_101）**。CUDA 12.8 下 `comfy_kitchen` 的版本门（`<13`）把 CUDA 后端**硬关闭**，int8 退化成 eager 反量化：慢（5 秒片段采样 ≈ 620 秒）且 **10 秒 OOM**。改用 **CUDA 13 前向兼容**（同一颗 sm_101）通过版本门 → **启用原生 int8/fp8**：**快 1.9 倍**、FP8 显存更低、**15 秒不 OOM**。FP8 + `--fast fp8_matrix_mult` + 8 步 LoRA 比 int8 4 步**又快又好**。 | The chip is **NVIDIA Thor (sm_101)**. Under CUDA 12.8 `comfy_kitchen`'s version gate (`<13`) **hard-disables** its CUDA backend, so int8 falls back to eager dequant: slow (5 s clip ≈ 620 s sampling) and **OOM at 10 s**. Booting under **CUDA 13 forward-compat** (same sm_101 chip) passes the gate → **native int8/fp8**: **1.9× faster**, lower FP8 memory, **no OOM to 15 s**. FP8 + `--fast fp8_matrix_mult` + an 8-step LoRA beats the int8 4-step baseline on **both speed and quality**. |

## 性能对比 · Benchmark

*768×432 · 124 帧/5s · CFG 1.0 · euler/normal*

| 配置 Config | CUDA / 架构 arch | DiT | 步 Steps | 5s 采样 sample | 峰值 peak | 10s | 15s |
|---|---|---|---|---|---|---|---|
| 基线 Baseline | 12.8 / sm_101 | int8_convrot 34 GB | 4 | 620 s | 34.8 GB | ❌ OOM | ❌ OOM |
| cu130 int8 | 13.0 / sm_101 | int8_convrot 34 GB | 4 | **331 s** | 55.6 GB | ✅ | ✅ |
| ⭐ cu130 fp8 | 13.0 / sm_101 | fp8_scaled 19.5 GB | 8 | **281 s** | **48.9 GB** | ✅ | ✅ |

⭐ = 生产配方 production recipe（FP8 + 8-step LoRA + audio）。

> 🔬 **稀疏注意力 · Sparse attention**：生产配方之上再叠加速度的那一层，**2026-10-22 起换成 Veda**（`VedaSparseAttention` 节点，学习式 tile 预测）——固定 topk 的 **SpargeAttn**（int8-QK + FP8-PV，同 seed 768×432×5s：**271.8s → 151.3s ≈1.8×**）在运动多的片段上会画**横向条纹**，现保留为可切换回退（开关文件 `sparge_topk` 现为 `0` = 关；`--use-sage-attention` + shim + 预编译内核的开启方式不变）。稠密量化 SageAttention 只有 ~1.1×（flash 已很快），**块稀疏是杠杆——学习式的更干净**。详见 [GUIDE §5.3](GUIDE.md) / [§5.6 预算](GUIDE.md) / [INSTALL §8](INSTALL.md)。
> 🔬 **Sparse attention**: the accelerator stacked on the production recipe is **Veda** (learned tile prediction, `VedaSparseAttention` node) **since 2026-10-22** — fixed-topk **SpargeAttn** (same-seed 768×432×5s: 271.8s → 151.3s, ≈1.8×) drew **horizontal streaks** on motion-heavy clips and is kept only as a toggle-off fallback (`sparge_topk` = `0`; enable path unchanged: `--use-sage-attention` + shim + prebuilt kernel). Dense quantized SageAttention is only ~1.1× (flash is already fast) — **block-sparsity is the lever; the learned kind is the clean one.** See [GUIDE §5.3](GUIDE.md) / [§5.6 budgets](GUIDE.md) / [INSTALL §8](INSTALL.md).

## 快速开始 · Quick start

```bash
# 1) 下载 5 个模型到 models/（清单见 INSTALL.md §3）· download the 5 models into models/
# 2) 安装：加载镜像 + 解压 compat/源码（软件零下载）· install (no software download)
./setup.sh
# 3) 启动 ComfyUI :8189 · start ComfyUI on :8189
./run.sh
```

## 包内容 · Contents

| 路径 Path | 说明 | Description |
|---|---|---|
| `image/comfy-h3-cu130-v1.tar.gz` | Docker 镜像（torch 2.13+cu130、comfy_kitchen、依赖） | docker image (env only) |
| `code/h3_cu130_code.tar.gz` | ComfyUI 0.34 源码（含 H3 + 音频节点） | ComfyUI source (H3 + audio nodes) |
| `compat/compat_cu130.tar.gz` | CUDA-13 前向兼容 libcuda + JIT | CUDA-13 forward-compat libs |
| `app/` | 模型路径配置 + 启动脚本 | model paths + launcher |
| `setup.sh` / `run.sh` | 一键安装 / 启动 | one-shot install / launch |
| `sample_workflow.json` | FP8+8步+音频 测试图 | FP8+8step+audio test graph |
| `models/` | **空**，需自下 5 个模型 | **empty**; download 5 models |

> 📦 大二进制（镜像/源码/compat tar）现在全部放在本仓库的 **GitHub Releases**（tag `appliance-v1`），不再走网盘。
> 镜像超过 GitHub 单资产 2 GB 上限，拆成 `.part00`~`.part04` 五个资产，`cat` 合并后用 `SHASUMS-256.txt` 校验，详见 [INSTALL §1.1](INSTALL.md)。
> 📦 All large binaries (docker image / ComfyUI source / CUDA-13 compat) now ship as **GitHub Release assets**
> (tag `appliance-v1`) — no cloud drive. The image is split in five parts (2 GB per-asset cap): re-join with
> `cat`, verify with `SHASUMS-256.txt`. See [INSTALL §1.1](INSTALL.md).

## 文档 · Documentation

| 🇨🇳 中文 | 🇬🇧 English |
|---|---|
| **[GUIDE.zh.md](GUIDE.zh.md)** — 完整技术复盘：cu 版本、各模型配置速度/显存、踩坑清单 | **[GUIDE.md](GUIDE.md)** — full field guide: CUDA versions, per-config speed/memory, pitfalls |
| [INSTALL.md](INSTALL.md) — 给 AI 的自动安装（含模型清单 + sha256） | [INSTALL.md](INSTALL.md) — AI-driven install (model list + sha256) |

## 需要的模型 · Required models

来自 HuggingFace `Comfy-Org/MiniMax-H3` · from HuggingFace `Comfy-Org/MiniMax-H3`：
FP8 DiT (19.5 GB) · 8-step turbo LoRA (1.8 GB) · video VAE (4.9 GB) · audio VAE (0.6 GB) · text encoder (14.6 GB)。
精确路径与 sha256 见 [INSTALL.md §3](INSTALL.md)。/ Exact paths & sha256 in [INSTALL.md §3](INSTALL.md).
