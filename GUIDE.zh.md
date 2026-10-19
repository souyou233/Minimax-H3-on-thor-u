# 在 NVIDIA Thor（sm_101）上用 CUDA 13 跑 MiniMax H3 —— 实战复盘

> 本页是 [GUIDE.md](GUIDE.md) 的中文版。

在 **NVIDIA Thor** SoC（极氪 Thor-U 车机）上跑 **MiniMax H3** 音视频扩散模型（文/图 → 视频+音频）
的完整生产复盘：从 CUDA 12.8 迁移到 CUDA 13 从而解锁原生 int8/fp8 内核、各模型配置的速度/显存
实测、以及一路踩过的坑。

> **一句话结论** —— 芯片是 **NVIDIA Thor（sm_101）**。在 CUDA 12.8 下 `comfy_kitchen` 的
> `cuda_version < (13,)` 版本门把 CUDA 后端**硬关闭**，int8 权重退化成 eager 反量化：慢（5 秒片段
> 采样 ≈ 620 秒）且 **10 秒就 OOM**。把同一套栈放到 **CUDA 13 前向兼容**下运行，通过该版本门 →
> CUDA 后端**启用原生 int8/fp8**（同一颗 sm_101）。
> 结果：**快 1.9 倍**、FP8 显存更低、**15 秒都不 OOM**。FP8 + `--fast fp8_matrix_mult` + 8 步 LoRA
> 比 int8 4 步基线**又快又好**。

---

## 1. 硬件与基线

| | |
|---|---|
| SoC | NVIDIA **Thor**（Blackwell 数据中心/车载线），aarch64 |
| 计算能力 | **sm_101**（Thor）—— 跨 CUDA 版本不变；变量是 CUDA 工具链版本，不是架构 |
| 内存 | **60 GB 统一内存**（CPU/GPU 共享）；由 `comfy-aimdo` 管理 |
| 软件栈 | ComfyUI 0.34 · `comfy_kitchen` 0.2.31 · `comfy-aimdo` · torch 2.13.0+cu130 |
| 模型 | MiniMax H3（`FLOW_AV`）：AV latent = `NestedTensor(视频[B,24,T,H/16,W/16], 音频[B,32,2,T40])` |

除特别说明，下文所有片段均为 **768×432、124 帧（5 秒 @ 24fps）、CFG 1.0、`shift_video=12`、euler/normal**。
"sample" = KSampler 阶段；"total" = 编码 + 采样 + 解码。

---

## 2. 核心问题：CUDA 12.8 悄悄关掉了快速路径

`comfy_kitchen` 用 CUDA 工具链版本给它的 CUDA 后端设了门槛：

```python
# comfy_kitchen 后端选择（大意）
if cuda_version < (13,):
    disable("cuda")     # <- CUDA 12.8 下整个 CUDA 后端被关掉
```

所以在 **CUDA 12.8** 下，每个 int8/fp8 算子（`int8_linear`、`convrot_w4a4_linear`、`scaled_mm_*`、
`quantize_int8_*` …）都落到 **eager** 后端，后者**每次矩阵乘都把权重临时反量化成 bf16**。这意味着：

- **慢** —— int8 检查点在运行时一点便宜都占不到；
- **以错误的方式吃内存** —— 到 10 秒时工作集直接 **OOM**。

实测基线（CUDA 12.8，int8_convrot DiT，4 步）：

```
编码 28s · 采样 620s · 解码 36s  →  合计 696s   峰值 ~34.8 GB
10 秒片段：采样阶段 OOM
```

---

## 3. 解法：CUDA 13 前向兼容 → CUDA 后端开启 → 原生 int8/fp8

车机的内核 GPU 驱动是 CUDA-12 时代的，但 NVIDIA 提供了一份 **CUDA 13 前向兼容的 `libcuda.so.1`**
（外加 JIT 伴生库：`libnvidia-ptxjitcompiler`、`libnvidia-nvvm` 等）。把它放到 `LD_LIBRARY_PATH`
**最前面**，同时保留 `/usr/lib` 里的**系统** `libnvrm_*` 用户态：

```bash
# /compat = CUDA-13 前向兼容 libcuda + JIT 伴生库（这里没有 libnvrm）
export LD_LIBRARY_PATH=/compat:$LD_LIBRARY_PATH:/host_lib:/host_thor_lib:/host_cuda_lib
#            ^^^^^^^ compat 的 libcuda 必须在最前；/host_lib 提供系统 libnvrm_*
# 千万不要把宿主的 libcuda 覆盖到 compat 上——那会让你退回 CUDA 12
```

确认 CUDA 13 生效：

```
cuInit rc=0 · cuDriverGetVersion = 13020（CUDA 13）· 设备 = NVIDIA Thor（sm_101）
Found comfy_kitchen backend cuda: {'available': True, 'disabled': False, ...}
Native ops: mxfp8, convrot_w4a4, float8_e5m2, nvfp4, float8_e4m3fn, int8_tensorwise, asym_w4a8_int8
```

`Native ops: … int8_tensorwise … float8_e4m3fn` 就是**原生 int8/fp8 已生效**的铁证（没有 eager 反量化）。
两颗"这颗芯片根本没发布对应 SASS"也能跑起来的定心丸：

- **PTX JIT**：`comfy_kitchen` 的 aarch64 wheel 带的是 `sm_100`/`sm_120` cubin + PTX；CUDA-13 的 JIT
  编译器在加载时把 PTX 编到 **sm_101**。（实测确认：为 sm_100/sm_120 编的内核在 sm_101 上正确运行。）
- **注意力**：PyTorch SDPA 的 **FLASH 后端在 sm_101 上可用**（`sdpa_kernel([FLASH_ATTENTION])` 成功），
  所以 ComfyUI 默认的 `Using pytorch attention` 其实已经走 flash。不需要 `sageattention`——它也没用
  （见 §5.3）。

---

## 4. 结果 —— 各配置的速度与显存

| 配置 | CUDA / 架构 | DiT | 步数 | 5s 采样 | 5s 合计 | 峰值显存 | 10s | 15s |
|---|---|---|---|---|---|---|---|---|
| **基线** | 12.8 / sm_101 | int8_convrot (34 GB) | 4 | 620s | 696s | 34.8 GB | **OOM** | **OOM** |
| **cu130 int8** | 13.0 / sm_101 | int8_convrot (34 GB) | 4 | **331s** | **369s** | 55.6 GB | 656s ✓ | 1088s + 78s ✓ |
| **cu130 fp8** ⭐ | 13.0 / sm_101 | fp8_scaled (19.5 GB) | 8 | **281s** | ~391s | **48.9 GB** | ✓ | ✓ |

⭐ = 当前生产配方（`--fast fp8_matrix_mult`、8 步 turbo LoRA、音频开）。

怎么读这张表：

- **光换 CUDA 13 就是 1.9 倍**（同一个 int8 模型，采样 620 → 331 秒），而且消除了 OOM。
- **FP8 即便步数翻倍也赢过 int8**：8 步 FP8 采样只要 **281 秒**，比 4 步 int8（331 秒）还快。原因：
  权重更小（19.5 vs 34 GB → 加载更快、流式更少），加上 `--fast fp8_matrix_mult`（原生 `float8_e4m3fn` 矩阵乘）。
- **内存是被管理的天花板，不是泄漏。** cu130 峰值看着*更高*（55.6 GB）是因为 `comfy-aimdo` 为了速度
  故意把统一内存填到 ~56 GB（60 GB − ~3.7 GB 余量）。它是个**上限**——aimdo 在压力下换出而不是 OOM，
  这正是 10s/15s 现在能跑通的原因。FP8 常驻集更小，峰值更低更稳（48.9 GB）。
- **长片段是超线性的**（15s 采样 1088 秒），因为 aimdo 换入换出会反复读写磁盘；把权重从 **USB SSD
  （439 MB/s）搬到内置 NVMe（1.6 GB/s）**——3.6 倍的读速差——是长片段最大的单点提速杠杆。

---

## 5. 踩坑清单

### 5.1 CUDA / 架构
- **`comfy_kitchen` 版本下限。** 0.2.13 能 import 但 ComfyUI 会崩在
  `cannot import name 'TensorCoreConvRotW4A4Layout'` / `no attribute 'int8_attention_is_available'`。
  用 **0.2.31**（第一个带全套能力位的 aarch64 cu130 wheel）。
- **compat libcuda 的顺序。** 如果宿主的 `libcuda.so.1` 在 `LD_LIBRARY_PATH` 上赢了，你会**悄悄**又跑回
  CUDA 12（后端关闭、eager 回退）。`/compat` 必须在最前；绝不要把宿主 libcuda `cp` 进镜像。
- **没有 sm_101 cubin 没关系。** 别去为 sm_101 重编 `comfy_kitchen`——sm_100/120 的 PTX 会正确 JIT。
  重编纯属浪费几小时。
- **`docker commit` 会把 `ENTRYPOINT=["bash"]` 烤进镜像**（继承自某次 `--entrypoint bash` 运行）。于是
  systemd 的 `ExecStart` 写成 `… 镜像 bash /script.sh` 就变成 `bash bash /script.sh` →
  `cannot execute binary file`。修法：`--entrypoint bash <镜像> /script.sh`（去掉多余的 `bash`）。
- **镜像只是运行环境。** `docker commit` **不会**打包 bind-mount 的卷——ComfyUI 的**源码**在 `/workspace`
  挂载里，不在镜像里。要么单独分发源码，要么烤进去，否则对方拿到的是空的 `/workspace`。

### 5.2 模型配置（量化与步数）
- **int8_convrot 很大（34 GB）**，且在本机放在慢速 USB 上；**fp8_scaled（19.5 GB）** 和
  **nvfp4-pruned（11.7 GB）** 轻得多。FP8 是甜点（又快又好）；nvfp4 是"剪枝"版 → 质量肉眼可见下降，
  只在要最小/最快时用。
- **turbo LoRA 是按步数训练的。** 把 4 步 turbo 跑到 8 步会**变差**。要 8 步质量必须用**按 8 步训练的
  LoRA**（`turbo_8step_v1.0`），而不是给 4 步的多加步数。
- **`--fast fp8_matrix_mult`** 才是 FP8 快的关键；没有它 FP8 相比 int8 并不明显更好。**不要**加
  `--lowvram/--novram`（它们和 `comfy-aimdo` 打架、拖慢吞吐）。
- **按文件名子串做的显存预算启发式**（`"pruned"`、`"int8"`）会**误判** FP8 文件
  （`…pruned_fp8_scaled…` 里含 `"pruned"` → 被当成最小的 nvfp4 预算 → 768×432×124 的任务被拒）。
  要显式匹配 `"fp8"`。

### 5.3 注意力
- **Sage Attention 在 Thor 上是死路。** `sageattention` wheel 只有 `_qattn_sm80/_sm89/_sm90`——
  **没有 sm_100/sm_101**——根本加载不了。`comfy_kitchen` 有个 `int8_attention` 符号，但 0.2.31 里没给
  sm_101 注册。
- **你其实已经有 flash 了。** `torch.backends.cuda.flash_sdp_enabled()` 为 True，FLASH SDPA 后端在
  sm_101 上能跑，所以默认后端已经是快速路径。别再追 Sage 了。

### 5.4 音频（AV 分支）
- **老的"音频 = 整机冻死"红线是 CUDA-12.8 的问题。** cu128 上 AV 采样分支 5/5 冻机。在 **cu130 上
  AV latent 采样与解码都干净**（5/10/15 秒 + 反复 MCP 跑通）。冻机的是 eager 音频注意力路径，不是模型。
- **加载音频 VAE：用普通 `VAELoader`，不要用 `LTXVAudioVAELoader`。** H3 音频 VAE 是裸键
  （`decoder.*`、`pre_block.*` …）；`LTXVAudioVAELoader` 会跑
  `state_dict_prefix_replace(..., filter_keys=True)` 只保留 `audio_vae.`/`vocoder.` 前缀 → state dict
  变**空** → `VAE is invalid: None`。普通 `VAELoader` 让 `comfy.sd.VAE` 通过
  `if "pre_block.attn.zero_k_bias" in sd:` 自动识别为 `MiniMaxH3AudioVAE`。
- **从 AV latent 解出音频**：`LTXVAudioVAEDecode` 吃**整个** latent，做
  `audio_latent = samples["samples"]; if is_nested: audio_latent = audio_latent.unbind()[-1]`
  （最后一个元素是音频）。然后用 **`CreateVideo(images, fps, audio)` + `SaveVideo`** 合成——
  `SaveWEBM` **没有音频输入**。输出：`h264` 视频 + `aac 32000 Hz` 立体声。

### 5.5 运维 / 打包
- **没有 pip 的 aarch64 venv**：用 `python3 -m zipfile -e wheel.whl <purelib>` 装 wheel
  （该 venv 没有 `pip`/`ensurepip`；`pip install` 先撞 PEP668 再 `No module named pip`）。
- **只能 scp**：目标机没有 `rsync` → 不能续传；大文件拷贝要包在重试循环里并校验字节数/sha256
  （一次 19.5 GB 的 scp 在 1.1 GB 处断过连接）。
- **磁盘账**：34 GB int8 DiT + 19.5 GB FP8 + 14.6 GB TE 塞不进快满的内置盘；删掉没用的基础镜像
  （`nvcr.io/nvidia/pytorch`、旧的 cu128 镜像）腾出约 24 GB。

---

## 6. 复现（一体机）

整套东西以自包含文件夹交付：一个可 `docker load` 的镜像（仅环境）、ComfyUI 源码、CUDA-13 compat
库、以及启动脚本。模型从 HuggingFace `Comfy-Org/MiniMax-H3` 单独下载。

```bash
./setup.sh     # docker load + 解压 compat 与源码 + 检查模型
./run.sh       # ComfyUI 起在 :8189
```

`run.sh` 按**有讲究的顺序**挂载：`/compat`（CUDA-13 libcuda）→ `/host_lib`（系统 `libnvrm_*`）→
`/workspace/comfyui`（源码）→ `/workspace/models/h3`（权重）→ GPU 设备节点（`/dev/nvgpu`、`/dev/nvmap` …），
并带 `--privileged --network host --ipc=host`。

启动脚本：

```bash
export LD_LIBRARY_PATH=/compat:$LD_LIBRARY_PATH:/host_lib:/host_thor_lib:/host_cuda_lib
cd /workspace/comfyui
exec python3 main.py --listen 0.0.0.0 --port 8189 --enable-cors-header \
  --disable-xformers --disable-smart-memory --fast fp8_matrix_mult \
  --extra-model-paths-config /workspace/app/h3_model_paths.yaml
```

默认配方：FP8 DiT + `turbo_8step_v1.0`，`steps=8`，`cfg=1.0`，
`MiniMaxH3SigmaShift(shift_video=12.0, shift_audio=3.0)`，euler/normal。

---

## 7. 数据附录

| 指标 | 数值 |
|---|---|
| 内置盘读速 | 1.6 GB/s |
| USB SSD 读速 | 439 MB/s（慢 3.6 倍） |
| cu128 → cu130 采样提速（int8，5s） | 620 → 331 秒（**1.87×**） |
| cu128 → cu130 合计提速（int8，5s） | 696 → 369 秒（**1.89×**） |
| FP8 8 步采样（5s） | 281 秒（比 int8 4 步还快） |
| 峰值显存，int8 cu130 | 55.6 GB（aimdo 封顶） |
| 峰值显存，FP8 cu130 | 48.9 GB |
| 不 OOM 的最长片段 | cu130 可到 15 秒（362 帧）；cu128 只能 5 秒 |
| aimdo 余量 | 距 60 GB 池约 3.7 GB |

---

*在极氪 Thor-U 上实测。CUDA-13 前向兼容这一招、以及 `comfy_kitchen` 的版本/架构说明是承重墙；
其余都是调优。*
