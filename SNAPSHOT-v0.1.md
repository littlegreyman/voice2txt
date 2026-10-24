# voice2txt 快照 — v0.1（2026-10-24）

本文件记录 voice2txt 达到「基本可用」时的**代码、环境配置与依赖状态**，用于未来一旦出错可回滚到本版本。
代码快照 = git 标签 `v0.1`；依赖快照 = `requirements.lock.txt`；本文件 = 恢复手册。

> 路径均为示意：`<独立 Python 3.11 目录>` / `<项目目录>` / `<独立 Python 3.11 解释器路径>` 请按你本机实际位置替换（本文不写死任何个人路径）。

---

## 1. 运行时与关键依赖（精确版本）

| 项目 | 值 |
|---|---|
| Python | 3.11.9（独立嵌入包，非系统 Python） |
| 解释器路径 | 项目自带的独立 Python 3.11（embeddable 包，路径因机而异） |
| torch / torchaudio | 2.14.1+cu126 / 2.11.0+cu126（CUDA 12.6 构建） |
| funasr | 1.4.16 |
| faster-whisper | 1.2.1（备选引擎，需先 `setup --engine whisper`） |
| ctranslate2 | 4.8.3 |
| av (PyAV) | 18.1.0 |
| llama-cpp-python | 0.3.36（**CUDA cu125** 预编译 wheel，非 PyPI 默认 CPU 版） |
| modelscope / huggingface_hub | 1.40.2 / 2.3.0 |
| numpy | 2.4.6 |

> 完整 103 个第三方包见 `requirements.lock.txt`（由 `python -m pip freeze` 生成）。

## 2. 环境要点（易踩坑，恢复时务必注意）

### 2.1 torch 必须用 CUDA 索引安装
`torch==2.14.1+cu126` 的 `+cu126` 构建**只存在于 PyTorch CUDA 索引**，纯 PyPI 只有 CPU 版。
恢复依赖时必须带 `--extra-index-url https://download.pytorch.org/whl/cu126`，否则装到 CPU 版 → GPU 不可用。

### 2.2 llama-cpp-python 是 CUDA 预编译 wheel
PyPI 上的 llama-cpp-python 0.3.36 **没有 win_amd64 wheel**（仅源码 sdist，需编译器）。
本环境装的是 GitHub Releases 的 `v0.3.36-cu125` CUDA 变体（标签在 `cu125`，不在文件名）。
`requirements.lock.txt` 已把该依赖改为可复现的 ghproxy 直链；镜像不通时换 `ghproxy.com` / `gh.api.99988866.xyz`。

### 2.3 CUDA 运行时 DLL（关键！）
CUDA 版 `llama.dll` 依赖 `cudart64_12.dll` 等 CUDA 运行时，而这些只随 torch(cu126) 装在 `torch/lib/`，
**不在 llama-cpp-python 的搜索路径**。本环境已处理：
- 一次性复制 `cudart64_12.dll` / `cublas64_12.dll` / `cublasLt64_12.dll` / `cusparse64_12.dll`
  从 `torch/lib/` → `llama_cpp/lib/`（位于 site-packages 内）；
- 代码兜底：`voice2txt/core/summarizer.py` 的 `_ensure_cuda_runtime_on_path()` 在加载 llama_cpp 前
  best-effort 把 `torch/lib` 加入 `PATH`，**重装 llama-cpp-python 后仍能自动找到依赖**（无 torch/CPU 版静默跳过）。
> 若回滚后 `import llama_cpp` 仍报「找不到模块或其依赖」，手动执行上面的 DLL 复制即可。

### 2.4 离线优先
`transcribe` 运行时强制 `MODELSCOPE_OFFLINE=1`（断网可用）。FunASR 从本地 snapshot 目录加载（见 §3），
不再每次联网。whisper 备选引擎走本地 `whisper-large-v3` 目录。

## 3. 模型权重缓存位置（不在 git 内，需单独保留）

| 模型 | 路径 | 说明 |
|---|---|---|
| FunASR（SenseVoiceSmall / fsmn-vad / ct-punc / cam++） | `~/.cache/modelscope/models/iic--*/snapshots/<rev>/` | 主引擎，默认离线加载 |
| whisper-large-v3（备选） | `~/.cache/huggingface/whisper-large-v3` | 平铺 local_dir（绕过 blobs→snapshots 链接） |
| 纪要 LLM（Qwen2.5-3B-Instruct GGUF Q4_K_M） | `~/.cache/voice2txt/brief/qwen2.5-3b-instruct-q4_k_m.gguf` | `--brief` 用，约 2.1GB |

> 模型版本已钉 commit（`voice2txt/models/registry.py` 的 `MODEL_PINS`）：
> SenseVoiceSmall `7bf452403abd7353a300cd760f7adae7701c92c1`、qwen2.5-3b-brief `7dabda4d13d513e3e842b20f0d435c732f172cbe`、whisper `edaa852ec7e145841d8ffdb056a99866b5f0a478`。
> 若缓存丢失：联网跑一次 `voice2txt setup` 即可重新下载（国内网络自动走镜像、关 Xet）。

## 4. v0.1 已实现功能

- `transcribe`：SRT / TXT 输出；说话人分离（默认开，`--no-diarize` 关）；中英混合保留；
  多语种 `auto/zh/en/ja/ko/yue`；离线优先。
- 双引擎：FunASR（默认，中文最强、带标点）/ faster-whisper（备选，多语种）。
- `--brief`：本地 LLM（Qwen2.5-3B GGUF）把 TXT 整理为结构化 Markdown 会议纪要，写入 **`[同名].brief.md`**；
  RTX 4060 走 CUDA（`n_gpu_layers=-1` 全量卸载）。
- 爆显存自动降级（fp16→int8_float16→int8→CPU）。
- 详细 `--help`（含典型场景示例）：`voice2txt -h` / `voice2txt transcribe -h` / `voice2txt setup -h`。

## 5. 回滚方法

### 5.1 代码回滚（git）
```bash
cd <voice2txt 仓库目录>
git stash                 # 先暂存当前未提交改动（如有）
git checkout v0.1         # 切到 v0.1 标签（ detach HEAD ）；或 git checkout -b rollback-v0.1 v0.1
```
> 当前工作树在 v0.1 提交之上如无未提交改动，可直接 `git checkout v0.1`。

### 5.2 依赖回滚（精确版本）
```bash
PY="<独立 Python 3.11 解释器路径>"
"$PY" -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.lock.txt
"$PY" -m pip install -e ".[brief]" --no-deps   # 重新以可编辑方式安装 voice2txt 自身
```
> 若 `import llama_cpp` 报找不到依赖，按 §2.3 复制 CUDA 运行时 DLL。

### 5.3 整环境重建（极端情况：独立 Python 目录丢失）
1. 重新解压官方 embeddable Python 3.11 到 `<独立 Python 3.11 目录>` 并装 pip；
2. 按 §5.2 安装依赖（torch 走 CUDA 索引）；
3. 保留 `~/.cache/` 下的模型权重（或重新 `voice2txt setup` 下载）；
4. `git checkout v0.1` 取源码。

## 6. 不在快照范围内（明确排除）

- 模型权重（GB 级，位于 `~/.cache/`，不进 git）；
- 独立 Python 运行时目录（含 site-packages），不进 git，靠 §5.2 锁文件重建；
- 用户个人录音/转录产物（`1023meeting.*`、`out.srt`、`whisper_out.txt` 等已被 `.gitignore` 排除）。
