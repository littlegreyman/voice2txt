# voice2txt — 纯本地离线语音转写 CLI

把会议录音 / 手机录音机录音转为 **SRT 字幕** 或 **纯文本**，**完全离线运行**（模型仅安装时一次性下载，无需任何 API secret / 令牌）。本手册覆盖 **Windows 10/11、Ubuntu 26.04、macOS（Apple Silicon 为主）** 三个平台。

## 功能概览

| 功能 | 说明 |
|---|---|
| 离线转写 | 运行时强制离线，断网也能用 |
| 双引擎 | FunASR 全家桶（默认，中文最强、带标点）/ faster-whisper large-v3（备选，多语种兜底） |
| 说话人分离 | 默认开启，可选 `[SPEAKER_00]` 标签；`--no-diarize` 关闭提速 |
| 中英混合 | 保留原文混合，不翻译改写 |
| 输出格式 | SRT 字幕 / TXT 文本（UTF-8 with BOM，Windows 记事本不乱码） |
| 会议纪要 | TXT 输出可加 `--brief`，本地 LLM（默认 Qwen2.5-7B GGUF）自动整理为结构化 `.brief.md` 纪要 |
| 长音频 | 按 `batch_size_s` 自动分块（设计支持长录音，建议先用样本实测） |
| 显存自适应 | 爆显存自动降级：fp16 → int8_float16 → int8 → CPU |
| 多平台 | Windows / Ubuntu / macOS 一致运行（GPU：CUDA / MPS；无 GPU 走 CPU） |

- 方案 B（默认引擎）：**FunASR 全家桶**（SenseVoice/Paraformer + CAM++）— 中文最强、零 token、单 Pipeline 出「说话人+时间戳+文本」。
- 方案 A（备选引擎）：**faster-whisper + CAM++ 分离** — 多语种/强混码质量兜底，用 `--engine whisper` 切换。
- 已针对 **RTX 4060 8GB** 调优；**设计支持长音频**（按 `batch_size_s` 分块，长录音建议先用你的样本实测后再依赖）。

## 通用使用流程（三平台一致）

```bash
# ① 安装依赖（需联网，仅一次）      —— 见下方各平台「安装」
# ② 下载模型（需联网，仅一次）
voice2txt setup

# ③ 转写（之后断网也能用）
voice2txt transcribe 会议录音.mp3 -f srt -o 会议录音.srt
```

> 只要完成过一次 `voice2txt setup`，第 ③ 步在任何时候、断网状态下都能运行。

## 安装（按平台）

### Windows 版

1. 安装 **Python 3.10 ~ 3.12**（python.org 下载，安装时勾选 “Add python.exe to PATH”）。
2. 打开 **PowerShell / CMD**，建立虚拟环境并安装：
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   # 要用 GPU 加速，先从 PyTorch 官方源装 CUDA 版 torch（版本按你的 CUDA 调整）：
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
   # 再装本项目
   pip install -e ".[dev]"
   ```
   > 不装 CUDA 版 torch 也能用，只是转写走 CPU，速度较慢。
3. 继续「下载模型」。

**离线 / 便携（高级）**：若目标机无法联网，可用官方 embeddable Python + 仓库根目录的
`get-pip.py` 自建独立运行时，再把模型缓存（`~\.cache\...` 或 `models_dir`）一并拷贝。
常规用户无需此步。

### Ubuntu 26.04 版

1. 确认 Python 版本为 **3.10 ~ 3.12**（26.04 系统默认可能是 3.13，需用 deadsnakes / pyenv 装 3.12）：
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt install python3.12 python3.12-venv
   python3.12 -m venv .venv && . .venv/bin/activate
   ```
2. 有 NVIDIA GPU 时，先装好**驱动 + CUDA 12.x 运行时**，再装 CUDA 版 torch：
   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -e ".[dev]"
   ```
   无 GPU 的机器直接 `pip install -e ".[dev]"`（CPU 版）。
3. 继续「下载模型」。

### macOS 版（Apple Silicon）

1. 安装 **Python 3.12**（官方 pkg / `brew install python@3.12` / `pyenv install 3.12.x`）。
   不要用系统自带的 `/usr/bin/python3`。
2. 安装依赖（Apple Silicon 自动走 **MPS** 加速，无 CUDA）：
   ```bash
   python3.12 -m venv .venv && . .venv/bin/activate
   pip install -e ".[dev]"
   ```
   > macOS 上 faster-whisper 因底层 CTranslate2 无 Metal 后端，走 **CPU** 推理（大文件较慢）；
   > 中文场景建议用默认的 FunASR 引擎（走 MPS）。
3. 继续「下载模型」。

## 下载模型（仅一次，需联网）

```bash
voice2txt setup                 # 下载全部引擎模型（FunASR + whisper）
voice2txt setup --engine funasr # 只下 FunASR（默认引擎）
voice2txt setup --engine whisper# 只下 whisper 备选引擎
voice2txt setup --engine brief  # 只下纪要总结模型 Qwen2.5-7B-Instruct GGUF（可选，--brief 才需要）
```

- 国内网络若 HuggingFace 主站不可达，会自动走 `hf-mirror.com` 镜像并关闭 Xet，无需手动设环境变量。
- 模型缓存位置：`~/.cache/modelscope`、`~/.cache/huggingface`（各平台一致）。
- 模型已**钉死到可验证版本**，防止上游权重被静默篡改（见「模型完整性与供应链安全」）。
- 纪要模型（brief）默认落到 `~/.cache/voice2txt/brief/`；可用 `models_dir` 一并带走，彻底离线可移植。

## 转写（核心用法）

### 基本命令

```bash
# 默认引擎 FunASR，输出 SRT
voice2txt transcribe meeting.mp3 -f srt -o meeting.srt

# 输出纯文本，并关闭说话人分离（更快）
voice2txt transcribe meeting.mp3 --no-diarize -f txt -o meeting.txt

# 切换 faster-whisper 备选引擎（多语种兜底）
voice2txt transcribe interview.m4a -e whisper -f txt -o interview.txt

# 指定语种（FunASR 支持；whisper 用 None 自动检测）
voice2txt transcribe call.mp3 -l zh -f srt -o call.srt
```

### 参数详解

| 参数 | 说明 | 默认值 |
|---|---|---|
| `input` | 音频文件路径（mp3/wav/m4a/flac/...） | 必填 |
| `-e, --engine` | `funasr`（默认）/ `whisper` | `funasr` |
| `-f, --format` | `srt` / `txt` | `srt` |
| `-l, --language` | `auto`/`zh`/`en`/`ja`/`ko`/`yue`/... | `auto` |
| `--no-diarize` | 关闭说话人分离（提速） | 开 |
| `--no-speaker-labels` | 输出不带 `[SPEAKER_xx]` 前缀 | 带 |
| `--vram` | `auto`/`aggressive`/`balanced`/`conservative` 显存策略 | `auto` |
| `-o, --output` | 输出文件路径；不填则打印到终端 | 终端 |
| `--brief` | 仅 `-f txt` 生效：额外生成结构化会议纪要 `[同名].brief.md` | 关 |
| `--asr-model` | 覆盖 ASR 模型 id | 引擎默认 |
| `--beam-size` | SenseVoice 解码 beam；调大(5~10)提升词准确率，更慢 | `1` |
| `--hotword` | 上下文偏置热词（空格分隔），提升专有名词/人名准确率 | — |
| `--brief-model` | 纪要 LLM 规模：`7b`（默认，质量高）/ `3b`（轻量更快） | `7b` |
| `--offline` | 强制离线（现已是默认行为，保留用于显式声明） | — |

> `-o` 输出文件带 **UTF-8 BOM**，Windows 记事本 / 播放器直接打开中文不乱码。

### 输出格式

- **SRT**：保留逐句时间戳，适合字幕软件。例：
  ```
  1
  00:00:00,000 --> 00:00:02,500
  [SPEAKER_00] 大家好，欢迎参加今天的会议。

  2
  00:00:02,500 --> 00:00:05,000
  [SPEAKER_01] 谢谢，我们开始吧。
  ```
- **TXT**：合并连续同一说话人，输出更连贯的段落（带/不带时间戳可选）。

### 说话人分离

默认开启。关闭可显著提速：`--no-diarize`。若已知人数，可在 `config.yaml` 设 `max_speakers`
提升稳定性（可选）。

### 长音频

按 `batch_size_s`（默认 300 秒）自动分块处理，设计上支持长录音。
极长文件（如 1~2 小时）建议先用自己的样本实测一次耗时与显存。

### 提升转写准确率（精度优先）

默认配置偏向速度。若更看重词准确率、能接受更慢，可用以下杠杆（可叠加）：

- **加大解码 beam**：`--beam-size 8`（5~10 均可）。SenseVoice 默认 1，越大词错率越低、越慢。
  ```bash
  voice2txt transcribe 会议.mp3 -f txt --beam-size 8 -o 会议.txt
  ```
- **上下文偏置热词**：`--hotword "张三 项目A 第二季度"`。把会议里的专有名词 / 人名喂给模型做偏置，
  能显著降低这类词的错误（尤其生僻人名、产品代号）。
- **指定语种**：中文会议加 `-l zh`、粤语 `-l yue`，避免 `auto` 误检语种导致质量下降。
- **高质量音源**：程序已统一重采样到 16k 单声道（PyAV），无需手动处理；尽量提供清晰录音。
- **换更强模型（高级）**：纯中文场景可试 `iic/paraformer-zh`（Paraformer-large，准确率常高于 SenseVoice，
  但更慢、无内置 ITN）；英文 / 强混码用 `--engine whisper`（large-v3）。经 `--asr-model` / `--engine` 切换。

> 经验：中文会议最划算的两步是「`-l zh` + `--hotword '关键人名 术语'`」，几乎零成本显著提准确率；
> 若仍不满意再上 `--beam-size 8`。

### 中英混合

默认保留原文混合（不翻译）。FunASR 引擎自带标点恢复（ct-punc），中文标点更自然；
whisper 备选引擎默认不带中文标点，属引擎特性差异。

### 会议纪要（`--brief`，仅 TXT）

转写为 **TXT** 时加 `--brief`，会在本地用 **Qwen2.5-7B-Instruct GGUF（Q4_K_M，默认）** 通过
`llama-cpp-python` 离线把转录整理为结构化中文会议纪要，并写入 **`[同名].brief.md`** 文件
（与 `-o` 输出同 stem，UTF-8 无 BOM）。整条链路无联网、无 API key。

```bash
# 转写为 TXT 同时生成会议纪要 meeting.brief.md
voice2txt transcribe meeting.mp3 --no-diarize -f txt -o meeting.txt --brief

# 不指定 -o 时，.brief.md 以输入文件名 stem 落到当前目录：rec.m4a -> rec.brief.md
voice2txt transcribe rec.m4a -f txt --brief
```

- 纪要模型需先一次性下载（仅一次、需联网）：`voice2txt setup --engine brief`。
- 默认纪要模型已升级为 **7B（Q4_K_M，约 4.7GB）**，质量优于 3B；显存吃紧或只想要快速草稿时，
  可在 `config.yaml` 设 `brief_model: 3b` 或加 `--brief-model 3b` 退回 **3B（约 2GB）**。
  7B 在 RTX 4060 上可走 GPU 卸载（`llama-cpp-python`
  若安装的是带 CUDA 的构建）；纯 CPU 构建也能跑，只是稍慢。
  - 安装 `llama-cpp-python`（纪要功能的可选依赖）：**PyPI 上 `llama-cpp-python` 没有 Windows
    预编译 wheel**（只有源码 sdist），直接 `pip install` 会从源码编译，需本机有 C/C++ 工具链
    （如 Visual Studio Build Tools），否则失败。推荐装社区在 GitHub Releases 发布的预编译 wheel：

    **重要更正（0.3.36 实测）**：CUDA 变体不在 `v0.3.36` 主发布里，而是各自独立的发布标签
    `v0.3.36-cu118 / cu121 / cu122 / cu123 / cu124 / cu125 / cu130`（**没有 cu126**）。
    wheel 文件名统一为 `llama_cpp_python-0.3.36-py3-none-win_amd64.whl`，CUDA 版本体现在**发布标签**，
    不是 `+cuXX` 文件名后缀（旧文档写的 `*cp311*cu126*win_amd64*.whl` 是错误的）。本机 RTX 4060
    驱动支持到 CUDA 13.1、已装 torch cu126（提供 `cudart64_12.dll`），故选 **`v0.3.36-cu125`**
    （CUDA 12.5 构建，向后兼容 12.6 运行时与 13.x 驱动）；驱动较新也可试 `cu130`。

    把 `$py` 换成你的 Python 解释器路径（例如项目自带的独立 Python 3.11 解释器
    `C:\path\to\Python311\python.exe`）：

    ```powershell
    $py = "你的-python.exe"   # 例如上面那个独立 Python 3.11
    # 0.3.36 无 cu126 变体；RTX 4060 选 cu125（CUDA 12.5，兼容 12.6/13.x）
    $tag = "v0.3.36-cu125"
    # 直连 github 文件难时，用 ghproxy 镜像代理下载（哪个通换哪个）：
    #   https://ghproxy.com/  https://ghproxy.net/  https://gh.api.99988866.xyz/
    $url = "https://ghproxy.com/https://github.com/abetlen/llama-cpp-python/releases/download/$tag/llama_cpp_python-0.3.36-py3-none-win_amd64.whl"
    Write-Host "下载并安装: $url"
    & $py -m pip install --force-reinstall --no-deps $url
    & $py -m pip install -e ".[brief]"
    ```

    - 本项目从本地源码安装，命令是 `pip install -e ".[brief]"`，**不是** `pip install "voice2txt[brief]"`
      （voice2txt 未发布到 PyPI）。
    - 验证 GPU 是否生效：装好后跑一次 `voice2txt ... --brief`，看 `[info] device=` 是否为 `cuda`；
      或 `& $py -c "import llama_cpp, os; print(os.path.dirname(llama_cpp.__file__))"` 确认装的是 CUDA 版。
    - **CUDA 运行库**：cu125 wheel 运行时会找 `cudart64_12.dll`。已装 torch cu126（自带该 DLL）通常可直接用；
      若加载报缺 `cudart64_12.dll`，装一份 NVIDIA CUDA 12.x Toolkit 并加入 PATH 即可。
    - 本机 `api.github.com` 常被网络/代理拦截，不要再依赖「GitHub API 自动发现 wheel」的旧做法
      （会拿到空 URL，导致 404 或 `You must give at least one requirement`）。上面用 ghproxy 直链
      拉取发布文件，绕开 API，稳定得多。
- `--brief` 仅对 `-f txt` 生效；与 `-f srt` 同时指定会报错。
- 超长转录会自动按 32K 上下文分块、各自摘要后再合并，不会溢出上下文窗口。
- 纪要生成失败**不影响**主产物 TXT（仅打印警告）；未下载纪要模型时也会跳过并提示。

## 离线说明

`transcribe` **运行时默认强制离线**：模型已在 `setup` 阶段下载到本地缓存，运行时不发起任何
联网 / 下载，因此断网也能用。若运行时提示找不到模型，请先在联网环境执行一次
`voice2txt setup`。

## 纯离线 / 便携（高级）

若目标机无法联网，或你希望把整套环境（含 Python 运行时）随 U 盘 / 内网机带走，可用官方
embeddable Python + `get-pip.py` 自建独立运行时，再把模型缓存（`~/.cache/...` 或
`models_dir` 指定目录）一并拷贝。仓库根目录的 `get-pip.py` 仅用于该离线方案，属安装产物，
**不随源码仓库分发**。常规用户用上面的 `venv` 方式即可。

也可编辑 `config.yaml` 设置 `models_dir` 把模型缓存指向任意目录，便于 U 盘 / 内网机分发：
```yaml
models_dir: /path/to/my_models   # 模型随程序一起带走，彻底离线可移植
```

## 架构

```
音频 -> io/audio(16k单声道wav) -> Engine.transcribe()
        B: FunASR(VAD+ASR+PUNC+CAM++)  -> Utterance[]
        A: faster-whisper + CAM++ -> align -> Utterance[]
     -> io/formatter -> TXT / SRT
```

统一内部表示 `Utterance{start,end,text,speaker}`，两种引擎产出一致，格式化层共用。

## 显存策略（RTX 4060 8GB）

自动降级链：`fp16 → int8_float16 → int8 → CPU`。遇 CUDA OOM 自动降一档并重试。
- `aggressive`：从 fp16 起手，质量最高、最吃显存。
- `balanced`：从 int8_float16 起手（默认 `auto` 等同此项，8GB 稳妥）。
- `conservative`：从 int8 起手，最省显存。

## 模型完整性与供应链安全

模型权重来自公开 hub，默认在 `voice2txt setup` 时一次性下载并缓存到本地，之后完全离线。
为防止上游权重被静默篡改，所有模型已**钉死到可验证版本**（commit / 分支），见
`voice2txt/models/registry.py` 的 `MODEL_PINS`：

- `Systran/faster-whisper-large-v3` → commit `edaa852ec7e145841d8ffdb056a99866b5f0a478`
- `iic/SenseVoiceSmall` → commit `7bf452403abd7353a300cd760f7adae7701c92c1`
- 辅助模型（fsmn-vad / ct-punc / cam++ / paraformer-zh）暂钉 `master` 分支
- `Qwen/Qwen2.5-3B-Instruct-GGUF`（纪要轻量模型，`qwen2.5-3b-instruct-q4_k_m.gguf`）→ commit `7dabda4d13d513e3e842b20f0d435c732f172cbe`
- `Qwen/Qwen2.5-7B-Instruct-GGUF`（纪要比默认模型，7B Q4_K_M 约 4.7GB；上游以 2 个 GGUF 分片发布：`qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` + `...-00002-of-00002.gguf`）→ 钉 commit `bb5d59e06d9551d752d08b292a50eb208b07ab1f`（loader 指向首个分片，llama-cpp 自动读取同目录兄弟分片）

whisper 下载额外通过 `allow_patterns` 仅拉取推理必需文件。`MODEL_PINS` 的哈希改版时需人工
复核与上游一致再更新。

## 服务端 / 桌面端（预留）

- `voice2txt serve --host 127.0.0.1 --port 8000`：启动内网转录服务（当前为预留骨架，
  默认仅本机 `127.0.0.1`，内网开放需显式设 `--host 0.0.0.0` 并自管防火墙）。
- 桌面 GUI 亦在路线中（复用同一引擎层）。两者尚未实现，不影响 CLI 使用。

## 常见问题（FAQ）

**Q1：运行时提示找不到模型 / 要联网？**
A：模型未下载。联网执行一次 `voice2txt setup` 即可，之后离线可用。

**Q2：GPU 没加速 / 很慢？**
A：确认装的是 CUDA 源 torch（Windows/Linux）或 MPS（macOS Apple Silicon）。
默认 `pip install torch` 是 CPU 版。可用 `python -c "import torch; print(torch.cuda.is_available())"` 自查。

**Q3：Python 3.13 装不上依赖？**
A：本项目要求 **3.10 ~ 3.12**（FunASR 依赖链无 3.13 wheel）。请换 3.12。

**Q4：输出中文乱码？**
A：用 `-o` 写文件已带 BOM；若直接看终端乱码，是终端编码问题，文件本身正常。

**Q5：whisper 转写没有中文标点？**
A：whisper large-v3 默认不带中文标点，属引擎特性；中文场景用默认 FunASR 引擎。

**Q6：转换中爆显存（CUDA OOM）？**
A：程序会自动降级（fp16→int8_float16→int8→CPU）并重试；也可手动 `--vram conservative`。

**Q7：加了 `--brief` 没生成 `.brief.md` 文件？**
A：两种常见原因：① `--brief` 只能配合 `-f txt`，与 `-f srt` 一起会报错；② 纪要模型未下载，
先 `voice2txt setup --engine brief` 联网下载一次（默认 7B 约 4.7GB；`--brief-model 3b` 约 2GB）。两者都会给出明确提示而非静默跳过。

**Q8：`--brief` 纪要用的是 CPU / 生成很慢？**
A：纪要后端是 llama-cpp-python，只有装了 **CUDA 版** wheel 才会走 GPU。PyPI/默认装到的是 CPU 版，
故 `--brief` 走 CPU、较慢（长转录可能几分钟）。要让 RTX 4060 加速：按上文「安装 llama-cpp-python」
装 `v0.3.36-cu125`（或 cu130）的预编译 CUDA wheel（用 ghproxy 拉），重装后 `[info] device=` 会显示
`cuda`，速度可快数十倍。**注意：CPU 只影响速度，不影响纪要质量。**

**Q9：`--brief` 生成的纪要比言无物 / 很空洞？**
A：空洞主因**不是 CPU**（CPU 只拖慢速度）。本版默认纪要模型已是 **7B Q4_K_M**（质量优于 3B），并优化了
提示词（强制行动项填全「责任人/时限」、信息密度要高）、放宽 `max_tokens`(1536→2048)、降低
`frequency_penalty`(0.3→0.2)。若仍不够，想要更快的草稿用 `--brief-model 3b`；想要更强则换更大模型
（如 Qwen2.5-14B，需更多显存）或继续打磨提示词。

## 命令速查

| 目的 | 命令 |
|---|---|
| 下载全部模型 | `voice2txt setup` |
| 仅下 FunASR | `voice2txt setup --engine funasr` |
| 转写→SRT | `voice2txt transcribe 文件.mp3 -f srt -o 出.srt` |
| 转写→TXT（不分离） | `voice2txt transcribe 文件.mp3 --no-diarize -f txt -o 出.txt` |
| 用 whisper 引擎 | `voice2txt transcribe 文件.m4a -e whisper -f txt -o 出.txt` |
| 指定中文 | `voice2txt transcribe 文件.mp3 -l zh -f srt -o 出.srt` |
| 转写并出纪要 | `voice2txt transcribe 文件.mp3 -f txt -o 出.txt --brief` |
| 下载纪要模型 | `voice2txt setup --engine brief` |
| 查看版本 | `voice2txt version` |

## 路线图

- P0 脚手架/模型下载 ✅ → P1 引擎B → P2 长音频+SRT → P3 显存自适应
- P4 引擎A 可切换 → P5 中英混合质量 → P6 单文件离线分发 → P7 服务端/桌面端预留 → P8 文档合规
- `serve`（内网 FastAPI 服务）与 GUI 已在代码层预留，复用同一引擎层，便于「单机 → 内网服务器」「CLI → GUI」演进。

## 许可与合规

- 代码本体：**Apache-2.0**（见 `LICENSE`）。
- 第三方模型与库见 `NOTICE`（FunASR 系列、faster-whisper、Whisper 权重、PyAV、PyTorch、Qwen2.5-3B/7B、llama-cpp-python 等）。
- 分发时请保留 `LICENSE` 与 `NOTICE` 及上游署名；若修改源码请按 Apache-2.0 第 4(b) 条标明。
