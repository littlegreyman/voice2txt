"""模型注册表：默认模型、setup 预下载清单、合规 NOTICE 数据。"""
from __future__ import annotations

from typing import Dict, List

# 引擎 -> 默认 ASR 模型
DEFAULT_ASR: Dict[str, str] = {
    "funasr": "iic/SenseVoiceSmall",   # 多语种(中/粤/英/日/韩)非自回归
    "whisper": "large-v3",             # OpenAI Whisper (CTranslate2)
}


def default_asr_model(engine: str) -> str:
    return DEFAULT_ASR.get(engine, "large-v3")


# setup 预下载清单 + 合规信息（用于 NOTICE）
MODELS: List[Dict] = [
    {"id": "iic/SenseVoiceSmall", "engine": "funasr", "source": "ModelScope",
     "license": "模型许可见 ModelScope 模型卡（商用需署名）",
     "desc": "阿里达摩院多语种 ASR（中/粤/英/日/韩），含语种/情绪/事件标签"},
    {"id": "iic/paraformer-zh", "engine": "funasr", "source": "ModelScope",
     "license": "模型许可见 ModelScope 模型卡",
     "desc": "中文 Paraformer-large，远场/方言鲁棒"},
    {"id": "iic/cam++", "engine": "funasr", "source": "ModelScope",
     "license": "模型许可见 ModelScope 模型卡",
     "desc": "说话人嵌入 + 聚类（CAM++），无需 token"},
    {"id": "iic/fsmn-vad", "engine": "funasr", "source": "ModelScope",
     "license": "模型许可见 ModelScope 模型卡", "desc": "语音活动检测 VAD"},
    {"id": "iic/ct-punc", "engine": "funasr", "source": "ModelScope",
     "license": "模型许可见 ModelScope 模型卡", "desc": "标点恢复"},
    {"id": "large-v3", "engine": "whisper", "source": "HuggingFace (SYSTRAN/faster-whisper-large-v3)",
     "license": "MIT (代码) / MIT (权重)",
     "desc": "OpenAI Whisper large-v3，多语种 code-switching 强"},
    {"id": "qwen2.5-3b-brief", "engine": "brief", "source": "HuggingFace (Qwen/Qwen2.5-3B-Instruct-GGUF)",
     "license": "Apache-2.0",
     "desc": "Qwen2.5-3B-Instruct GGUF (Q4_K_M)，本地纪要总结后端（轻量、快）"},
    {"id": "qwen2.5-7b-brief", "engine": "brief", "source": "HuggingFace (Qwen/Qwen2.5-7B-Instruct-GGUF)",
     "license": "Apache-2.0",
     "desc": "Qwen2.5-7B-Instruct GGUF (Q4_K_M)，本地纪要总结后端（默认，质量更高）"},
]


# ---- 模型版本钉死（供应链安全）----
# 固定到可验证的 commit / 分支，避免上游静默改权重后被拉取到本地。
# - whisper 的 commit 取自 HuggingFace 镜像 API 的 sha 字段；
# - SenseVoiceSmall 的 commit 取自 ModelScope API；
# - 其余辅助模型暂钉 master 分支，生产环境建议进一步钉到具体 commit。
# 改版时必须人工复核这些哈希与上游一致，再行更新。
MODEL_PINS: Dict[str, Dict[str, object]] = {
    "whisper-large-v3": {
        "repo": "Systran/faster-whisper-large-v3",
        "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    },
    "SenseVoiceSmall": {"model": "iic/SenseVoiceSmall", "revision": "7bf452403abd7353a300cd760f7adae7701c92c1"},
    "fsmn-vad": {"model": "iic/fsmn-vad", "revision": "master"},
    "ct-punc": {"model": "iic/ct-punc", "revision": "master"},
    "cam++": {"model": "iic/cam++", "revision": "master"},
    "paraformer-zh": {"model": "iic/paraformer-zh", "revision": "master"},
    # 纪要总结模型（llama-cpp-python 本地 GGUF 后端）。commit 取自 HuggingFace 镜像 API 的 sha 字段。
    "qwen2.5-3b-brief": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "revision": "7dabda4d13d513e3e842b20f0d435c732f172cbe",
        "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
    },
    # 7B 为纪要比默认模型：质量明显优于 3B，RTX 4060 8GB 可显存卸载。
    # 已钉到可验证 commit（2024-09-20，sha 经 hf-mirror API 取得）；上游把 Q4_K_M
    # 以 2 个分片发布。GGUF 分片各自带完整头，不能简单字节拼接成单文件
    # （会触发 llama.cpp "invalid split file name"），故下载器保留原始分片名、
    # 加载器指向首个分片（llama-cpp 自动读取同目录兄弟分片）。part_sizes 用于校验分片完整性。
    "qwen2.5-7b-brief": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "revision": "bb5d59e06d9551d752d08b292a50eb208b07ab1f",
        # file 指向首个分片；llama-cpp 会按 -NNNNN-of-MMMMM 命名自动加载兄弟分片
        "file": "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        "parts": [
            "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
            "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
        ],
        "part_sizes": [3993201344, 689872288],
    },
}


def ms_revision(model_id: str) -> str:
    """返回 ModelScope 模型钉死的 revision（commit 或分支）；缺省 master。"""
    for pin in MODEL_PINS.values():
        if pin.get("model") == model_id:
            return pin.get("revision", "master")
    return "master"


def brief_pin(settings=None) -> Dict[str, str]:
    """返回纪要总结 GGUF 模型钉死的 (repo, revision, file)。

    默认 7B（settings.brief_model 未指定或为 '7b'）；设为 '3b' 退回轻量 3B。
    """
    key = "qwen2.5-7b-brief"
    if settings is not None:
        bm = getattr(settings, "brief_model", None)
        if bm == "3b":
            key = "qwen2.5-3b-brief"
    return MODEL_PINS[key]
