"""音频加载与格式归一化：任意格式 -> 16k 单声道 wav（自包含，不依赖系统 ffmpeg）。"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def load_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """用 PyAV 解码并重采样到 16k 单声道，返回 float32 数组（[-1,1]）。"""
    import av

    with av.open(path, "r") as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"无音频流: {path}")
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=sr
        )
        chunks = []
        for frame in container.decode(stream):
            for res in resampler.resample(frame):
                chunks.append(res.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError(f"空音频: {path}")
    arr = np.concatenate(chunks).astype(np.float32) / 32768.0
    return arr, sr


def to_wav(path: str, sr: int = 16000) -> str:
    """转写为临时 16k 单声道 wav，返回路径（调用方负责清理）。"""
    arr, _ = load_audio(path, sr)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=tempfile.gettempdir())
    tmp.close()
    sf.write(tmp.name, arr, sr)
    return tmp.name


def cleanup(path: str) -> None:
    try:
        if path and os.path.exists(path) and str(Path(path)).endswith(".wav"):
            os.remove(path)
    except Exception:
        pass
