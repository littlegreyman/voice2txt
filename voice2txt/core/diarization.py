"""说话人分离封装：CAM++（FunASR，无 token）。

用于方案 A（faster-whisper）的外挂分离；方案 B 在引擎内已内置。
使用 paraformer-zh + fsmn-vad + cam++ 组合，仅取 sentence_info 的
说话人/时间戳，ASR 文本由 whisper 提供。
"""
from __future__ import annotations

from typing import Dict, List

from ..models.registry import ms_revision


class CamDiarizer:
    def __init__(self, device: str = "cuda"):
        from funasr import AutoModel

        self.model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            spk_model="cam++",
            device=device,
            disable_update=True,
            # 供应链安全：钉死到可验证版本
            model_revision=ms_revision("iic/paraformer-zh"),
            vad_model_revision=ms_revision("iic/fsmn-vad"),
            spk_model_revision=ms_revision("iic/cam++"),
        )

    def diarize(self, wav_path: str, batch_size_s: int = 300) -> List[Dict]:
        res = self.model.generate(input=wav_path, batch_size_s=batch_size_s)
        info = (res[0].get("sentence_info") or []) if res else []
        turns: List[Dict] = []
        for s in info:
            turns.append(
                {
                    "start": s["start"] / 1000.0,
                    "end": s["end"] / 1000.0,
                    "spk": f"SPEAKER_{int(s['spk']):02d}",
                }
            )
        return turns
