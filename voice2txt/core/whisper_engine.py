"""方案 A：faster-whisper 引擎 + CAM++ 外挂分离 + 时间对齐。"""
from __future__ import annotations

import os
from typing import List, Optional

from .engine import Engine, TranscribeResult, Utterance
from .align import assign_speakers
from .diarization import CamDiarizer
from ..config import Settings
from ..models.download import whisper_local_dir


class WhisperEngine(Engine):
    name = "whisper"

    def __init__(self, settings: Settings):
        self.s = settings
        self._model = None
        self._model_key = None

    def _resolve_model(self) -> str:
        """优先用本地平铺目录（setup 下载的）；缺失时回退 HF repo id。"""
        local = whisper_local_dir(self.s)
        if os.path.isfile(os.path.join(local, "model.bin")):
            return local
        return self.s.resolved_asr_model()

    def _ensure(self, vram):
        # 按 device / compute_type 缓存；OOM 降级后 key 变化触发重建
        key = (vram.device, vram.compute_type)
        if self._model is None or self._model_key != key:
            from faster_whisper import WhisperModel, BatchedInferencePipeline

            model_id = self._resolve_model()
            dev = vram.device
            # CTranslate2 无 Metal 后端：macOS(mps) 走 CPU 推理；其余 GPU 用对应 compute_type
            ct = "int8" if dev in ("cpu", "mps") else vram.compute_type
            ct_dev = "cpu" if dev == "mps" else dev
            try:
                m = WhisperModel(model_id, device=ct_dev, compute_type=ct)
            except Exception as e:
                raise RuntimeError(
                    f"加载 whisper 模型失败：{e}\n"
                    f"若提示找不到模型，请先联网执行一次 `voice2txt setup --engine whisper` "
                    f"下载权重（默认缓存于 {whisper_local_dir(self.s)}），之后即可离线使用。"
                ) from e
            self._model = BatchedInferencePipeline(model=m)
            self._model_key = key
        return self._model

    def reset(self) -> None:
        self._model = None
        self._model_key = None

    def transcribe(self, audio_path: str, *, diarize: bool, language: Optional[str],
                   vram) -> TranscribeResult:
        pipe = self._ensure(vram)
        # faster-whisper 用 None 表示自动检测语种，"auto" 非法，需转换
        lang = None if (not language or language == "auto") else language
        seg_iter, info = pipe.transcribe(
            audio_path,
            batch_size=16,
            language=lang,
            vad_filter=True,
        )
        segs = [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in seg_iter
        ]

        if diarize:
            dia = CamDiarizer(device=vram.device)
            turns = dia.diarize(audio_path, batch_size_s=self.s.batch_size_s)
            assign_speakers(segs, turns)

        utts: List[Utterance] = [
            Utterance(
                start=x["start"],
                end=x["end"],
                text=x["text"],
                speaker=x.get("speaker"),
            )
            for x in segs
        ]
        return TranscribeResult(
            utterances=utts,
            language=getattr(info, "language", None),
            engine=self.name,
        )
