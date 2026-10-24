"""方案 B：FunASR 全家桶引擎（SenseVoice/Paraformer + CAM++）。"""
from __future__ import annotations

import os
from typing import List, Optional

from .engine import Engine, TranscribeResult, Utterance
from ..config import Settings
from ..models.registry import ms_revision


# FunASR 的短模型别名（fsmn-vad / ct-punc / cam++）在 ModelScope 缓存里以「完整 id」
# 存储（如 iic/speech_fsmn_vad_zh-cn-16k-common-pytorch），直接传短别名给
# snapshot_download 会找不到缓存。下面这张映射用于在本地命中时还原出完整 id。
_FUNASR_FULL_ID = {
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    "cam++": "iic/speech_campplus_sv_zh-cn_16k-common",
}


def _offline_model_path(hub_id: str, revision: str) -> Optional[str]:
    """若模型已缓存到本地，返回其 snapshot 目录（供 AutoModel 彻底离线加载）。

    仅当本地命中时返回路径；未命中返回 None，由调用方回退到 hub id（首跑联网下载）。
    实测：配合 ``MODELSCOPE_OFFLINE=1`` 下 ``snapshot_download(..., local_files_only=True)``
    完全不触网，彻底消除「每次运行都联网下载模型」的问题。
    """
    candidates = [hub_id]
    if hub_id in _FUNASR_FULL_ID:
        candidates.insert(0, _FUNASR_FULL_ID[hub_id])
    for mid in candidates:
        try:
            from modelscope import snapshot_download
            p = snapshot_download(mid, revision=revision, local_files_only=True)
            if os.path.isdir(p):
                return p
        except Exception:
            continue
    return None


def _pick_model_arg(hub_id: str, revision: str):
    """优先返回本地缓存目录（离线加载），未命中则回退到 hub id（首跑联网）。"""
    local = _offline_model_path(hub_id, revision)
    return local if local else hub_id


class FunASREngine(Engine):
    name = "funasr"

    def __init__(self, settings: Settings):
        self.s = settings
        self._model = None
        self._model_device = None
        self._model_diarize = None

    def _ensure(self, diarize: bool, vram):
        # 按 device / diarize 缓存；OOM 降级后 device 变化会触发重建
        if (
            self._model is None
            or self._model_device != vram.device
            or self._model_diarize != diarize
        ):
            from funasr import AutoModel

            model_id = self.s.resolved_asr_model()
            # 优先用本地 snapshot 目录离线加载（彻底不触网）；本地不存在才回退 hub id。
            # revision 钉死到可验证版本（见 models/registry.py），首跑联网下载时同样生效。
            rev_main = ms_revision(model_id)
            rev_vad = ms_revision("iic/fsmn-vad")
            rev_punc = ms_revision("iic/ct-punc")
            rev_spk = ms_revision("iic/cam++")
            kwargs = dict(
                model=_pick_model_arg(model_id, rev_main),
                vad_model=_pick_model_arg("fsmn-vad", rev_vad),
                punc_model=_pick_model_arg("ct-punc", rev_punc),
                device=vram.device,
                disable_update=True,
                model_revision=rev_main,
                vad_model_revision=rev_vad,
                punc_model_revision=rev_punc,
            )
            if diarize:
                kwargs["spk_model"] = _pick_model_arg("cam++", rev_spk)
                kwargs["spk_model_revision"] = rev_spk
            self._model = AutoModel(**kwargs)
            self._model_device = vram.device
            self._model_diarize = diarize
        return self._model

    def reset(self) -> None:
        self._model = None
        self._model_device = None
        self._model_diarize = None

    def transcribe(self, audio_path: str, *, diarize: bool,
                   language: Optional[str], vram) -> TranscribeResult:
        model = self._ensure(diarize, vram)
        lang = language or "auto"
        gen = dict(
            input=audio_path,
            language=lang,
            use_itn=True,
            batch_size_s=self.s.batch_size_s,
        )
        # 精度杠杆（以效率为代价）：
        # - beam_size>1：更大解码 beam，词错率↓、速度↓（SenseVoice 默认 1）
        # - hotword：上下文偏置，显著提升专有名词 / 人名识别（空格分隔）
        if self.s.beam_size and self.s.beam_size > 1:
            gen["beam_size"] = self.s.beam_size
        if self.s.hotword:
            gen["hotword"] = self.s.hotword
        res = model.generate(**gen)
        data = res[0] if res else {}
        utts: List[Utterance] = []
        info = data.get("sentence_info")
        if info:
            for s in info:
                spk = f"SPEAKER_{int(s['spk']):02d}" if diarize else None
                utts.append(
                    Utterance(
                        start=s["start"] / 1000.0,
                        end=s["end"] / 1000.0,
                        text=s.get("text", ""),
                        speaker=spk,
                    )
                )
        else:
            text = data.get("text", "")
            utts.append(Utterance(start=0.0, end=0.0, text=text, speaker=None))
        return TranscribeResult(utterances=utts, language=lang, engine=self.name)
