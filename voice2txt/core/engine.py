"""引擎抽象与统一内部表示。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Utterance:
    start: float               # 秒
    end: float                 # 秒
    text: str
    speaker: Optional[str] = None   # e.g. "SPEAKER_00"；无说话人分离时为 None


@dataclass
class TranscribeResult:
    utterances: List[Utterance]
    language: Optional[str] = None
    engine: Optional[str] = None


class Engine(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: str, *, diarize: bool,
                   language: Optional[str], vram) -> TranscribeResult:
        """audio_path 为 16k 单声道 wav。返回统一 Utterance 列表。"""
        ...
