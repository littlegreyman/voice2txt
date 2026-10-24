"""将 faster-whisper 的 ASR 段落与说话人分段按时间重叠对齐。"""

from __future__ import annotations

from typing import Dict, List, Optional


def assign_speakers(segments: List[Dict], turns: List[Dict]) -> None:
    """就地给 segments 增加 'speaker' 字段（多数重叠原则）。

    segments / turns 均为 {start, end, ...}；turns 含 'spk'。
    """
    for seg in segments:
        best_spk: Optional[str] = None
        best_overlap = -1.0
        for t in turns:
            ov = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
            if ov > best_overlap:
                best_overlap = ov
                best_spk = t.get("spk")
        seg["speaker"] = best_spk if best_overlap > 0 else None
