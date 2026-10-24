"""输出格式化：TXT / SRT。输入统一为 Utterance 列表。

清洗策略（详见 io.cleanup）：
- SRT 保留细粒度时间戳，但逐段清理碎片标点、丢弃纯标点空段；
- TXT 额外合并连续同一说话人，避免被切成过碎的短行。
"""
from __future__ import annotations

from typing import List

from ..core.engine import Utterance
from .cleanup import clean_utterances


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _line(u: Utterance, speaker_labels: bool, with_ts: bool = False) -> str:
    prefix = ""
    if with_ts:
        prefix += f"[{_ts(u.start)}] "
    if speaker_labels and u.speaker:
        prefix += f"[{u.speaker}] "
    return prefix + u.text


def format_srt(utts: List[Utterance], speaker_labels: bool = True) -> str:
    # SRT 需要时间戳，不合并说话人；仅清理标点与空段
    items = clean_utterances(utts, merge_speakers=False)
    out: List[str] = []
    idx = 1
    for u in items:
        # 无效时间窗（如整体一段）给一个最小长度避免 SRT 非法
        end = u.end if u.end > u.start else u.start + 1.0
        out.append(str(idx))
        out.append(f"{_ts(u.start)} --> {_ts(end)}")
        out.append(_line(u, speaker_labels))
        out.append("")
        idx += 1
    return "\n".join(out).rstrip() + "\n"


def format_txt(utts: List[Utterance], speaker_labels: bool = True,
               with_timestamps: bool = False) -> str:
    # TXT 合并连续同一说话人，输出更连贯的段落
    items = clean_utterances(utts, merge_speakers=True)
    lines = [_line(u, speaker_labels, with_ts=with_timestamps) for u in items]
    return "\n".join(lines).rstrip() + "\n"
