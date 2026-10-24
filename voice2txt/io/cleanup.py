"""文本后处理：清理 ASR 碎片标点、合并连续同说话人。

FunASR/SenseVoice 的 sentence_info 高度碎片化，经常产出形如
``[SPEAKER_00] ，，``、``[SPEAKER_01] ，你好，``、``[SPEAKER_00] 。 嗯，``
的短片段，且标点（逗号/句号）常误挂到碎片开头。这里做两层清洗：

1. 逐片段清洗：折叠连续重复标点、剔除句首残留标点。
2. 合并（仅 TXT）：把连续同一说话人的碎片拼成一个块，再对最终块做一次
   句尾软标点清理，避免把"句中连接逗号"误删、也避免块尾挂多余逗号。
"""
from __future__ import annotations

import re
from typing import List

from ..core.engine import Utterance

# 句首不应出现的残留标点（VAD/标点模型常把上一句的尾标点误挂到下一碎片开头）
_LEAD_PUNCT = set("，。、？！,.!?;:…—~·")
# 句末可清理的"软标点"（逗号/顿号等）；保留句号/问号/感叹号作为自然断句
_TRAIL_SOFT = set("，、,;:… ")
# 句末"硬标点"（自然断句符，尽量保留）
_HARD_PUNCT = set("。！？!?")
# 折叠连续相同的标点（含空格）："，，，" -> "，"，"。。 " -> "。"
_PUNCT_RE = re.compile(r"([，。、？！,.!?;: ])\1+")
# 删除紧贴在标点两侧的空白（SenseVoice 偶发 "。 ，" 之类）
_SPACE_AROUND_PUNCT_RE = re.compile(r"\s*([，。、？！；：,.!?;:])\s*")
# 至少含一个"实义字符"（字母/数字/汉字）才保留，否则视为纯标点碎片
_CONTENT_RE = re.compile(r"[\w\u3400-\u9fff]")


def _collapse(t: str) -> str:
    return _PUNCT_RE.sub(r"\1", t)


def _strip_leading(t: str) -> str:
    while t and t[0] in _LEAD_PUNCT:
        t = t[1:].strip()
    return t


def _strip_trailing_soft(t: str) -> str:
    """清理块尾标点：删掉尾部软标点；若尾部为"软+硬"或"硬+硬"的冗余组合，
    只保留最后一个硬标点（如 "，。" -> "。"，"。，。" -> "。"）。"""
    while t:
        if t[-1] in _TRAIL_SOFT:
            t = t[:-1]
        elif t[-1] in _HARD_PUNCT and len(t) >= 2 and t[-2] in (_TRAIL_SOFT | _HARD_PUNCT):
            # 末位是硬标点，但其前一位仍是标点（软或硬），说明它是冗余的，
            # 删除前一位、保留末位硬标点
            t = t[:-2] + t[-1]
        else:
            break
    return t


def _squeeze(t: str) -> str:
    """删除标点两侧的空白，并折叠连续相同标点。"""
    t = _SPACE_AROUND_PUNCT_RE.sub(r"\1", t)
    t = _collapse(t)
    return t


def clean_text(text: str) -> str:
    """清洗单段文本（句首/句尾/重复标点/标点两侧空白一次到位）。"""
    t = text.strip()
    if not t:
        return ""
    t = _squeeze(t)
    t = _strip_leading(t)
    t = _strip_trailing_soft(t)
    return t


def _has_content(t: str) -> bool:
    return bool(_CONTENT_RE.search(t))


def _loose_fragment(text: str) -> str:
    """宽松片段清洗：只去句首残留 + 折叠重复 + 去标点两侧空白，保留句尾软标点。

    句尾软标点在合并时可能是合法的句中连接符（如 "嗯，" + "好的"），
    故不在片段阶段删除，留到最终块再统一清理。
    """
    t = text.strip()
    if not t:
        return ""
    t = _squeeze(t)
    t = _strip_leading(t)
    return t


def clean_utterances(utts: List[Utterance],
                     merge_speakers: bool) -> List[Utterance]:
    """清洗片段列表；merge_speakers=True 时合并连续同一说话人。"""
    cleaned: List[Utterance] = []
    for u in utts:
        t = _loose_fragment(u.text)
        if not t or not _has_content(t):
            continue  # 丢弃空片段与纯标点碎片（如 "，，，")
        cleaned.append(Utterance(start=u.start, end=u.end, text=t, speaker=u.speaker))

    if merge_speakers:
        # 把连续同一说话人的碎片直接拼接成一个块（碎片本质是句中续接）
        units: List[Utterance] = []
        for u in cleaned:
            if units and units[-1].speaker == u.speaker:
                prev = units[-1]
                units[-1] = Utterance(
                    start=prev.start,
                    end=u.end,
                    text=_collapse(prev.text + u.text),
                    speaker=prev.speaker,
                )
            else:
                units.append(u)
    else:
        units = cleaned

    # 最终每个单元再统一清理一次：去标点两侧空白 + 折叠重复 + 清理句尾冗余标点
    final: List[Utterance] = []
    for u in units:
        t = _strip_trailing_soft(_squeeze(u.text))
        if not t or not _has_content(t):
            continue
        final.append(Utterance(start=u.start, end=u.end, text=t, speaker=u.speaker))
    return final
