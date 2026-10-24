"""会议纪要总结：本地 llama-cpp-python 后端，默认加载 Qwen2.5-7B-Instruct GGUF（Q4_K_M）；
可经 settings.brief_model='3b' 退回轻量 Qwen2.5-3B-Instruct GGUF（更快、占显存小）。

仅对 TXT 输出生效。输出写入 [同名].brief.md 文件（Markdown，见 cli.py）。
完全离线：模型在 `voice2txt setup --engine brief` 阶段一次性下载到本地。

设计要点：
- 纯函数 ``split_into_chunks`` 负责长文本分块（按段落边界，必要时硬切），可被单测覆盖，
  无需加载模型。
- 纯函数 ``dedup_markdown`` 负责结构感知去重：模型偶尔会陷入重复循环（尤其表格），
  此函数在代码层按「表格首列 / 列表条目」去重，是质量兜底，保证最终纪要无冗余。
- ``Summarizer`` 懒加载 GGUF，使用 Qwen2.5 自带 chat 模板生成结构化中文会议纪要；
  超长转录按上下文窗口分块后做一次二次合并，避免超出 32K 上下文。
- 内置 token 安全护栏：若单块仍超出上下文，按模型 tokenizer 精确截断，绝不崩溃。
- 生成参数已调优（repeat_penalty / 低 temperature / 受限 max_tokens）以抑制退化循环。
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from ..config import Settings, get_settings
from ..models.download import brief_local_path, configure_cache

# Qwen2.5 上下文窗口与预算（tokens）
_CONTEXT_TOKENS = 32768
# 输入预算 = 上下文 - 系统提示(~200) - 输出预算(max_tokens) - 安全余量
_INPUT_BUDGET_TOKENS = 28000
# 字符/token 的保守估算（中文偏 ~1.5，英文 ~4，取 2.0 作为最坏情形上限，确保不溢出）
_CHARS_PER_TOKEN = 2.0

_SYSTEM_PROMPT = (
    "你是一款专业的会议纪要生成助手，擅长把冗长的会议/通话语音转录整理为"
    "结构化的中文会议纪要（Markdown 格式）。\n"
    "核心要求：\n"
    "1. 严格只依据给定转录，不编造未提及的人名、数据、结论或行动项；\n"
    "2. 信息密度要高：保留关键决策、具体数据、明确待办，不要泛泛而谈或过度压缩；\n"
    "3. 行动项尽量完整：事项/责任人/时限三列都要填；转录未明确责任人或时限时写"
    "「待定」，不要留空或省略；同一行动项只列一次，不重复；\n"
    "4. 关键结论与讨论要点要具体到原文要点，必要时保留关键数字、时间节点与分歧。"
)

_USER_PROMPT = (
    "请把下面这段会议转录整理为会议纪要，输出中文 Markdown，结构如下：\n"
    "1. 概述（2-3 句话，说明会议主题与主要结论）\n"
    "2. 关键结论与决议（要点式，每一条都要有实质内容，不重复）\n"
    "3. 行动项（用表格 | 事项 | 责任人 | 时限 |，逐条列出；未明确者写「待定」）\n"
    "4. 主要讨论要点（要点式，保留讨论中的关键分歧、数据与依据）\n"
    "5. 遗留问题/风险（如有，具体说明）\n\n"
    "注意：转录较长时更要写全，不要因篇幅大而压缩成寥寥数语；"
    "每条结论/行动项都应有信息量。\n\n"
    "转录文本：\n"
)

_FINAL_PROMPT = (
    "下面是同一会议分多段转录后各自生成的摘要。请去重、消除冲突，"
    "整合成一份完整、连贯、可直接使用的中文 Markdown 会议纪要"
    "（沿用 概述/关键结论/行动项/主要讨论要点/遗留问题 结构）。\n"
    "要求：合并重复条目；保留全部具体行动项（含责任人、时限，未明确写「待定」）；"
    "不要删减有信息量的结论与讨论要点；不编造。最终纪要应信息完整、条理清晰。\n\n"
)


def _norm_key(s: str) -> str:
    """把一条文本归一化为去重键：去列表符号、去空白与标点、转小写。"""
    s = (s or "").strip().lower()
    s = re.sub(r"^[\-\*\d+\.\s、]+", "", s)  # 去掉开头的 - * 1. 2、等列表符号
    s = re.sub(r"[\s，。、；：,.!?；:!?\"'（）()\[\]【】]+", "", s)  # 去标点空白
    return s


def _is_table_sep(cells: List[str]) -> bool:
    """判断是否为 Markdown 表格分隔行（如 | :--- | :--- |）。"""
    merged = "".join(cells)
    return merged != "" and set(merged) <= set("-: ")


def dedup_markdown(text: str) -> str:
    """结构感知地去重重排 Markdown：表格按首列去重、无序/有序列表按条目去重。

    模型在生成长表格时偶尔陷入重复循环（同一行输出几十遍），本函数在代码层兜底，
    保证最终纪要无冗余。纯函数，不依赖模型，可被单测覆盖。
    """
    if not text:
        return text
    lines = text.split("\n")
    out: List[str] = []
    seen_table: set = set()
    seen_bullet: set = set()
    in_table = False
    for line in lines:
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if _is_table_sep(cells):
                # 分隔行：原样保留，不改变表格状态
                out.append(line)
                continue
            if not in_table:
                # 表头行：进入表格区
                in_table = True
                seen_table = set()
                out.append(line)
                continue
            # 数据行：按「事项」首列去重
            key = _norm_key(cells[0]) if cells else ""
            if key in seen_table:
                continue
            seen_table.add(key)
            out.append(line)
            continue
        else:
            in_table = False
        # 列表条目（无序 - / * 及有序 1. 2、）去重
        if re.match(r"^[\-\*]\s+\S", s) or re.match(r"^\d+[.、]\s+\S", s):
            key = _norm_key(s)
            if key in seen_bullet:
                continue
            seen_bullet.add(key)
        out.append(line)
    return "\n".join(out).strip() + "\n"


def split_into_chunks(text: str, max_chars: int) -> List[str]:
    """把长文本切成若干块，每块不超过 ``max_chars`` 字符。

    优先在空行（段落）边界切分；单个段落超过 ``max_chars`` 时再按字符硬切，
    硬切点尽量落在句子结尾（。！？；.）之后。纯函数，便于单测，不依赖模型。
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # 先按空行拆成段落
    paras = [p for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_len = 0

    def hard_split(long_para: str) -> None:
        # 在句子边界附近硬切，避免把句子从中剖开
        start = 0
        n = len(long_para)
        while start < n:
            end = min(start + max_chars, n)
            if end < n:
                # 向前找最近的句子结尾
                cut = end
                for j in range(end, max(start, end - 200), -1):
                    if long_para[j - 1] in "。！？；.!?":
                        cut = j
                        break
                end = cut
            chunks.append(long_para[start:end].strip())
            start = end

    for p in paras:
        if len(p) > max_chars:
            flush()
            hard_split(p)
            continue
        if buf_len + len(p) + 2 > max_chars and buf:
            flush()
        buf.append(p)
        buf_len += len(p) + 2
    flush()
    return [c for c in chunks if c]


def _ensure_cuda_runtime_on_path() -> None:
    """best-effort：把 torch 的 lib 目录加入 PATH，使 CUDA 版 llama-cpp-python 的
    ``llama.dll`` 能找到 ``cudart64_12.dll`` 等 CUDA 运行时（否则报「找不到模块或其依赖」）。

    仅在 torch 已安装且含 CUDA 运行时库时生效；CPU 版 / 无 torch 环境静默跳过，不影响其它路径。
    """
    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if torch_lib and os.path.isdir(torch_lib):
            path = os.environ.get("PATH", "")
            if torch_lib not in path:
                os.environ["PATH"] = torch_lib + os.pathsep + path
    except Exception:
        pass


class Summarizer:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_path: Optional[str] = None,
        n_gpu_layers: int = -1,
        max_tokens: int = 2048,
    ) -> None:
        self.settings = settings or get_settings()
        configure_cache(self.settings)
        self._model_path = model_path or brief_local_path(self.settings)
        self._n_gpu_layers = n_gpu_layers
        self._max_tokens = max_tokens
        self._max_chars = int(_INPUT_BUDGET_TOKENS * _CHARS_PER_TOKEN)
        self._llm = None

    @property
    def model_path(self) -> str:
        return self._model_path

    def available(self) -> bool:
        """模型权重是否已就绪（用于 CLI 提前给出友好提示）。"""
        return os.path.isfile(self._model_path) and os.path.getsize(self._model_path) > 1_000_000_000

    def _ensure(self) -> None:
        if self._llm is not None:
            return
        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"纪要模型未找到：{self._model_path}\n"
                f"请先联网运行：voice2txt setup --engine brief"
            )
        try:
            _ensure_cuda_runtime_on_path()
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "未安装 llama-cpp-python。请运行：pip install 'voice2txt[brief]' "
                "（或 pip install llama-cpp-python）"
            ) from e
        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=_CONTEXT_TOKENS,
            n_gpu_layers=self._n_gpu_layers,  # -1 尽数卸载到 GPU；CPU-only 构建自动回退 CPU
            n_threads=os.cpu_count() or 4,
            verbose=False,
        )

    def _fit_text(self, text: str, budget_tokens: int) -> str:
        """若模型已加载且文本 token 数超出预算，按 tokenizer 精确截断（安全护栏）。"""
        if self._llm is None:
            return text
        try:
            ids = self._llm.tokenize(text.encode("utf-8"))
        except Exception:
            return text
        if len(ids) <= budget_tokens:
            return text
        # 截断到预算内并重新解码（可能含不完整尾字，影响很小）
        trimmed = ids[:budget_tokens]
        try:
            return self._llm.detokenize(trimmed).decode("utf-8", errors="ignore")
        except Exception:
            return text

    def _summarize_one(self, text: str, on_progress=None) -> str:
        self._ensure()
        safe = self._fit_text(text, _INPUT_BUDGET_TOKENS)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT + safe},
        ]
        resp = self._llm.create_chat_completion(
            messages=messages,
            temperature=0.2,
            top_p=0.85,
            repeat_penalty=1.2,
            # frequency_penalty 不宜过高：过高会让输出过于简短（纪要空洞的主因之一）。
            # 表格重复循环由 dedup_markdown 在代码层兜底，故此处只需轻度抑制即可。
            frequency_penalty=0.2,
            presence_penalty=0.1,
            max_tokens=self._max_tokens,
        )
        content = resp["choices"][0]["message"]["content"]
        return dedup_markdown(content.strip()) if isinstance(content, str) else ""

    def summarize(self, transcript: str, on_progress=None) -> str:
        """把转录文本整理为结构化中文会议纪要（Markdown 字符串）。

        长文本先分块各自摘要，再做一次二次合并，避免超出 32K 上下文窗口。
        每层均经过 ``dedup_markdown`` 去重兜底，确保最终纪要无冗余。
        ``on_progress`` 为可选回调（msg:str），用于向用户实时反馈进度
        （CPU 模式下生成本地纪要较慢，避免看起来像卡死）。
        """
        transcript = (transcript or "").strip()
        if not transcript:
            return ""
        prog = on_progress or (lambda m: None)
        chunks = split_into_chunks(transcript, self._max_chars)
        # 单块且本身很短：直接摘要
        if len(chunks) == 1:
            prog("正在生成本地会议纪要（本地 LLM，CPU 模式下可能较慢）…")
            return self._summarize_one(chunks[0], on_progress=prog)
        prog(f"转录较长，分 {len(chunks)} 段各自摘要（本地 LLM，请稍候）…")
        partials = []
        for i, c in enumerate(chunks, 1):
            prog(f"摘要进度 {i}/{len(chunks)}")
            partials.append(dedup_markdown(self._summarize_one(c, on_progress=prog)))
        prog("合并各段摘要为最终纪要…")
        merged = "\n\n".join(
            f"【第 {i + 1} 段摘要】\n{p}" for i, p in enumerate(partials)
        )
        return dedup_markdown(self._summarize_one(_FINAL_PROMPT + merged, on_progress=prog))
