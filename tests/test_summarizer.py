"""summarizer 单测：覆盖纯函数分块逻辑，以及对 LLM 后端的最小集成（用假后端，无需真模型）。"""
from __future__ import annotations

import os

from voice2txt.config import Settings
from voice2txt.core import summarizer
from voice2txt.core.summarizer import Summarizer, dedup_markdown, split_into_chunks


def test_split_short_returns_single_chunk():
    text = "这是一段很短的会议转录。"
    chunks = split_into_chunks(text, max_chars=1000)
    assert chunks == [text]


def test_split_empty_returns_empty():
    assert split_into_chunks("", max_chars=1000) == []
    assert split_into_chunks("   \n  ", max_chars=1000) == []


def test_split_long_multiline_respects_max_chars():
    # 用多个段落拼出远超 max_chars 的文本
    max_chars = 500
    para = "会议讨论了项目进度与风险。" * 10  # ~ 110 字
    text = "\n\n".join([para for _ in range(20)])  # 20 段
    assert len(text) > max_chars * 2
    chunks = split_into_chunks(text, max_chars=max_chars)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= max_chars + 200  # 允许一个段落略微超出（单段不超过 max_chars）


def test_split_single_huge_paragraph_hard_splits():
    max_chars = 300
    giant = "客户要求在月底交付新版客户端并修复已知崩溃。" * 50  # 单段远超 max_chars
    chunks = split_into_chunks(giant, max_chars=max_chars)
    assert len(chunks) >= 2
    # 硬切后每块都不应显著超过 max_chars（末尾可能多一段句子，留 200 余量）
    for c in chunks:
        assert len(c) <= max_chars + 200


def test_summarize_single_chunk_with_fake_backend():
    # 假后端：直接回显用户内容长度，验证 summarize 调用链路与 chunk 数无关
    captured = {}

    class FakeLlama:
        def tokenize(self, b):
            return list(b)  # 1 token/byte，便于触发护栏但本测试不会超预算

        def detokenize(self, ids):
            return bytes(ids)

        def create_chat_completion(self, messages, **kw):
            captured["messages"] = messages
            user = messages[-1]["content"]
            return {"choices": [{"message": {"content": f"摘要::{len(user)}"}}]}

    s = Summarizer(settings=Settings(), model_path="fake.gguf")
    s._llm = FakeLlama()  # 跳过真实模型加载与文件检查
    out = s.summarize("短会议转录内容。")
    assert out.startswith("摘要::")
    # 系统提示 + 用户提示应出现
    assert any(m["role"] == "system" for m in captured["messages"])


def test_summarize_long_text_triggers_merge():
    calls = []

    class FakeLlama:
        def tokenize(self, b):
            return list(b)

        def detokenize(self, ids):
            return bytes(ids)

        def create_chat_completion(self, messages, **kw):
            calls.append(messages[-1]["content"])
            return {"choices": [{"message": {"content": "部分摘要"}}]}

    # 构造远超单块上限的文本（max_chars 默认约 56000）
    big = "项目进度与风险讨论要点。" * 8000  # ~ 12 万字符
    s = Summarizer(settings=Settings(), model_path="fake.gguf")
    s._llm = FakeLlama()
    out = s.summarize(big)
    # 多块 -> 先做分块摘要，再做一次合并摘要，至少 2 次调用
    assert len(calls) >= 2
    assert out.strip() == "部分摘要"  # dedup_markdown 会补结尾换行


def test_dedup_markdown_collapses_repeated_table_rows():
    # 模拟模型陷入重复循环：同一行输出几十遍
    header = "| 事项 | 责任人 | 时限 |\n| :--- | :--- | :--- |"
    row = "| 会议名称调整为X | 无 | 无 |"
    bad = header + "\n" + "\n".join([row] * 50)
    clean = dedup_markdown(bad)
    data_rows = [l for l in clean.split("\n") if l.strip().startswith("|") and "事项" not in l and "---" not in l]
    assert len(data_rows) == 1, data_rows
    # 表头与分隔行保留
    assert "| 事项 | 责任人 | 时限 |" in clean
    assert "| :--- | :--- | :--- |" in clean


def test_dedup_markdown_collapses_repeated_bullets():
    bad = "- 完成了需求评审\n" * 8 + "- 推动了上线排期\n" * 5
    clean = dedup_markdown(bad)
    assert clean.count("- 完成了需求评审") == 1
    assert clean.count("- 推动了上线排期") == 1


def test_dedup_markdown_preserves_distinct_rows():
    # 不同措辞（将X vs X）归一化键不同，应都保留，不应误合并
    text = (
        "| 事项 | 责任人 | 时限 |\n| :--- | :--- | :--- |\n"
        "| 会议时间调整为2点半 | 无 | 无 |\n"
        "| 将会议时间调整为2点半 | 无 | 无 |\n"
    )
    clean = dedup_markdown(text)
    # 表头+分隔+2 行不同措辞的数据行都应保留
    assert "| 事项 | 责任人 | 时限 |" in clean
    assert "| :--- | :--- | :--- |" in clean
    assert clean.count("| 会议时间调整为2点半 |") == 1
    assert clean.count("| 将会议时间调整为2点半 |") == 1
    # 数据行总数应为 2
    data_rows = [l for l in clean.split("\n") if l.strip().startswith("|")
                 and "事项" not in l and "---" not in l]
    assert len(data_rows) == 2


def test_dedup_markdown_keeps_empty_input():
    assert dedup_markdown("") == ""
    assert dedup_markdown("   \n  ") == "   \n  ".strip() + "\n"


def test_summarizer_model_missing_raises_clear_error(tmp_path):
    # 不注入假后端、且模型文件不存在 -> 应给出可定位的文件未找到错误
    missing = tmp_path / "no_such_model.gguf"
    s = Summarizer(settings=Settings(), model_path=str(missing))
    try:
        s.summarize("x")
        assert False, "应当抛出 FileNotFoundError"
    except FileNotFoundError as e:
        assert "setup --engine brief" in str(e)
