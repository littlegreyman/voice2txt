"""桌面 GUI（预留，P7 阶段实现）。Textual TUI / PyQt 均可挂接同一引擎层。"""
from __future__ import annotations


def run_gui() -> None:
    try:
        from textual.app import App  # noqa: F401
    except ImportError:
        print("请先安装 gui 扩展：pip install voice2txt[gui]")
        return
    # TODO(P7): 构建文件选择 -> 进度条 -> 结果预览/导出
    print("[gui] 桌面 GUI 为预留骨架，将在 P7 阶段实现，复用 core 引擎层。")
