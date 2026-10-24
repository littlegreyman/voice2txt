"""显存策略：激进用满，爆显存自动降级（fp16 -> int8_float16 -> int8 -> cpu）。"""
from __future__ import annotations

import os
from typing import Callable, Optional


def resolve_device(device: str) -> str:
    """auto -> 探测 cuda，否则 mps(Apple Silicon)，否则 cpu。"""
    if device not in ("auto", None, ""):
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon 走 MPS（Metal）加速；无 MPS 支持则退回 CPU
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# 每档策略的尝试顺序（从激进到保守）
_PRESETS = {
    "aggressive":   ["float16", "int8_float16", "int8", "cpu"],
    "balanced":     ["int8_float16", "int8", "cpu"],
    "conservative": ["int8", "cpu"],
    "auto":         ["int8_float16", "int8", "cpu"],  # 8GB 默认从中档起手更安全
}


class VramStrategy:
    """管理 compute_type 与 device，并在 OOM 时自动降级。"""

    def __init__(self, mode: str = "auto", device: str = "auto"):
        self.mode = mode if mode in _PRESETS else "auto"
        self._initial_device = resolve_device(device)
        self.levels = list(_PRESETS[self.mode])
        self.idx = 0
        self.device = self._initial_device
        self.compute_type: Optional[str] = None
        self._apply()

    def _apply(self) -> None:
        lv = self.levels[self.idx]
        if lv == "cpu":
            self.device = "cpu"
            self.compute_type = "int8"
        else:
            self.device = self._initial_device  # 保持 GPU
            self.compute_type = lv

    def downgrade(self) -> bool:
        if self.idx < len(self.levels) - 1:
            self.idx += 1
            self._apply()
            return True
        return False

    @property
    def level_name(self) -> str:
        return self.levels[self.idx]

    def wrap(self, func: Callable, *args, **kwargs):
        """执行 func，遇 CUDA OOM 自动降级重试，最终仍失败则抛出。"""
        last_err: Optional[Exception] = None
        while True:
            try:
                return func(*args, device=self.device, compute_type=self.compute_type, **kwargs)
            except RuntimeError as e:
                msg = str(e).lower()
                if ("out of memory" in msg) or ("cuda" in msg and "alloc" in msg):
                    if self.downgrade():
                        self._free()
                        last_err = e
                        continue
                    raise
                raise
            except Exception:
                raise
        if last_err:
            raise last_err

    @staticmethod
    def _free() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def is_oom_error(e: Exception) -> bool:
    """判断是否为显存不足导致的 CUDA OOM / 分配失败。"""
    msg = str(e).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or ("cuda" in msg and "alloc" in msg)
    )


def apply_offline(env: bool = True) -> None:
    """离线模式：禁止任何模型下载/联网。"""
    if env:
        os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
