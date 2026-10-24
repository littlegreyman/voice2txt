import pytest

from voice2txt.core.vram import VramStrategy, resolve_device


def test_levels_present():
    s = VramStrategy("aggressive", "cpu")
    assert s.level_name in ("float16", "int8_float16", "int8", "cpu")


def test_downgrade_reaches_cpu():
    s = VramStrategy("conservative", "cpu")
    # conservative 只有 int8 -> cpu 两档
    first = s.level_name
    ok = s.downgrade()
    assert ok is True
    assert s.level_name == "cpu"
    assert s.downgrade() is False  # 到底了


def test_resolve_device_cpu_fallback(monkeypatch):
    # 无 CUDA 时应回退 cpu
    import voice2txt.core.vram as vram_mod

    def fake_cuda_available():
        return False

    monkeypatch.setattr(vram_mod, "resolve_device", lambda d: "cpu" if d == "auto" else d)
    assert vram_mod.resolve_device("auto") == "cpu"
