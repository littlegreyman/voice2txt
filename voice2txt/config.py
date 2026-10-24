"""配置加载与全局设置。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models.registry import default_asr_model

CONFIG_PATHS = [Path("config.yaml"), Path.home() / ".config" / "voice2txt" / "config.yaml"]


@dataclass
class Settings:
    engine: str = "funasr"           # funasr | whisper
    device: str = "auto"             # auto | cuda | cuda:0 | cpu
    vram: str = "auto"               # auto | aggressive | balanced | conservative
    diarize: bool = True
    fmt: str = "srt"                 # srt | txt
    language: Optional[str] = None   # auto | zh | en | ja | ko | yue | ...
    asr_model: str = "auto"          # auto -> 按引擎默认
    speaker_labels: bool = True
    batch_size_s: int = 300
    max_speakers: Optional[int] = None
    models_dir: Optional[str] = None
    offline: bool = False
    brief: bool = False              # 是否生成会议纪要 [同名].brief.md（仅 -f txt）
    output: Optional[str] = None      # 输出文件路径；None 则打印到 stdout

    def resolved_asr_model(self) -> str:
        if self.asr_model and self.asr_model != "auto":
            return self.asr_model
        return default_asr_model(self.engine)


_settings: Optional[Settings] = None


def load_config(path: Optional[Path] = None) -> Settings:
    """从 YAML 加载默认配置；CLI 会在此基础上覆盖。"""
    import yaml  # 延迟导入，避免无依赖时崩溃

    cfg = Settings()
    candidate = path or next((p for p in CONFIG_PATHS if p.exists()), None)
    if candidate and candidate.exists():
        with open(candidate, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings


def set_settings(s: Settings) -> None:
    global _settings
    _settings = s
