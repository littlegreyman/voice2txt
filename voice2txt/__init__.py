"""voice2txt — 纯本地离线语音转写 CLI（方案 B+A）。"""
__version__ = "0.1.0"

from .config import Settings, get_settings, load_config

__all__ = ["Settings", "get_settings", "load_config", "__version__"]
