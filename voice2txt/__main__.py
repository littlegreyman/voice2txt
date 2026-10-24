"""允许 `python -m voice2txt` 直接运行 CLI。"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
