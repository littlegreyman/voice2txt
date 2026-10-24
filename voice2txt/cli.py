"""voice2txt CLI 入口（PowerShell / CMD / Linux 通用）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .config import Settings, get_settings


def build_engine(name: str, settings: Settings):
    if name == "funasr":
        from .core.funasr_engine import FunASREngine
        return FunASREngine(settings)
    if name == "whisper":
        from .core.whisper_engine import WhisperEngine
        return WhisperEngine(settings)
    raise ValueError(f"未知引擎: {name}")


def transcribe_cmd(input_path: str, settings: Settings) -> None:
    from .core.vram import VramStrategy, apply_offline, is_oom_error
    from .io.audio import to_wav, cleanup
    from .io.formatter import format_srt, format_txt
    from .models.download import configure_cache

    # --brief 仅对 TXT 输出有意义（SRT 是带时间戳的字幕，不适合做纪要输入）
    if settings.brief and settings.fmt != "txt":
        print("[error] --brief 仅支持 -f txt；请改用 --format txt 以生成会议纪要。")
        return 2

    # 离线优先（需求 #1：完全离线）。模型已在 `voice2txt setup` 阶段一次性下载
    # 到本地缓存，运行时强制离线可保证断网可用，也避免每次运行都去 hub 校验/下载。
    apply_offline(True)
    configure_cache(settings)

    vram = VramStrategy(settings.vram, settings.device)
    print(
        f"[info] engine={settings.engine} device={vram.device} "
        f"compute={vram.compute_type} diarize={settings.diarize} "
        f"format={settings.fmt}"
    )

    wav = to_wav(input_path)
    try:
        engine = build_engine(settings.engine, settings)
        # 爆显存自动降级：捕获 CUDA OOM -> 降级策略 -> 重建模型重试
        while True:
            try:
                result = engine.transcribe(
                    wav,
                    diarize=settings.diarize,
                    language=settings.language,
                    vram=vram,
                )
                break
            except RuntimeError as e:
                if is_oom_error(e) and vram.downgrade():
                    engine.reset()
                    print(
                        f"[warn] 显存不足，已降级到 {vram.level_name} "
                        f"（device={vram.device}）后重试..."
                    )
                    continue
                raise
    finally:
        cleanup(wav)

    text = (
        format_srt(result.utterances, settings.speaker_labels)
        if settings.fmt == "srt"
        else format_txt(result.utterances, settings.speaker_labels)
    )

    if settings.output:
        out = Path(settings.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        # 用 UTF-8 with BOM：Windows 记事本/大部分播放器默认按 GBK 解读无 BOM
        # 的 UTF-8 文件会显示为乱码，BOM 可正确识别中文。
        out.write_text(text, encoding="utf-8-sig")
        print(f"[ok] 已写出 {out}（{len(result.utterances)} 段，引擎={result.engine}）")
    else:
        sys.stdout.write(text)

    # 会议纪要总结：仅 -f txt + --brief。写入 [同名].brief.md（Markdown，与输出同 stem，UTF-8 无 BOM，
    # 可直接在 VS Code / Obsidian 等预览；.brief 段保留原纪要标识）。
    if settings.brief:
        from .core.summarizer import Summarizer

        brief_path = (
            out.with_name(out.stem + ".brief.md")
            if settings.output
            else Path(input_path).with_name(Path(input_path).stem + ".brief.md")
        )
        try:
            summ = Summarizer(settings)
            if not summ.available():
                print(f"[warn] 纪要模型未缓存（{summ.model_path}），跳过 --brief；"
                      f"请先运行 `voice2txt setup --engine brief` 联网下载一次。")
            else:
                # on_progress 把本地 LLM 的生成进度实时打印出来，避免 CPU 模式下像卡死
                brief = summ.summarize(
                    text,
                    on_progress=lambda m: print(f"[brief] {m}", flush=True),
                )
                brief_path.parent.mkdir(parents=True, exist_ok=True)
                brief_path.write_text(brief, encoding="utf-8")
                print(f"[ok] 已写出会议纪要 {brief_path}（{len(brief)} 字）")
        except Exception as e:  # 纪要失败不应影响主产物 TXT
            print(f"[warn] 会议纪要生成失败（已跳过）：{e}")
    return 0


def main(argv: Optional[list] = None) -> int:
    # 控制台 stdout 强制 UTF-8，避免中文在部分终端下乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        prog="voice2txt",
        formatter_class=fmt,
        description=(
            "voice2txt — 纯本地离线语音转写 CLI\n"
            "\n"
            "把会议录音 / 手机录音转为 SRT 字幕或纯文本，并可一键生成结构化 Markdown 会议纪要。\n"
            "完全离线：模型仅在 `voice2txt setup` 时联网下载一次，运行时强制离线，无需任何 API key / token。\n"
            "（等价调用：python -m voice2txt ...）"
        ),
        epilog=(
            "典型场景：\n"
            "  # 1) 转写为 SRT 字幕（默认格式，保留逐句时间戳与说话人）\n"
            "  voice2txt transcribe meeting.m4a\n"
            "\n"
            "  # 2) 转写为 TXT 文本（带说话人标签）\n"
            "  voice2txt transcribe meeting.m4a -f txt -o meeting.txt\n"
            "\n"
            "  # 3) 转写并自动生成会议纪要（写入 meeting.brief.md）\n"
            "  voice2txt transcribe meeting.m4a -f txt -o meeting.txt --brief\n"
            "\n"
            "  # 4) 首次使用先下载模型（联网一次）\n"
            "  voice2txt setup\n"
            "\n"
            "更多子命令用法：voice2txt transcribe -h / voice2txt setup -h"
        ),
    )
    sub = parser.add_subparsers(
        dest="cmd", required=True, title="可用子命令", metavar="<command>"
    )

    p_tr = sub.add_parser(
        "transcribe",
        formatter_class=fmt,
        help="转写音频文件为 SRT / TXT（可加 --brief 生成会议纪要）",
        description=(
            "转写音频文件。默认输出 SRT 字幕（逐句时间戳 + 说话人标签）；\n"
            "用 -f txt 输出纯文本；加 --brief 额外生成结构化 Markdown 会议纪要（[同名].brief.md）。\n"
            "运行时强制离线，断网可用。"
        ),
        epilog=(
            "示例：\n"
            "  # 基础转写（SRT，默认）\n"
            "  voice2txt transcribe recording.m4a\n"
            "\n"
            "  # 纯文本 + 关闭说话人分离（更快）\n"
            "  voice2txt transcribe recording.m4a -f txt --no-diarize -o recording.txt\n"
            "\n"
            "  # 中文会议 + 生成会议纪要（写入 recording.brief.md）\n"
            "  voice2txt transcribe recording.m4a -f txt -o recording.txt --brief\n"
            "\n"
            "  # 指定语言为粤语\n"
            "  voice2txt transcribe recording.m4a -l yue -f txt\n"
            "\n"
            "  # 显存吃紧时用更保守的显存策略\n"
            "  voice2txt transcribe recording.m4a --vram conservative\n"
            "\n"
            "  # 用备用引擎 whisper（需先 `voice2txt setup --engine whisper`）\n"
            "  voice2txt transcribe recording.m4a -e whisper -f txt\n"
            "\n"
            "说明：\n"
            "  - 输出文件带 UTF-8 BOM（Windows 记事本不乱码）；纪要 .brief.md 为 UTF-8 无 BOM。\n"
            "  - --brief 仅对 -f txt 生效；与 -f srt 一起会报错提示。\n"
            "  - 运行时强制离线；若提示模型缺失，先 `voice2txt setup` 联网下载一次。"
        ),
    )
    p_tr.add_argument("input", help="音频文件路径（mp3 / wav / m4a / flac / ogg 等，任意 ffmpeg 可解码格式）")
    p_tr.add_argument("-e", "--engine", choices=["funasr", "whisper"],
                      help="ASR 引擎：funasr（默认，中文最强、自带标点）/ whisper（备选，多语种兜底）")
    p_tr.add_argument("--no-diarize", action="store_true", help="不区分说话人（关闭说话人分离，显著提速）")
    p_tr.add_argument("-f", "--format", choices=["srt", "txt"],
                      help="输出格式：srt（默认，带逐句时间戳）/ txt（纯文本段落）")
    p_tr.add_argument("--brief", action="store_true",
                      help="仅 -f txt 生效：额外生成结构化 Markdown 会议纪要，写入 [同名].brief.md 文件")
    p_tr.add_argument("-l", "--language", help="语言：auto / zh / en / ja / ko / yue（粤语）/ …；默认 auto（自动检测）")
    p_tr.add_argument("-o", "--output", help="输出文件路径；不填则打印到终端")
    p_tr.add_argument("--vram", choices=["auto", "aggressive", "balanced", "conservative"],
                      help="显存策略：auto（默认，按可用显存自适应）/ aggressive（占满提速）/ balanced / conservative（省显存）")
    p_tr.add_argument("--no-speaker-labels", action="store_true",
                      help="输出不带 [SPEAKER_xx] 前缀（仍做分离，仅不打印标签）")
    p_tr.add_argument("--offline", action="store_true", help="强制离线（现已是默认行为，保留用于显式声明）")
    p_tr.add_argument("--asr-model", help="覆盖 ASR 模型 id（高级，默认按引擎自动选）")
    p_tr.add_argument("--beam-size", type=int, default=None,
                      help="SenseVoice 解码 beam（默认 1）；调大(5~10)可提升词准确率，但更慢")
    p_tr.add_argument("--hotword", default=None,
                      help="上下文偏置热词（空格分隔），提升专有名词/人名准确率，如 '张三 项目A'")
    p_tr.add_argument("--brief-model", choices=["3b", "7b"], default=None,
                      help="纪要 LLM 规模：7b（默认，质量高）/ 3b（轻量更快）")

    p_setup = sub.add_parser(
        "setup",
        formatter_class=fmt,
        help="一次性下载模型（需联网，仅首次）",
        description=(
            "一次性下载所需模型到本地缓存（仅首次需联网；之后运行强制离线）。\n"
            "可选 engine：all（默认，全部）/ funasr / whisper / brief（仅纪要 LLM）。"
        ),
        epilog=(
            "示例：\n"
            "  # 下载全部模型（funasr + whisper 备选 + 纪要 LLM）\n"
            "  voice2txt setup\n"
            "\n"
            "  # 仅下载会议纪要用的本地 LLM（默认 Qwen2.5-7B GGUF；--brief-model 3b 可选 3B）\n"
            "  voice2txt setup --engine brief\n"
            "\n"
            "  # 仅下载 whisper 备选引擎（国内网络自动走镜像，关 Xet）\n"
            "  voice2txt setup --engine whisper"
        ),
    )
    p_setup.add_argument("--engine", choices=["all", "funasr", "whisper", "brief"], default="all",
                         help="要下载的引擎：all（默认）/ funasr / whisper / brief（纪要 LLM）")

    p_serv = sub.add_parser(
        "serve",
        formatter_class=fmt,
        help="启动内网转录服务（预留）",
        description="启动内网 HTTP 转录服务（当前为预留 stub，尚未实现完整服务端）。",
    )
    p_serv.add_argument("--host", default="127.0.0.1",
                        help="监听地址（默认仅本机 127.0.0.1；内网开放需显式设 0.0.0.0 并自管防火墙）")
    p_serv.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")

    sub.add_parser("version", formatter_class=fmt, help="显示版本号")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.cmd == "version":
        print(__version__)
        return 0
    if args.cmd == "setup":
        from .models.download import prefetch
        prefetch(args.engine)
        return 0
    if args.cmd == "serve":
        from .server.app import run_server
        run_server(args.host, args.port)
        return 0

    # transcribe
    if args.engine:
        settings.engine = args.engine
    if args.format:
        settings.fmt = args.format
    if args.language:
        settings.language = args.language
    if args.vram:
        settings.vram = args.vram
    if args.asr_model:
        settings.asr_model = args.asr_model
    if args.beam_size is not None:
        settings.beam_size = args.beam_size
    if args.hotword:
        settings.hotword = args.hotword
    if args.brief_model:
        settings.brief_model = args.brief_model
    settings.diarize = not args.no_diarize
    settings.speaker_labels = not args.no_speaker_labels
    if args.offline:
        settings.offline = True
    if args.brief:
        settings.brief = True
    if args.output:
        settings.output = args.output

    return transcribe_cmd(args.input, settings)


if __name__ == "__main__":
    raise SystemExit(main())
