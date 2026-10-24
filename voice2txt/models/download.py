"""安装期一次性模型下载（需联网；之后可离线）。"""
from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from typing import Dict, List, Optional

from ..config import Settings, get_settings
from ..models.registry import MODEL_PINS, brief_pin

# FunASR 公开模型（ModelScope，无需 token）
# 注意：fsmn-vad / ct-punc / cam++ 是作为主模型的 vad_model/punc_model/spk_model
# 参数被触发下载的，单独以 model= 构造不一定能拉全，故按运行时真实组合拉取。
# 默认引擎为 SenseVoiceSmall（多语种，含中英混合）。paraformer-zh 等可选模型
# 的 ModelScope id 可能随平台调整而失效，故不纳入强制下载，避免单点失败中止整体。
FUNASR_PIPELINES = [
    # (主 ASR 模型, 是否带说话人分离)
    ("iic/SenseVoiceSmall", True),
]


# whisper 仅拉取推理必需的权重文件（供应链安全：避免仓库内被混入的其它文件进入本地）
_WHISPER_ALLOW_PATTERNS = [
    "model.bin",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
    "README.md",
    ".gitattributes",
]


@contextmanager
def _env_override(env: Dict[str, str]):
    """临时覆盖环境变量，退出时还原，避免污染全局进程环境。"""
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def prefetch(engine: str = "all", settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    if engine in ("all", "funasr"):
        from funasr import AutoModel

        for mid, with_spk in FUNASR_PIPELINES:
            try:
                print(f"[setup] 下载 FunASR 模型管线: {mid} (+fsmn-vad +ct-punc"
                      f"{' +cam++' if with_spk else ''})")
                kwargs = dict(model=mid, vad_model="fsmn-vad", punc_model="ct-punc")
                if with_spk:
                    kwargs["spk_model"] = "cam++"
                # 不设置 disable_update，确保首次联网拉取到本地缓存
                AutoModel(**kwargs)
                print(f"[setup]   OK: {mid}")
            except Exception as e:  # 单模型失败不应中止整体 setup
                print(f"[warn] FunASR 模型 {mid} 下载失败（已跳过）: {e}")
    if engine in ("all", "whisper"):
        _download_whisper(settings)
    if engine in ("all", "brief"):
        _download_brief(settings)
    print("[setup] 完成：模型已缓存，可断网离线使用。缓存位于 ModelScope/HF 默认缓存目录。")


def _whisper_cached(local_dir: str) -> bool:
    """本地平铺目录是否已包含完整权重（model.bin 存在且体量正常）。"""
    mb = os.path.join(local_dir, "model.bin")
    return os.path.isfile(mb) and os.path.getsize(mb) > 1_000_000_000


def _download_whisper(settings: Settings) -> None:
    """下载 faster-whisper large-v3 到本地平铺目录（local_dir），绕开 HF 缓存快照链接。

    - 直接下载/复制到 ``whisper_local_dir(settings)`` 这个普通目录，引擎从该目录加载，
      避免本机 Windows 上 blobs->snapshots 硬链接失效导致 ctranslate2 找不到 model.bin。
    - 若本地目录已完整（model.bin 就绪），直接离线复用，不再触网。
    - HuggingFace 主站不可达时自动改走国内镜像 ``hf-mirror.com`` 并关闭 Xet
      （其 blob 走 CAS 存储，镜像不代理会 401）。
    """
    from huggingface_hub import snapshot_download
    from faster_whisper import WhisperModel

    repo = MODEL_PINS["whisper-large-v3"]["repo"]
    revision = MODEL_PINS["whisper-large-v3"]["revision"]
    local_dir = whisper_local_dir(settings)
    # 已完整缓存则离线复用，不触网（镜像偶发 502 也不影响）
    if _whisper_cached(local_dir):
        print(f"[setup] large-v3 已缓存于 {local_dir}，跳过下载（离线可用）")
        return
    # 供应链安全：钉死到可验证 commit（见 models/registry.py MODEL_PINS），
    # 并仅拉取推理必需的权重文件，避免仓库内被混入的其它文件进入本地。
    # 镜像环境变量用上下文管理器作用域限制，避免污染全局进程环境。
    attempts = [
        ("HuggingFace 主站", {}),
        ("国内镜像 hf-mirror.com", {
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HF_HUB_DISABLE_XET": "1",
            "HF_XET_DISABLED": "1",
        }),
    ]
    last_err: Optional[Exception] = None
    for label, env in attempts:
        try:
            with _env_override(env):
                print(f"[setup] 下载 faster-whisper large-v3 (CTranslate2) via {label} -> {local_dir}")
                snapshot_download(
                    repo,
                    revision=revision,
                    local_dir=local_dir,
                    allow_patterns=_WHISPER_ALLOW_PATTERNS,
                )
                # 就地离线验证可加载（不依赖 HF 缓存快照）
                os.environ["HF_HUB_OFFLINE"] = "1"
                WhisperModel(local_dir, device="cpu", compute_type="int8")
                print(f"[setup]   OK: large-v3 已缓存于 {local_dir}（revision={revision}）")
                return
        except Exception as e:  # 该源失败，尝试下一种
            last_err = e
            print(f"[warn] large-v3 经 {label} 下载/加载失败：{e}")
    if last_err:
        print(f"[warn] whisper large-v3 下载失败（已跳过，可稍后用 "
              f"`voice2txt setup --engine whisper` 重试）: {last_err}")


def ensure_models(engine: str = "all") -> None:
    """运行时若尚未下载则尝试下载（离线模式会失败并报错）。"""
    prefetch(engine)


def configure_cache(settings: Settings) -> None:
    """若用户指定了 models_dir，则把模型缓存根目录指向它。

    这样模型可以与程序一起打包到任意目录（U 盘/内网机），彻底离线可用。
    未指定时沿用 ModelScope/HF 默认缓存（~/.cache/...）。
    """
    if settings.models_dir:
        p = os.path.abspath(settings.models_dir)
        # ModelScope 缓存根
        os.environ["MODELSCOPE_CACHE"] = p
        # HuggingFace / transformers 缓存根（whisper 权重等）
        os.environ["HF_HOME"] = p
        print(f"[info] 使用本地模型目录: {p}")


def whisper_local_dir(settings: Settings) -> str:
    """whisper 权重的本地平铺目录。

    不走 HuggingFace 新缓存的 blobs->snapshots 硬链接机制（本机 Windows 上该链接
    会失效，导致 ctranslate2 在快照目录找不到 model.bin）。改为直接把模型下载/复制到
    这个普通目录，引擎从该目录加载。默认位于 ``~/.cache/huggingface/whisper-large-v3``，
    受 ``models_dir`` 影响时落到其下。
    """
    base = settings.models_dir or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(base, "whisper-large-v3")


# ---- 纪要总结 GGUF 模型（默认 Qwen2.5-7B-Instruct Q4_K_M；可选 3B）----
def brief_local_path(settings: Settings) -> str:
    """纪要总结 GGUF 权重落盘路径。

    默认位于 ``~/.cache/voice2txt/brief/<file>``；指定 ``models_dir`` 时落到其下，
    与 ASR 模型一起可随程序打包到 U 盘/内网机，彻底离线可用。
    具体 file 由 settings.brief_model（默认 7b）经 brief_pin 决定。
    """
    pin = brief_pin(settings)
    fname = pin.get("file", "qwen2.5-7b-instruct-q4_k_m.gguf")
    base = settings.models_dir or os.path.join(os.path.expanduser("~"), ".cache", "voice2txt")
    return os.path.join(base, "brief", fname)


def _merge_parts(part_paths: List[str], target: str) -> None:
    """把按序排列的 GGUF 分片字节拼接为单个文件（等价于 gguf-split --merge）。

    仅做流式拷贝，避免把数 GB 模型读进内存；合并后校验总大小，防止半成品。
    """
    total = 0
    with open(target, "wb") as out:
        for p in part_paths:
            size = os.path.getsize(p)
            if size == 0:
                raise IOError(f"分片为空，合并中止：{p}")
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, length=16 * 1024 * 1024)
            total += size
    if os.path.getsize(target) != total:
        raise IOError(f"合并后大小不符：期望 {total}，实际 {os.path.getsize(target)}")


def _download_brief(settings: Settings) -> None:
    """下载 Qwen2.5 纪要 GGUF（默认 7B Q4_K_M；可经 settings.brief_model 选 3B）到本地平铺目录。

    - 仅拉取推理必需的权重文件（供应链安全：不拉取仓库内其它文件）。
    - 钉死到可验证 commit（见 models/registry.py MODEL_PINS）。
    - 7B 的 Q4_K_M 在上游以 2 个分片发布（见 pin 的 ``parts`` 字段）：本函数先把各分片
      下载到本地，再合并为单个 ``file``（llama-cpp-python 直接加载单文件最稳），合并后清理分片。
    - 3B 等为单文件（无 ``parts``），直接 hf_hub_download 到 ``file``。
    - HuggingFace 主站不可达时自动改走国内镜像 hf-mirror.com 并关闭 Xet。
    - 镜像环境变量用上下文管理器作用域限制，避免污染全局进程环境。
    """
    from huggingface_hub import hf_hub_download

    pin = brief_pin(settings)
    repo = pin["repo"]
    revision = pin["revision"]
    fname = pin["file"]
    parts = pin.get("parts")  # 分片列表（7B）；3B 等单文件无此键
    local_dir = os.path.dirname(brief_local_path(settings))
    os.makedirs(local_dir, exist_ok=True)
    target = os.path.join(local_dir, fname)

    # 离线复用：单文件（3B）需体量正常；分片情形需已完成合并（单文件在且分片已清理）。
    if os.path.isfile(target):
        if parts:
            if not any(os.path.isfile(os.path.join(local_dir, p)) for p in parts):
                print(f"[setup] 纪要模型已缓存于 {target}，跳过下载（离线可用）")
                return
            # 上次合并被中断（单文件与分片俱在）→ 丢弃半成品，重新下载/合并
            try:
                os.remove(target)
            except OSError:
                pass
        else:
            if os.path.getsize(target) > 1_000_000_000:
                print(f"[setup] 纪要模型已缓存于 {target}，跳过下载（离线可用）")
                return
            # 单文件过小视为损坏，重新下载

    attempts = [
        ("HuggingFace 主站", {}),
        ("国内镜像 hf-mirror.com", {
            "HF_ENDPOINT": "https://hf-mirror.com",
            "HF_HUB_DISABLE_XET": "1",
            "HF_XET_DISABLED": "1",
        }),
    ]
    last_err: Optional[Exception] = None

    if parts:
        # ---- 分片情形（7B）----
        part_paths = [os.path.join(local_dir, p) for p in parts]
        part_ready = lambda p: os.path.isfile(p) and os.path.getsize(p) > 10_000_000
        if all(part_ready(p) for p in part_paths):
            print(f"[setup] 分片已存在，跳过下载，直接合并：{fname}")
        else:
            for label, env in attempts:
                try:
                    with _env_override(env):
                        for p in parts:
                            dst = os.path.join(local_dir, p)
                            if part_ready(dst):
                                print(f"[setup] 分片已存在，复用：{p}")
                                continue
                            if os.path.exists(dst):
                                os.remove(dst)  # 清掉不完整分片，强制重拉
                            print(f"[setup] 下载纪要分片 {p} via {label} -> {local_dir}")
                            hf_hub_download(repo_id=repo, filename=p, revision=revision, local_dir=local_dir)
                    break  # 全部分片下载成功
                except Exception as e:  # 该源失败，尝试下一种
                    last_err = e
                    print(f"[warn] 纪要分片经 {label} 下载失败：{e}")
            else:
                print(f"[warn] 纪要模型下载失败（已跳过，可稍后用 "
                      f"`voice2txt setup --engine brief` 重试）: {last_err}")
                return
        # 合并分片 → 单文件，并校验
        _merge_parts(part_paths, target)
        # 合并成功后删除分片，释放空间（约半个模型体积）
        for p in part_paths:
            try:
                os.remove(p)
            except OSError:
                pass
        print(f"[setup]   OK: 纪要模型已合并缓存于 {target}（revision={revision}）")
        return

    # ---- 单文件情形（3B 等）----
    for label, env in attempts:
        try:
            with _env_override(env):
                print(f"[setup] 下载纪要模型 {fname} via {label} -> {local_dir}")
                hf_hub_download(
                    repo_id=repo,
                    filename=fname,
                    revision=revision,
                    local_dir=local_dir,
                )
            print(f"[setup]   OK: 纪要模型已缓存于 {target}（revision={revision}）")
            return
        except Exception as e:  # 该源失败，尝试下一种
            last_err = e
            print(f"[warn] 纪要模型经 {label} 下载失败：{e}")
    if last_err:
        print(f"[warn] 纪要模型下载失败（已跳过，可稍后用 "
              f"`voice2txt setup --engine brief` 重试）: {last_err}")
