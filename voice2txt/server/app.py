"""内网转录服务（预留，P7 阶段实现）。FastAPI + 作业队列。

启用：pip install voice2txt[serve] 后 `voice2txt serve`
核心引擎层复用，便于单机 -> 内网服务器演进。
"""
from __future__ import annotations


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    try:
        import uvicorn
        from fastapi import FastAPI, File, UploadFile
    except ImportError:
        print("请先安装 serve 扩展：pip install voice2txt[serve]")
        return

    app = FastAPI(title="voice2txt server", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        # TODO(P7): 落盘 -> 调用 build_engine -> 返回 SRT/JSON
        # 安全注意（实现时必须满足）：
        #  1) 校验 Content-Type / 文件头，仅接受音频；限制单文件大小，防 DoS；
        #  2) 用 tempfile 落盘（勿用客户端文件名，防路径遍历/覆盖）；
        #  3) 服务端需鉴权（token/内网白名单），勿对公网开放；CORS 显式限定来源；
        #  4) 解码走 PyAV，注意恶意构造媒体可能触发 libavformat 漏洞，建议在隔离/沙箱内执行。
        return {"status": "not_implemented", "filename": file.filename}

    print(f"[serve] 监听 http://{host}:{port} （当前为预留骨架）")
    uvicorn.run(app, host=host, port=port)
