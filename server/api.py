# _*_ coding: utf-8 _*_
# @Time    : 2024/7/6 21:02
# @Author  : JINYI LIANG
import argparse

import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import uvicorn

from configs import NLTK_DATA_PATH
from configs.server_config import OPEN_CROSS_DOMAIN
from server.chat import chat, agent_chat
from server.utils import BaseResponse
from server.transform import v2t, t2v

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


async def document():
    return RedirectResponse(url="/docs")


def create_app(run_mode: str = None):
    app = FastAPI(
        title="Ada API Server",
    )
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(chat.router, prefix="/chat")
    app.include_router(agent_chat.router, prefix="/chat")
    app.include_router(v2t.router, prefix="/transform")
    app.include_router(t2v.router, prefix="/transform")

    @app.get("/", response_model=BaseResponse, summary="swagger 文档")
    async def document():
        return RedirectResponse(url="/docs")

    return app


def run_api(host, port, **kwargs):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='',
                                     description='')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)

    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)

    app = create_app()

    run_api(host=args.host,
            port=args.port,
            )
