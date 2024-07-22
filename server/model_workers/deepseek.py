import httpx
import requests
from fastchat.conversation import Conversation

from server.model_workers.base import *
from fastchat import conversation as conv
import sys
from typing import List, Dict, Iterator, Literal, Any


class DeepSeekWorker(ApiModelWorker):

    def __init__(
            self,
            *,
            model_names: List[str] = ("deepseek-api",),
            controller_addr: str = None,
            worker_addr: str = None,
            version: Literal["deepseek-chat"] = "deepseek-chat",
            **kwargs,
    ):
        # print(model_names)
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 16384)
        super().__init__(**kwargs)
        self.version = version

    def do_chat(self, params: ApiChatParams) -> Iterator[Dict]:
        params.load_config(self.model_names[0])
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {params.api_key}"
        }
        data = {
            "model": params.version,
            "messages": params.messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "stream": False
        }
        # print("***********")
        # print(data)
        # print("***********")

        url = "https://api.deepseek.com/v1/chat/completions"
        with httpx.Client(headers=headers, timeout=60.0) as client:
            response = client.post(url, json=data)
            response.raise_for_status()
            chunk = response.json()
            # print(chunk)
            yield {"error_code": 0, "text": chunk["choices"][0]["message"]["content"]}

    def get_embeddings(self, params):
        print("embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="",
            messages=[],
            roles=["user", "assistant", "system"],
            sep="\n###",
            stop_str="###",
        )


if __name__ == "__main__":
    import uvicorn
    from fastchat.serve.model_worker import app

    worker = DeepSeekWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21001",
    )
    sys.modules["fastchat.serve.model_worker"].worker = worker
    uvicorn.run(app, port=21001)
