from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Literal, Union, cast, Optional
from uuid import UUID

from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.schema import AgentFinish, AgentAction

DEFAULT_ANSWER_PREFIX_TOKENS = ["Final Answer:"]

class CustomAsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """Callback handler that returns an async iterator."""

    queue: asyncio.Queue[str]

    done: asyncio.Event
    def append_to_last_tokens(self, token: str) -> None:
        self.last_tokens.append(token)


    def check_if_answer_reached(self) -> bool:
        # print(''.join(self.answer_prefix_tokens))
        # print(''.join(self.last_tokens))
        return  ''.join(self.answer_prefix_tokens_stripped) in ''.join(self.last_tokens)


    @property
    def always_verbose(self) -> bool:
        return True

    def __init__(
        self,
        *,
        answer_prefix_tokens: Optional[List[str]] = None,
    ) -> None:
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()
        if answer_prefix_tokens is None:
            self.answer_prefix_tokens = DEFAULT_ANSWER_PREFIX_TOKENS
        else:
            self.answer_prefix_tokens = answer_prefix_tokens

        self.answer_prefix_tokens_stripped = self.answer_prefix_tokens

        self.last_tokens = []
        self.answer_reached = False

    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID,
                                  parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                                  metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        pass

    async def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # If two calls are made in a row, this resets the state
        self.answer_reached = False
        self.done.clear()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token is not None and token != "":
            if not self.check_if_answer_reached():
                self.append_to_last_tokens(token)

            else:
                token = token.split("Final Answer:")[-1]
                self.queue.put_nowait(token)


    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        # self.queue.put_nowait("无法调用llm")
        self.done.set()

    async def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""


    async def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when tool ends running."""


    async def on_tool_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        self.queue.put_nowait(str(error))
        self.done.set()

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        # 返回最终答案
        # self.queue.put_nowait(finish.return_values["output"])
        self.done.set()

    async def aiter(self) -> AsyncIterator[str]:
        while not self.queue.empty() or not self.done.is_set():
            # Wait for the next token in the queue,
            # but stop waiting if the done event is set
            done, other = await asyncio.wait(
                [
                    # NOTE: If you add other tasks here, update the code below,
                    # which assumes each set has exactly one task each
                    asyncio.ensure_future(self.queue.get()),
                    asyncio.ensure_future(self.done.wait()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the other task
            if other:
                other.pop().cancel()

            # Extract the value of the first completed task
            token_or_done = cast(Union[str, Literal[True]], done.pop().result())

            # If the extracted value is the boolean True, the done event was set
            if token_or_done is True:
                break

            # Otherwise, the extracted value is a token, which we yield
            yield token_or_done
