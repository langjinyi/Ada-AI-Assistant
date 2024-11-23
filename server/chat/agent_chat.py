# _*_ coding: utf-8 _*_
# @Time    : 2024/7/18 11:08
# @Author  : JINYI LIANG

import asyncio
import io
import json
import string

import edge_tts
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import List
from ast import literal_eval

from server.agent import model_container
from server.agent.callbacks import CustomAsyncIteratorCallbackHandler
from server.agent.custom import CustomPromptTemplate, CustomOutputParser
from server.agent.tools_select import tools, tool_names
from server.utils import get_model_worker_config, fschat_openai_api_address, remove_special_characters, \
    write_wav_to_buffer, convert_to_int16, convert_numbers_to_chinese
from configs import LLM_MODELS, TEMPERATURE, PROMPT_TEMPLATES
import torch
import librosa
import ChatTTS
from server.models import models
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import LLMSingleActionAgent, AgentExecutor
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler


router = APIRouter()


@router.post("/agent_chat",
             tags=["Agent Chat"],
             summary="与llm模型语音Agent对话(通过LLMChain)")
async def agent_chat(audio_file: UploadFile = File(None, description="用户输入语音文件"),
                     text_query: str = Form(None, description="用户输入文字", examples=["你好"]),
                     history: List[str] = Form([], description="用户聊天记录"),
                     K: int = Form(3, description="保留K轮历史"),
                     stream: bool = Form(False, description="流式输出"),
                     model_name: str = Form(LLM_MODELS, description="LLM 模型名称。"),
                     temperature: float = Form(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
                     max_tokens: int = Form(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                     prompt_name: str = Form("default", description="使用的prompt模板名称"),
                     speed: str = Form("[speed_2]", description="语速"),
                     seed: int = Form(1506, description="音色"),
                     tts_type: str = Form("tts_infer", description="TTS引擎类型")):

    callback = CustomAsyncIteratorCallbackHandler()
    # callback = FinalStreamingStdOutCallbackHandler()
    query = text_query

    if audio_file is not None:
        audio_bytes = await audio_file.read()
        audio_io = io.BytesIO(audio_bytes)

        async def process_audio_and_transcribe(audio_io):
            audio, sampling_rate = librosa.load(audio_io, sr=16000)
            input_features = models.v2t_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(
                "cuda")
            prompt_ids = models.v2t_processor.get_prompt_ids("以下是普通话的句子:", return_tensors="pt").to("cuda")

            predicted_ids = models.v2t_model.generate(input_features=input_features, prompt_ids=prompt_ids)
            transcription = models.v2t_processor.batch_decode(predicted_ids, skip_special_tokens=True)
            return transcription[0][len(prompt_ids) - 1:]

        audio_transcription_task = asyncio.create_task(process_audio_and_transcribe(audio_io))
        query = await audio_transcription_task

    config = get_model_worker_config(model_name)
    model = ChatOpenAI(
        streaming=True,
        callbacks=[callback],
        verbose=True,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_proxy=config.get("openai_proxy"),
    )

    model_container.MODEL = model

    converted_output = ""
    if len(history) >= 2:
        history = [literal_eval(item) for item in history]
        converted_examples = [f'{his["role"]}：{his["content"]}' for his in history]
        converted_output = "\n".join(converted_examples[-K * 2:])

    prompt_template = PROMPT_TEMPLATES['girl_friend(agent_chat)']
    prompt_template_agent = CustomPromptTemplate(
        template=prompt_template,
        tools=tools,
        input_variables=["input", "intermediate_steps", "history"]
    )
    output_parser = CustomOutputParser()
    chain = LLMChain(prompt=prompt_template_agent, llm=model, verbose=True)

    agent = LLMSingleActionAgent(
        llm_chain=chain,
        output_parser=output_parser,
        stop=["\nObservation:", "Observation"],
        allowed_tools=tool_names,
        max_iterations=2
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                        tools=tools,
                                                        verbose=True,
                                                        )

    # Create a task for LLMChain query processing
    chain_task = asyncio.create_task(agent_executor.acall(inputs={"input": query, "history": converted_output},
                                                          callbacks=[callback]))

    torch.manual_seed(seed)
    rand_spk = models.tts.sample_random_speaker()

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        prompt=speed,
        spk_emb=rand_spk,
        temperature=.3,
    )

    async def tts_infer(text_):
        return models.tts.infer(text_, stream=stream, params_infer_code=params_infer_code,
                                do_text_normalization=True, use_decoder=True)
    async def edge_tts_infer(text_, voice="zh-CN-XiaoxiaoNeural"):
        communicate = edge_tts.Communicate(text_, voice)
        async for chunked in communicate.stream():
            if chunked["type"] == "audio":
                yield chunked["data"]
    async def generate_response():
        if stream:
            sentence = ""
            async for token in callback.aiter():
                sentence += token
                # Prepare the response data
                response_data = {
                    "input": query,
                    "text": token,
                }
                yield json.dumps(response_data)
            # Remove special characters from the final sentence
            clean_sentence = remove_special_characters(sentence)
            chinese_sentence = convert_numbers_to_chinese(clean_sentence)
            if tts_type == "tts_infer":
                tts_task = asyncio.create_task(tts_infer(chinese_sentence))
                audio_chunks = await tts_task
                for chunk in audio_chunks:
                    audio_data = convert_to_int16(np.array(chunk, dtype=np.float32))
                    buffer = write_wav_to_buffer(audio_data, rate=24000)
                    # Prepare the response data
                    response_data = {
                        "input": query,
                        "audio": buffer.read().decode('latin1')  # Encode binary data as latin1
                    }
                    yield json.dumps(response_data)
            elif tts_type == "edge_tts_infer":
                async for chunk in edge_tts_infer(chinese_sentence):
                    # Prepare the response data
                    response_data = {
                        "input": query,
                        "audio": chunk.decode('latin1')  # Encode binary data as latin1
                    }
                    yield json.dumps(response_data)
        else:
            sentence = ""
            # Process all tokens first
            async for token in callback.aiter():
                sentence += token

            # Remove special characters from the final sentence
            clean_sentence = remove_special_characters(sentence)
            chinese_sentence = convert_numbers_to_chinese(clean_sentence)

            if tts_type == "tts_infer":
                # Perform TTS inference on the cleaned sentence
                tts_task = asyncio.create_task(tts_infer(chinese_sentence))
                audio_chunk = await tts_task

                # Convert the audio chunk to int16 format
                audio_data = convert_to_int16(np.array(audio_chunk, dtype=np.float32))

                # Write the audio data to a buffer
                buffer = write_wav_to_buffer(audio_data, rate=24000)

                # Prepare the response data
                response_data = {
                    "input": query,
                    "text": sentence,
                    "audio": buffer.read().decode('latin1')  # Encode binary data as latin1
                }

                # Yield the response data as JSON
                yield json.dumps(response_data)
            elif tts_type == "edge_tts_infer":
                # Prepare the response data
                response_data = {
                    "input": query,
                    "text": sentence,
                }
                # Yield the response data as JSON
                yield json.dumps(response_data)
                async for chunk in edge_tts_infer(chinese_sentence):
                    # Prepare the response data
                    response_data = {
                        "input": query,
                        "audio": chunk.decode('latin1')  # Encode binary data as latin1
                    }
                    yield json.dumps(response_data)

        await chain_task
    return StreamingResponse(generate_response(), media_type="application/json")
