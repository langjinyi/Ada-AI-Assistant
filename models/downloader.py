# _*_ coding: utf-8 _*_
# @Time    : 2024/7/5 13:53
# @Author  : JINYI LIANG

from modelscope import snapshot_download
llm_model_path="Qwen/Qwen-7B-Chat"
llm_cache_path="/root/autodl-tmp/Ada-AI-Assistant/models/Qwen-7B-Chat"

emb_model_path="AI-ModelScope/bge-large-zh-v1.5"
emb_cache_path="/root/autodl-tmp/Ada-AI-Assistant/models/BAAI/bge-large-zh-v1.5"

voice_model_path="AI-ModelScope/fish-speech-1.2"
voice_cache_path="/root/autodl-tmp/Ada-AI-Assistant/models/fish-speech-1.2"

voice2text_model_path="mwei23/whisper-large-v2"
voice2text_cache_path="/root/autodl-tmp/Ada-AI-Assistant/models/whisper-large-v2"

text2voice_model_path="pzc163/chatTTS"
text2voice_cache_path="/root/autodl-tmp/Ada-AI-Assistant/models/chatTTS"

# snapshot_download(llm_model_path, cache_dir=llm_cache_path)
# snapshot_download(emb_model_path, cache_dir=emb_cache_path)
snapshot_download(text2voice_model_path, cache_dir=text2voice_cache_path)