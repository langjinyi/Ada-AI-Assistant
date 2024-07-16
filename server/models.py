# _*_ coding: utf-8 _*_
# @Time    : 2024/7/16 12:51
# @Author  : JINYI LIANG

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import ChatTTS
from configs.model_config import MODEL_PATH, V2T_MODELS, T2V_MODELS

class Models:
    def __init__(self):
        self.v2t_processor = WhisperProcessor.from_pretrained(MODEL_PATH["v2t_model"][V2T_MODELS])
        self.v2t_model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH["v2t_model"][V2T_MODELS]).to("cuda")
        self.v2t_model.config.forced_decoder_ids = None

        self.tts = ChatTTS.Chat()
        self.tts.load(device="cuda", source='custom', custom_path=MODEL_PATH["t2v_model"][T2V_MODELS], compile=False)

models = Models()