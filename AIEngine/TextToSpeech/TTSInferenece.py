import io
import os

import numpy
import requests
import soundfile


class TTSInference:
    text_to_speech_url = "http://localhost:5070/text_to_speech"
    text_to_speech_set_gpt_weights = text_to_speech_url + '/set_gpt_weights'
    text_to_speech_set_sovits_weights = text_to_speech_url + '/set_sovits_weights'
    tts_prompt_path = os.path.join(r"D:\programs\aris_chatbot\aris_chatbot", r'assets\tts_prompt')
    gpt_model_path = os.path.join("GPT_weights_v2ProPlus", "Azusa-e15.ckpt")
    sovits_model_path = os.path.join("SoVITS_weights_v2ProPlus", "Azusa_e25_s800.pth")
    requests.get(text_to_speech_set_gpt_weights + f"?weights_path={gpt_model_path}")
    requests.get(text_to_speech_set_sovits_weights + f"?weights_path={sovits_model_path}")

    def __init__(self):
        pass

    @classmethod
    def api_inference(cls, text) -> tuple[numpy.ndarray, int]:
        """
        :return: tuple of (audio_data, sample_rate)
        """
        try:
            post_json = {'text': text, "text_lang": "zh",
                         "ref_audio_path": os.path.join(cls.tts_prompt_path, "azusa/Azusa_Peaceful.wav"),
                         "text_split_method": "cut5", "prompt_text": "",
                         "prompt_lang": "all_ja"}
            tts_response = requests.post(cls.text_to_speech_url, json=post_json).content
            return soundfile.read(io.BytesIO(tts_response))
        except Exception as exception:
            print(exception)
            exit(0)
