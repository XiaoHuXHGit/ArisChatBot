import re

import torch

from configs import ConfigManager
from AIEngine.LargeLanguageModel.LLMInference import LLMInference
from AIEngine.AutomaticSpeechRecognition.ASRInference import ASRInference
from AIEngine.TextToSpeech.TTSInferenece import TTSInference
from DataAccessObject.DatabaseOperate.LLMRoleRepository import LLMRoleRepository

config = ConfigManager()
asr = ASRInference(model_path=config.asr_config.model_path, device="cuda",
                   vad_model_path=config.asr_config.vad_model_path)
model_path = (r"E:\Artificial-Intelligence\AI-large-language-model\模型和权重\gemma\gemma-3-12B-Uncensored-GGUF\gemma-3"
              r"-uncensored.i1-Q6_K.gguf")
with LLMRoleRepository() as repo:
    system_prompt = repo.get_role("白洲梓")
llm = LLMInference(model_path, n_ctx=1024, auto_n_ctx=True, system_prompt=system_prompt)
asr.load_to_gpu(device_num=0, if_half=True)
llm.load_to_cpu()
# TTS 模块

print("===== 模型加载完成 =====")
for text in asr.microphone_inference(language="zh", task="transcribe"):
    print("️检测到音频:", text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]
    asr.load_to_cpu()
    llm.load_to_gpu(device_num=0)
    response = llm.generate(
        messages, max_tokens=512, do_sample=False, top_k=80, top_p=0.7, temperature=0.7,
        think=True
    )
    print("AI回复:", response)
    pattern = r'^[^:]*:|(\([^)]*\))'
    tts_text = re.sub(pattern, '', response).strip()
    data, samplerate = TTSInference().api_inference(tts_text)
    import sounddevice as sd
    sd.play(data, samplerate)
    sd.wait()
    asr.load_to_gpu(device_num=0, if_half=True)
    llm.load_to_cpu()
del asr
llm.model_del()
torch.cuda.empty_cache()
