"""
存在bug：识别出来中文没标点
修复方法：微调优化或者直接换模型
"""
import os
import pyaudio
import torch
import librosa
from typing import Optional, Generator
from numpy import ndarray
import numpy as np
import logging
from transformers.modeling_utils import SpecificPreTrainedModelType
from transformers.processing_utils import SpecificProcessorType
from silero_vad import load_silero_vad, get_speech_timestamps
from configs import ConfigManager


class ASRInferToolTorch:
    def __init__(
            self,
            model_path: Optional[str],
            device: Optional[str] = None,  # 修改默认值为None
            data_type: Optional[torch.dtype] = torch.float32,
    ):
        if model_path and not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model_path = model_path
        self.device = device
        self.data_type = data_type
        self.processor = None
        self.model = None

    def inference(self, audio: ndarray, language: Optional[str] = None, task: str = "transcribe") -> str:
        # 检测模型是否为None，如果是则默认加载到CPU
        if self.model is None or self.processor is None:
            logging.info("未选择初始化设备，默认将模型加载到CPU")
            self.load_to_cpu()

        # 使用模型当前的数据类型来处理输入特征
        input_features = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.model.dtype)  # 关键修改：使用模型的实际数据类型

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                task=task,
                language=language,
                attention_mask=torch.ones((1, input_features.shape[-1] // 2), device=self.device)
            )
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

    def model_clear(self) -> None:
        self.model: SpecificPreTrainedModelType | None = None
        self.processor: SpecificProcessorType | None = None
        if "cuda" in self.device:
            torch.cuda.empty_cache()

    def model_load(self) -> None:
        """原有方法，现在用于从硬盘加载模型"""
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.processor = WhisperProcessor.from_pretrained(self.model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.data_type,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device).eval()

    def load_to_cpu(self) -> None:
        """将模型加载到CPU，如果模型未加载则从硬盘加载，否则在内存和显存间调度"""
        device = "cpu"
        data_type = torch.float32  # CPU默认使用float32

        if self.model is None or self.processor is None:
            # 模型未加载，从硬盘加载
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self.processor = WhisperProcessor.from_pretrained(self.model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=data_type,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(device).eval()
        else:
            # 模型已加载，直接移动到CPU
            self.model = self.model.to(device).to(dtype=data_type)  # 确保数据类型正确
            if hasattr(self.processor, 'to'):
                self.processor = self.processor.to(device)

        self.device = device
        self.data_type = data_type

    def load_to_gpu(self, device_num: int = 0, if_half: bool = True) -> None:
        """将模型加载到GPU，如果模型未加载则从硬盘加载，否则在内存和显存间调度"""
        device = f"cuda:{device_num}" if torch.cuda.is_available() else "cuda"
        data_type = torch.float16 if if_half else torch.float32  # GPU可选择是否使用半精度

        if self.model is None or self.processor is None:
            # 模型未加载，从硬盘加载
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self.processor = WhisperProcessor.from_pretrained(self.model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=data_type,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(device).eval()
        else:
            # 模型已加载，直接移动到GPU并转换数据类型
            self.model = self.model.to(device).to(dtype=data_type)  # 确保数据类型正确
            if hasattr(self.processor, 'to'):
                self.processor = self.processor.to(device)

        self.device = device
        self.data_type = data_type


class ASRInferToolNPU:
    def __init__(self, model_path: Optional[str] = None):
        pass

    def inference(self, audio_path: str, language: Optional[str] = None, task: str = "transcribe") -> str:
        pass

    def realtime_inference(self):
        pass


class ASRInference:
    """
    Unified ASR inference interface.
    Automatically selects backend based on device.
    ==================== 中文说明 ====================
    统一的 ASR 推理接口。
    自动选择后端，根据设备。
    """

    def __init__(
            self,
            model_path: Optional[str],
            device: Optional[str] = None,
            vad_model_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.device = device
        self.data_type = torch.float32  # 初始化时默认使用float32

        self.backend = ASRInferToolTorch(model_path, device=device, data_type=self.data_type)

        # VAD 初始化 - 需要在模型加载后进行
        self.sample_rate = 16000
        if vad_model_path == "default":
            self.vad_model = load_silero_vad()
        else:
            if not vad_model_path:
                vad_model_path = os.path.join(os.path.dirname(__file__), "models", "vad_model")
                if not os.path.exists(vad_model_path):
                    os.makedirs(vad_model_path, exist_ok=True)
            self.vad_model = torch.package.PackageImporter(vad_model_path).load_pickle("silero_vad", "models")

        # VAD模型初始化时先设置为CPU，等选择设备后再移动
        self.vad_device = "cpu"  # 初始设置为CPU
        self.vad_model = self.vad_model.to(self.vad_device)

        # 参数设置
        self.min_speech_duration_ms = 800
        self.max_speech_duration_s = 15.0
        self.continuous_silence_threshold = 1.0
        self.min_voice_duration = 0.5
        self.energy_threshold = 0.01  # 调整能量阈值

    def load_to_cpu(self):
        """将ASR模型加载到CPU"""
        self.backend.load_to_cpu()
        self.device = "cpu"
        self.data_type = torch.float32
        # 同时移动VAD模型
        self.vad_model = self.vad_model.to("cpu")
        self.vad_device = "cpu"

    def load_to_gpu(self, device_num: int = 0, if_half: bool = True):
        """将ASR模型加载到GPU"""
        self.backend.load_to_gpu(device_num, if_half)
        device_str = f"cuda:{device_num}" if torch.cuda.is_available() else "cuda"
        self.device = device_str
        self.data_type = torch.float16 if if_half else torch.float32
        # 同时移动VAD模型
        self.vad_model = self.vad_model.to(device_str)
        self.vad_device = device_str

    def model_load(self):
        # 保留原有方法，但现在需要指定设备
        if self.device is None:
            logging.info("未选择初始化设备，默认将模型加载到CPU")
            self.load_to_cpu()
        else:
            if "cuda" in self.device:
                device_num = 0 if ":" not in self.device else int(self.device.split(":")[1])
                self.load_to_gpu(device_num, if_half=(self.data_type == torch.float16))
            else:
                self.load_to_cpu()

    def model_clear(self):
        self.backend.model_clear()
        self.device = None

    def local_file_inference(self, audio_path: str, language: Optional[str] = None, task: str = "transcribe") -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return self.backend.inference(audio, language=language, task=task)

    def _int16_to_float32(self, audio_int16: np.ndarray) -> np.ndarray:
        return audio_int16.astype(np.float32) / 32768.0

    def _is_silence(self, audio_chunk: np.ndarray, threshold: float = 0.01) -> bool:
        """检查音频块是否为静音（基于能量）"""
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        return energy < threshold

    def microphone_audio_stream(self) -> Generator[np.ndarray, None, None]:
        """
        简单但可靠的语音活动检测
        """
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=800  # 50ms块，提高响应速度
        )

        # 音频累积缓冲区
        audio_buffer = np.array([], dtype=np.float32)

        chunk_duration_sec = 0.05  # 50ms
        silence_counter = 0
        voice_detected = False
        max_silence_frames = int(self.continuous_silence_threshold / chunk_duration_sec)  # 转换为帧数

        try:
            while True:
                # 读取音频块
                chunk_bytes = stream.read(int(self.sample_rate * chunk_duration_sec), exception_on_overflow=False)
                chunk_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                chunk_float32 = self._int16_to_float32(chunk_int16)

                # 能量检测
                is_silence = self._is_silence(chunk_float32, self.energy_threshold)

                if not is_silence:
                    # 检测到声音，添加到缓冲区
                    audio_buffer = np.concatenate([audio_buffer, chunk_float32])
                    silence_counter = 0  # 重置静音计数器
                    voice_detected = True

                    # 限制缓冲区大小，防止内存溢出
                    max_buffer_size = int(self.sample_rate * self.max_speech_duration_s)
                    if len(audio_buffer) > max_buffer_size:
                        # 输出当前缓冲区
                        if len(audio_buffer) > self.sample_rate * self.min_voice_duration:
                            yield audio_buffer.copy()
                        # 重置
                        audio_buffer = np.array([], dtype=np.float32)
                        voice_detected = False
                else:
                    # 检测到静音
                    if voice_detected:
                        # 如果之前检测到语音，添加静音到缓冲区（用于自然过渡）
                        audio_buffer = np.concatenate([audio_buffer, chunk_float32])
                        silence_counter += 1

                        # 检查是否达到静音阈值
                        if silence_counter >= max_silence_frames:
                            # 输出语音段
                            if len(audio_buffer) > self.sample_rate * self.min_voice_duration:
                                # 移除末尾的静音部分（保留最后0.2秒静音用于平滑）
                                speech_end_idx = len(audio_buffer) - int(self.sample_rate * 0.2)
                                if speech_end_idx > 0:
                                    audio_buffer = audio_buffer[:speech_end_idx]
                                yield audio_buffer.copy()

                            # 重置
                            audio_buffer = np.array([], dtype=np.float32)
                            voice_detected = False
                            silence_counter = 0
                    else:
                        # 之前也没有语音，继续等待
                        continue

        except KeyboardInterrupt:
            # 程序中断时输出剩余音频
            if len(audio_buffer) > self.sample_rate * self.min_voice_duration:
                yield audio_buffer.copy()
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def microphone_inference(self, language: Optional[str] = None, task: str = "transcribe") -> Generator[
        str, None, None]:
        """
        实时麦克风转录：采集 → VAD → 推理 → yield 文本
        """
        if not hasattr(self.backend, 'inference'):
            raise NotImplementedError("Backend does not support raw audio transcription.")

        for audio_segment in self.microphone_audio_stream():
            if len(audio_segment) > 0:
                text = self.backend.inference(audio_segment, language=language, task=task)
                if text.strip():
                    yield text.strip()


if __name__ == "__main__":
    config = ConfigManager()
    # 初始化时不自动加载模型
    asr = ASRInference(model_path=config.asr_config.model_path, device=None,
                       vad_model_path=config.asr_config.vad_model_path)

    # 手动选择加载到GPU或CPU
    asr.load_to_gpu(device_num=0, if_half=True)  # 加载到GPU，使用半精度
    # asr.load_to_cpu()  # 加载到CPU

    for text in asr.microphone_inference(language="zh", task="transcribe"):
        print("🗣️:", text)
    del asr
    torch.cuda.empty_cache()