import os

import pyaudio
import torch
import librosa
from typing import Optional, Generator

from numpy import ndarray
import numpy as np
from transformers.modeling_utils import SpecificPreTrainedModelType
from transformers.processing_utils import SpecificProcessorType
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

from configs import ConfigManager


class ASRInferToolTorch:
    def __init__(
            self,
            # processor: SpecificProcessorType,
            # models: SpecificPreTrainedModelType,
            model_path: Optional[str],
            device: Optional[str] = "cpu",
            data_type: Optional[torch.dtype] = torch.float32,
    ):
        # AI models settings
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model_path = model_path
        self.device = device
        self.data_type = data_type
        self.processor = None
        self.model = None
        self.model_load()

    def inference(self, audio: ndarray, language: Optional[str] = None, task: str = "transcribe") -> str:
        """
        转录音频。

        Args:
            audio: 音频文件二进制数据
            language: 语言代码（如 "chinese", "english"），None 表示自动检测
            task: "transcribe"（转录）或 "translate"（翻译成英文）
        """
        if self.model is None or self.processor is None:
            self.model_load()
        input_features = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=self.data_type)

        # 设置生成参数
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
        """
        清除模型的显存。
        """
        self.model: SpecificPreTrainedModelType | None = None
        self.processor: SpecificProcessorType | None = None
        if "cuda" in self.device:
            torch.cuda.empty_cache()

    def model_load(self) -> None:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.processor = WhisperProcessor.from_pretrained(self.model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.data_type,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device).eval()

    def model_offload_to_cpu(self, device: str = "cpu") -> None:
        self.model = self.model.to(device)
        self.processor = self.processor.to(device)


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
            device: Optional[str] = "auto",
            vad_model_path: Optional[str] = None
    ):
        if device == "auto":
            # from openvino import Core
            # core = Core()
            # if "NPU" in core.available_devices:
            #     device = "npu"
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            # del core

        self.model_path = model_path
        self.device = device
        self.data_type = torch.float16 if self.device != "cpu" else torch.float32

        if device == "cuda" or device == "cpu":
            self.backend = ASRInferToolTorch(model_path, device=device, data_type=self.data_type)
        # elif device == "npu":
        #     self.backend = ASRInferToolNPU(model_path=model_path)
        else:
            raise ValueError(f"Unsupported device: {device}")

        # microphone settings
        # === VAD 初始化（使用 silero-vad）===
        self.sample_rate = 16000
        if vad_model_path == "default":
            self.vad_model = load_silero_vad()
        else:
            if not vad_model_path:
                vad_model_path = os.path.join(os.path.dirname(__file__), "models", "vad_model")
                if not os.path.exists(vad_model_path):
                    os.makedirs(vad_model_path, exist_ok=True)
            self.vad_model = torch.package.PackageImporter(vad_model_path).load_pickle("silero_vad", "models")
        self.vad_device = self.device
        self.vad_model.to(self.vad_device)

        # VAD 参数（可调）
        self.speech_pad_ms = 300  # 语音前后填充（毫秒）
        self.min_speech_duration_ms = 300
        self.max_speech_duration_s = 10.0
        self.min_silence_duration_ms = 1000  # 判定语音结束的静音时长

    def model_load(self):
        self.backend.model_load()

    def model_clear(self):
        self.backend.model_clear()

    def local_file_inference(self, audio_path: str, language: Optional[str] = None, task: str = "transcribe") -> str:
        """
        Transcribe audio file to text.
        :param audio_path: Path to audio file (supports WAV, MP3, etc. via librosa)
        :param language:
        :param task:
        :return: Transcribed text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return self.backend.inference(audio, language=language, task=task)

    def _int16_to_float32(self, audio_int16: np.ndarray) -> np.ndarray:
        return audio_int16.astype(np.float32) / 32768.0

    def microphone_audio_stream(self) -> Generator[np.ndarray, None, None]:
        """
        使用 silero-vad 实现更精准的语音活动检测。
        Yield: 有效语音片段（16kHz, float32, mono）
        """
        p = pyaudio.PyAudio()
        # 注意：silero-vad 推荐 16kHz，我们直接采 16kHz
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=4800  # 300ms chunk（可调）
        )

        audio_buffer = np.array([], dtype=np.float32)
        chunk_duration_sec = 0.3  # 300ms

        try:
            while True:
                # 读取一块音频
                chunk_bytes = stream.read(int(self.sample_rate * chunk_duration_sec), exception_on_overflow=False)
                chunk_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                chunk_float32 = self._int16_to_float32(chunk_int16)

                # 累积到缓冲区
                audio_buffer = np.concatenate([audio_buffer, chunk_float32])

                # 如果缓冲区太长（比如 > 15 秒），强制截断避免内存爆炸
                if len(audio_buffer) > self.sample_rate * 15:
                    audio_buffer = audio_buffer[-int(self.sample_rate * 10):]

                # 转为 torch tensor 并移到设备
                wav = torch.from_numpy(audio_buffer).to(self.vad_device)

                # 运行 VAD 检测
                speech_timestamps = get_speech_timestamps(
                    wav,
                    self.vad_model,
                    sampling_rate=self.sample_rate,
                    min_speech_duration_ms=self.min_speech_duration_ms,
                    min_silence_duration_ms=self.min_silence_duration_ms,
                    speech_pad_ms=self.speech_pad_ms,
                    return_seconds=False
                )

                # 如果检测到完整语音段（且是新段）
                if speech_timestamps:
                    # 取最后一个完整语音段（假设用户正在说话）
                    last_seg = speech_timestamps[-1]
                    start, end = last_seg['start'], last_seg['end']

                    # 避免重复输出同一段
                    if end <= len(audio_buffer):
                        speech_segment = audio_buffer[start:end].copy()
                        yield speech_segment

                        # 清空已处理部分（保留一点尾部以防切割）
                        audio_buffer = audio_buffer[end - int(self.sample_rate * 0.1):]

        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def microphone_inference(self, language: Optional[str] = None, task: str = "transcribe") -> Generator[str, None, None]:
        """
        实时麦克风转录：采集 → VAD → 推理 → yield 文本
        """
        if not hasattr(self.backend, 'inference'):
            raise NotImplementedError("Backend does not support raw audio transcription.")

        for audio_segment in self.microphone_audio_stream():
            text = self.backend.inference(audio_segment, language=language, task=task)
            yield text.strip()


# Example usage (uncomment for testing)
if __name__ == "__main__":
    config = ConfigManager()
    asr = ASRInference(model_path=config.asr_config.model_path, device=config.asr_config.device,
                       vad_model_path=config.asr_config.vad_model_path)
    while True:
        audio_path = input("path: ").strip("\"")
        if audio_path == "exit":
            break
        elif audio_path == "clear":
            asr.model_clear()
            continue
        elif audio_path == "load":
            asr.model_load()
            continue
        import time

        start = time.time()
        text = asr.local_file_inference(audio_path, language="chinese")
        end = time.time()
        print("Time:", end - start)
        print("Transcription:", text)
    del asr
    torch.cuda.empty_cache()

    # asr = ASRInference(model_path=config.asr_config.model_path, device="cuda",
    #                    vad_model_path=config.asr_config.vad_model_path)
    # for text in asr.microphone_inference(language="chinese"):
    #     print("🗣️:", text)
    # del asr
    # torch.cuda.empty_cache()
