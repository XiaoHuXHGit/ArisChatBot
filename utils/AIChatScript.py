from typing import Tuple
import torch
import psutil


class AIChatScript:
    def __init__(self):
        self.cuda_total_memory_gb, self.cuda_available_percent, self.cpu_total_memory_gb, self.cpu_available_percent = self.get_cuda_and_cpu_memory_usage()

    @staticmethod
    def get_cuda_and_cpu_memory_usage() -> Tuple[float, float, float, float]:
        """
        获取CUDA和CPU的总内存及可用内存百分比
        返回: (cuda_total_memory_gb, cuda_available_percent, cpu_total_memory_gb, cpu_available_percent)
        """
        # 获取CPU总内存和可用内存百分比
        cpu_total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_available_percent = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

        # 获取CUDA总显存和可用显存百分比
        if torch.cuda.is_available():
            cuda_total_memory = torch.cuda.get_device_properties(0).total_memory
            cuda_total_memory_gb = cuda_total_memory / (1024 ** 3)
            cuda_memory_free = cuda_total_memory - torch.cuda.memory_allocated()
            cuda_available_percent = (cuda_memory_free / cuda_total_memory) * 100
        else:
            cuda_total_memory_gb = 0.0
            cuda_available_percent = 0.0

        return cuda_total_memory_gb, cuda_available_percent, cpu_total_memory_gb, cpu_available_percent

    def llm_memory_usage_calculator(self) -> float:
        """
        计算大语言模型模型的显存占用
        返回: total_memory_gb
        """
        pass

    def asr_memory_usage_calculator(self) -> float:
        """
        计算语音识别模型的显存占用
        返回: total_memory_gb
        """
        pass

    def tts_memory_usage_calculator(self) -> float:
        """
        计算文本转音频模型的显存占用
        返回: total_memory_gb
        """
        pass

    def auto_memory_allocation(self):
        """
        自动管理内存和显存的分配、
        """
        pass


if __name__ == '__main__':
    ai_chat_script = AIChatScript()
    print(ai_chat_script.get_cuda_and_cpu_memory_usage())
