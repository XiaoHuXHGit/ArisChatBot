import os
import gc
import re
import logging
from abc import ABC, abstractmethod

from llama_cpp import Llama
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)


class BaseInference(ABC):
    """
    推理接口基类，定义了所有推理后端必须实现的抽象方法。
    
    该基类为不同的模型推理后端（如Transformers、Llama-cpp-python等）提供统一的接口规范，
    便于在不同后端之间进行切换和管理。所有子类必须实现generate、load_to_cpu、
    load_to_gpu和model_del四个核心方法。
    """

    @abstractmethod
    def generate(self, prompt: list, max_tokens: int = 100, do_sample: bool = False,
                 top_k: int = 20, top_p: float = 0.7, temperature: float = 0.7,
                 think: bool = True) -> str:
        """
        根据给定提示生成文本
        
        Args:
            prompt: 输入提示，可以是字符串列表或字符串
            max_tokens: 生成的最大token数量
            do_sample: 是否启用采样模式，False时使用贪心解码
            top_k: Top-k采样参数，限制候选词汇范围
            top_p: Top-p（核采样）参数，控制累积概率阈值
            temperature: 温度参数，控制生成的随机性
            think: 是否启用思维链模式，True时允许模型进行内部思考
        
        Returns:
            生成的文本内容
        """
        pass

    @abstractmethod
    def load_to_cpu(self):
        """
        将模型加载到CPU设备上
        
        该方法负责将模型参数和计算转移到CPU内存中，适用于资源受限或需要节省GPU显存的场景。
        """
        pass

    @abstractmethod
    def load_to_gpu(self, device_num: int):
        """
        将模型加载到指定GPU设备上
        
        Args:
            device_num: GPU设备编号，从0开始计数
        """
        pass

    @abstractmethod
    def model_del(self):
        """
        删除模型实例并释放相关内存
        
        该方法负责清理模型占用的内存资源，包括模型参数、优化器状态等，
        有助于防止内存泄漏和显存溢出问题。
        """
        pass


class TransformersInference(BaseInference):
    """
    Transformers推理实现类，基于Hugging Face Transformers库
    
    该类实现了Transformers框架的模型加载、推理和设备管理功能，
    支持CPU和GPU之间的动态切换，适用于各种基于Transformers的模型。
    """

    def __init__(self, model_path, n_ctx: int = 2048):
        """
        初始化Transformers推理实例
        
        Args:
            model_path: 模型文件路径或模型标识符
            n_ctx: 模型上下文长度，默认2048个token
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.tokenizer = None  # 分词器实例
        self.model = None      # 模型实例
        self.current_device = None  # 当前设备状态

    def _load_model(self):
        """
        内部方法：加载模型和分词器
        
        该方法从指定路径加载预训练模型和分词器，并将模型初始化到CPU上。
        使用float16精度以节省内存，启用低CPU内存使用模式。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.to('cpu')
        self.current_device = 'cpu'
        self.model.eval()

    def generate(self, prompt: list, max_tokens: int = 100, do_sample: bool = False,
                 top_k: int = 20, top_p: float = 0.7, temperature: float = 0.7,
                 think: bool = True) -> str:
        """
        使用Transformers模型生成文本
        
        Args:
            prompt: 输入提示，支持字符串列表或单个字符串
            max_tokens: 最大生成token数
            do_sample: 是否启用采样
            top_k: Top-k采样参数
            top_p: Top-p采样参数
            temperature: 温度参数
            think: 是否启用思维链
        
        Returns:
            生成的文本内容（去除原始提示部分）
        """
        if self.model is None or self.tokenizer is None:
            self._load_model()

        prompt_text = prompt if isinstance(prompt, str) else str(prompt)

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                'max_new_tokens': max_tokens,
                'do_sample': do_sample,
                'top_k': top_k,
                'top_p': top_p,
                'temperature': temperature,
                'pad_token_id': self.tokenizer.eos_token_id
            }

            outputs = self.model.generate(**generate_kwargs)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt_text):]

    def load_to_cpu(self):
        """
        将模型加载到CPU上
        
        如果模型尚未加载，则先加载模型。将模型参数转移到CPU内存中，
        并更新当前设备状态记录。
        """
        if self.model is None:
            self._load_model()
        self.model.to('cpu')
        self.current_device = 'cpu'

    def load_to_gpu(self, device_num: int):
        """
        将模型加载到指定GPU上
        
        Args:
            device_num: GPU设备编号
        
        如果CUDA不可用或指定GPU编号超出范围，则不执行操作。
        """
        if self.model is None:
            self._load_model()
        device_str = f'cuda:{device_num}'
        if not torch.cuda.is_available():
            return
        if device_num >= torch.cuda.device_count():
            return
        self.model.to(device_str)
        self.current_device = device_str

    def model_del(self):
        """
        删除模型实例并释放内存
        
        删除模型和分词器实例，执行垃圾回收和显存清理操作。
        """
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.current_device = None


class LlamaCppInference(BaseInference):
    """
    Llama-cpp-python推理实现类，基于llama.cpp优化推理
    
    该类实现了llama.cpp后端的模型加载和推理功能，支持GGUF格式模型文件，
    提供了高效的CPU推理和GPU加速能力。
    """

    def __init__(
            self, model_path: str,
            n_ctx: int = 2048,
            auto_n_ctx: bool = True,
            extra_n_ctx: int = 0,
            system_prompt: str | None = None
    ):
        """
        初始化Llama-cpp-python推理实例
        
        Args:
            model_path: GGUF格式模型文件路径
            n_ctx: 模型上下文长度
            auto_n_ctx: 是否自动计算上下文长度
            extra_n_ctx: 额外上下文长度
            system_prompt: 系统提示词，用于上下文长度计算
        
        Raises:
            ValueError: 当模型路径不是有效的GGUF文件时抛出异常
        """
        if not os.path.isfile(model_path) or not model_path.lower().endswith('.gguf'):
            raise ValueError(f"Model path must be a .gguf file. Got: {model_path}")
        self.model_path: str = model_path
        self.model: Llama | None = None
        self.current_device: str | None = None
        self.n_gpu_layers: int = 0  # GPU加速层数，0表示仅CPU
        self.verbose: bool = False  # 详细输出模式
        if system_prompt is None:
            logging.debug("系统角色为空自动设置为空字符串")
            self.system_prompt = ""
        else:
            self.system_prompt = system_prompt
        self.static_n_ctx: int = n_ctx
        self.auto_n_ctx: bool = auto_n_ctx
        self.extra_n_ctx: int = extra_n_ctx
        self.n_ctx: int = self._number_context_calc()

    def _load_model(self):
        """
        内部方法：加载Llama模型

        根据配置参数加载GGUF模型文件，设置聊天格式为chatml，
        计算实际上下文长度并初始化模型实例。
        """
        self.model = Llama(
            model_path=self.model_path,
            chat_format="chatml",
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose
        )
        self.current_device = 'cpu' if self.n_gpu_layers == 0 else 'gpu'
    
    def _number_context_calc(self) -> int:
        """
        内部方法：自动计算上下文长度
        
        根据系统提示词长度和额外长度计算最优上下文长度，
        以节省内存并提高推理效率。
        
        Returns:
            计算得出的上下文长度
        """
        if self.auto_n_ctx:
            llm_for_tokenizer = Llama(
                model_path=self.model_path,  # 使用self.model_path
                n_ctx=1,  # 设置最小上下文长度，仅用于加载分词器
                n_gpu_layers=0,  # 不加载模型层到内存，仅加载分词器
                verbose=False
            )
            # 获取分词器实例
            tokenizer = llm_for_tokenizer.tokenizer()
            # 使用实例变量system_prompt
            system_tokens = tokenizer.tokenize(
                self.system_prompt.encode("utf-8"),
                add_bos=False,  # 计算系统提示长度时不需要BOS
            )
            system_prompt_tokens = len(system_tokens)
            del llm_for_tokenizer
            return_number_context = system_prompt_tokens + self.extra_n_ctx
            return return_number_context
        else:
            return self.n_ctx

    def generate(self, prompt: list, max_tokens: int = 100, do_sample: bool = False,
                 top_k: int = 40, top_p: float = 0.95, temperature: float = 0.2,
                 think: bool = True) -> str:
        """
        使用Llama-cpp-python生成文本
        
        Args:
            prompt: 聊天消息列表，包含role和content字段
            max_tokens: 最大生成token数
            do_sample: 是否启用采样
            top_k: Top-k采样参数
            top_p: Top-p采样参数
            temperature: 温度参数
            think: 是否启用思维链模式
        
        Returns:
            生成的AI回复内容
        """
        if self.model is None:
            self._load_model()

        prompt_content = prompt if isinstance(prompt, list) else prompt

        if not think:
            # 修正：检查列表是否为空，避免索引错误
            if prompt_content and isinstance(prompt_content, list) and len(prompt_content) > 0:
                last_message = prompt_content[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    last_message["content"] = last_message["content"] + "/no_think"

        output = self.model.create_chat_completion(
            messages=prompt_content,
            max_tokens=max_tokens,
            seed=-1,
            temperature=temperature if do_sample else 0.2,
            top_k=top_k if do_sample else 40,
            top_p=top_p if do_sample else 0.95
        )
        return output["choices"][0]["message"]["content"]

    def load_to_cpu(self):
        """
        将模型切换到CPU模式

        重新加载模型实例，设置GPU层为0，实现CPU推理。
        保留原模型实例以防加载失败时回滚。
        """
        self.n_gpu_layers = 0
        old_model = self.model
        try:
            self.model = Llama(
                model_path=self.model_path,
                chat_format="chatml",
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose
            )
            self.current_device = 'cpu'
        except Exception as e:
            self.model = old_model
        finally:
            if old_model is not None and old_model != self.model:
                del old_model
                gc.collect()

    def load_to_gpu(self, device_num: int):
        """
        将模型切换到GPU模式

        重新加载模型实例，设置GPU层为-1（全部层），实现GPU加速推理。
        保留原模型实例以防加载失败时回滚。

        Args:
            device_num: GPU设备编号（当前实现中未使用，全部层都加载到GPU）
        """
        self.n_gpu_layers = -1  # -1表示所有层都加载到GPU
        old_model = self.model
        try:
            self.model = Llama(
                model_path=self.model_path,
                chat_format="chatml",
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose
            )
            self.current_device = 'gpu'
        except Exception as e:
            self.model = old_model
        finally:
            if old_model is not None and old_model != self.model:
                del old_model
                gc.collect()

    def model_del(self):
        """
        删除模型实例并释放内存
        
        删除Llama模型实例，执行垃圾回收，清除当前设备状态。
        """
        if self.model is not None:
            self.model = None
            gc.collect()
        self.current_device = None


class LLMInference:
    """
    LLM推理控制器，自动选择合适的推理后端
    
    该类根据模型文件类型自动选择Transformers或Llama-cpp-python后端，
    提供统一的推理接口，简化了不同后端之间的切换和管理。
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, auto_n_ctx: bool = True, system_prompt: str | None = None):
        """
        初始化LLM推理控制器
        
        Args:
            model_path: 模型路径，支持Transformers模型目录或GGUF文件
            n_ctx: 模型上下文长度
            auto_n_ctx: 是否自动计算上下文长度
            system_prompt: 系统提示词（用于上下文长度计算）
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.auto_n_ctx = auto_n_ctx
        self.system_prompt = system_prompt
        self.backend = self._select_backend()

    def _select_backend(self) -> BaseInference:
        """
        内部方法：根据模型路径自动选择推理后端

        检查模型路径类型，如果是GGUF文件则选择LlamaCppInference，
        如果是目录则选择TransformersInference。

        Returns:
            适配的推理后端实例

        Raises:
            ValueError: 当模型路径格式不支持时抛出异常
        """
        if os.path.isfile(self.model_path) and self.model_path.lower().endswith('.gguf'):
            return LlamaCppInference(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                auto_n_ctx=self.auto_n_ctx,
                extra_n_ctx=self.n_ctx,
                system_prompt=self.system_prompt
            )
        elif os.path.isdir(self.model_path):
            return TransformersInference(self.model_path, self.n_ctx)
        else:
            raise ValueError(f"模型路径 '{self.model_path}' 既不是 .gguf 文件也不是一个 文件夹.")
            # raise ValueError(f"Model path '{self.model_path}' is neither a .gguf file nor a directory.")

    def generate(self, prompt: list, max_tokens: int = 100, do_sample: bool = False,
                 top_k: int = 20, top_p: float = 0.7, temperature: float = 0.7,
                 think: bool = True) -> str:
        """
        生成文本内容
        
        通过选择的后端生成文本，并对输出进行后处理，移除思维链标记和多余空白。
        
        Args:
            prompt: 输入提示消息列表
            max_tokens: 最大生成token数
            do_sample: 是否启用采样
            top_k: Top-k采样参数
            top_p: Top-p采样参数
            temperature: 温度参数
            think: 是否启用思维链
        
        Returns:
            清理后的生成文本
        """
        response = self.backend.generate(
            prompt, max_tokens, do_sample, top_k, top_p, temperature, think
        )
        # 移除思维链标记
        clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        # 清理多余空白行
        clean_response = re.sub(r'\n\s*\n', '\n', clean_response).strip()
        return clean_response

    def load_to_cpu(self):
        """
        将后端模型加载到CPU
        
        调用当前后端的CPU加载方法，将模型转移到CPU设备上。
        """
        self.backend.load_to_cpu()

    def load_to_gpu(self, device_num: int):
        """
        将后端模型加载到GPU
        
        调用当前后端的GPU加载方法，将模型转移到指定GPU设备上。
        
        Args:
            device_num: GPU设备编号
        """
        self.backend.load_to_gpu(device_num)

    def model_del(self):
        """
        删除后端模型实例
        
        调用当前后端的模型删除方法，释放相关内存资源。
        """
        self.backend.model_del()


if __name__ == '__main__':
    model_path = r"E:\Artificial-Intelligence\AI-large-language-model\模型和权重\Qwen\Qwen3-32B-Uncensored-gguf\Qwen3-32B-Uncensored.i1-IQ3_XXS.gguf"
    from DataAccessObject.DatabaseOperate.LLMRoleRepository import LLMRoleRepository
    with LLMRoleRepository() as repo:
        system_prompt = repo.get_role("爱丽丝")
    # n_ctx = system_prompt_tokens + user_input_tokens(常值100) + history_prompt_len(常值512) + ai_gen_max_tokens
    llm = LLMInference(model_path, n_ctx=612, auto_n_ctx=True, system_prompt=system_prompt)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "爱丽丝，你的胸摸起来软软的（用脸蹭蹭），爱丽丝身上的味道也好好闻呀（嗅嗅）"}
    ]

    if torch.cuda.is_available():
        llm.load_to_gpu(0)
        response = llm.generate(messages, max_tokens=512, do_sample=False, top_k=80, top_p=0.7, temperature=0.7, think=False)
        print("AI Response:", response)
    else:
        print("CUDA not available, skipping GPU test.")

    llm.model_del()
