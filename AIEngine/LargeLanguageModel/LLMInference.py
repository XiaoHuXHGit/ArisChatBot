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
    """推理接口基类"""

    @abstractmethod
    def generate(self, prompt: list, max_tokens: int = 100, do_sample: bool = False,
                 top_k: int = 20, top_p: float = 0.7, temperature: float = 0.7,
                 think: bool = True) -> str:
        pass

    @abstractmethod
    def load_to_cpu(self):
        pass

    @abstractmethod
    def load_to_gpu(self, device_num: int):
        pass

    @abstractmethod
    def model_del(self):
        pass


class TransformersInference(BaseInference):
    """Transformers推理实现"""

    def __init__(self, model_path, n_ctx: int = 2048):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.tokenizer = None
        self.model = None
        self.current_device = None

    def _load_model(self):
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
        if self.model is None or self.tokenizer is None:
            self._load_model()

        prompt_text = prompt if isinstance(prompt, list) else str(prompt)

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

            if not think:
                generate_kwargs['enable_thinking'] = False

            outputs = self.model.generate(**generate_kwargs)

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt_text):]

    def load_to_cpu(self):
        if self.model is None:
            self._load_model()
        self.model.to('cpu')
        self.current_device = 'cpu'

    def load_to_gpu(self, device_num: int):
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
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.current_device = None


class LlamaCppInference(BaseInference):
    """Llama-cpp-python推理实现"""

    def __init__(self, model_path, n_ctx: int = 2048):
        if not os.path.isfile(model_path) or not model_path.lower().endswith('.gguf'):
            raise ValueError(f"Model path must be a .gguf file. Got: {model_path}")
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.model = None
        self.current_device = None
        self.n_gpu_layers = 0
        self.verbose = False

    def _load_model(self):
        self.model = Llama(
            model_path=self.model_path,
            chat_format="chatml",
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose
        )
        self.current_device = 'cpu' if self.n_gpu_layers == 0 else 'gpu'

    def generate(self, prompt: list, max_tokens: int = 100, do_sample: bool = False,
                 top_k: int = 40, top_p: float = 0.95, temperature: float = 0.2,
                 think: bool = True) -> str:
        if self.model is None:
            self._load_model()

        prompt_content = prompt if isinstance(prompt, list) else str(prompt)

        if not think:
            prompt_content[-1]["content"] = prompt_content[-1].get("content", "") + "/no_think"

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
        self.n_gpu_layers = -1
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
        if self.model is not None:
            self.model = None
            gc.collect()
        self.current_device = None


class LLMInference:
    """LLM推理控制器，自动选择后端"""

    def __init__(self, model_path: str, n_ctx: int = 2048):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.backend = self._select_backend()

    def _select_backend(self) -> BaseInference:
        if os.path.isfile(self.model_path) and self.model_path.lower().endswith('.gguf'):
            return LlamaCppInference(self.model_path, self.n_ctx)
        elif os.path.isdir(self.model_path):
            return TransformersInference(self.model_path, self.n_ctx)
        else:
            raise ValueError(f"Model path '{self.model_path}' is neither a .gguf file nor a directory.")

    def generate(self, prompt: list, max_tokens: int = 100, do_sample: bool = False,
                 top_k: int = 20, top_p: float = 0.7, temperature: float = 0.7,
                 think: bool = True) -> str:
        response = self.backend.generate(
            prompt, max_tokens, do_sample, top_k, top_p, temperature, think
        )
        clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        clean_response = re.sub(r'\n\s*\n', '\n', clean_response).strip()
        return clean_response

    def load_to_cpu(self):
        self.backend.load_to_cpu()

    def load_to_gpu(self, device_num: int):
        self.backend.load_to_gpu(device_num)

    def model_del(self):
        self.backend.model_del()


if __name__ == '__main__':
    model_path = r"E:\Artificial-Intelligence\AI-large-language-model\模型和权重\Qwen\Qwen3-32B-Uncensored-gguf\Qwen3-32B-Uncensored.i1-IQ3_XXS.gguf"
    from DataAccessObject.DatabaseOperate.LLMRoleRepository import LLMRoleRepository
    with LLMRoleRepository() as repo:
        system_prompt = repo.get_role("爱丽丝")
    # n_ctx = system_prompt_tokens + user_input_tokens(常值100) + history_prompt_len(常值512) + ai_gen_max_tokens
    llm_for_tokenizer = Llama(
        model_path=model_path,  # 仍然是你的模型文件
        n_ctx=1,  # 设置一个最小的上下文长度，因为我们只用分词器
        n_gpu_layers=0,  # 关键：不加载任何模型层到内存，仅加载分词器
        verbose=False  # 可选：减少输出
    )
    # 获取分词器实例
    tokenizer = llm_for_tokenizer.tokenizer()
    # tokenize 方法接收字节串
    system_tokens = tokenizer.tokenize(
        system_prompt.encode("utf-8"),
        add_bos=False,  # 通常计算系统提示本身长度时不需要 BOS
    )
    system_prompt_tokens = len(system_tokens)
    user_input_max = 100
    del llm_for_tokenizer
    llm = LLMInference(model_path, n_ctx=system_prompt_tokens + 512 + user_input_max)
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
