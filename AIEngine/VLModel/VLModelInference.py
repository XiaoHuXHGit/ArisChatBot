"""
还有很多待开发的功能：
1. 增加对视频的支持
2. 增加识别网络路径的支持
3. 和Yolo一样，增加对目标检测的支持
"""
import json
import base64

import cv2
import numpy
import torch
from transformers import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class VLModelConfig:
    # ==================== 量化设置 ====================
    bnb_4bit_quant_type = "nf4"
    # ==================== 推演设置 ====================
    do_sample = True
    top_k = 50
    top_p = 0.8
    temperature = 0.8


class VLModel:
    def __init__(self, model_path):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 启用 4-bit 量化
            bnb_4bit_quant_type=VLModelConfig.bnb_4bit_quant_type,  # 量化类型，推荐 'nf4' (Normal Float 4-bit)
            bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时使用的数据类型，推荐 bfloat16 或 float16
            bnb_4bit_use_double_quant=True,  # 可选：启用双量化，进一步减少内存 (默认 False)
        )

        # default: Load the models on the available device(s)
        # models = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_path, torch_dtype="auto", device_map="cuda:0"
        # )
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            # attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )

        # default processor
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

        # The default range for the number of visual tokens per image in the models is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    def _resize_image_if_needed(self, image_path, max_width=1280, max_height=720):
        """
        使用 OpenCV 读取图片，如果图片尺寸超过指定的最大宽高，则按比例缩小。
        优先尝试使用 cv2.imdecode 读取（兼容中文路径），若失败则抛出异常。

        Args:
            image_path (str): 图片文件的路径。
            max_width (int): 目标最大宽度 (默认 1280)。
            max_height (int): 目标最大高度 (默认 720)。

        Returns:
            numpy.ndarray: 调整尺寸后的 OpenCV 图像 (BGR 格式)。
        """
        img = None
        # 优先使用 imdecode，兼容中文路径和特殊字符
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            image_array = numpy.frombuffer(image_data, dtype=numpy.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"无法读取图片 '{image_path}': {e}")
        if img is None:
            raise ValueError(f"无法解码图片 '{image_path}'，可能文件已损坏或格式不受支持。")
        # 获取原始尺寸
        original_height, original_width = img.shape[:2]
        # 检查是否需要调整尺寸
        if original_width <= max_width and original_height <= max_height:
            return img
        # 计算缩放比例
        scale_width = max_width / original_width
        scale_height = max_height / original_height
        scale = min(scale_width, scale_height)
        # 新尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        # 缩放图像
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_img

    def _image_to_base64_data_uri_from_cv2_img(self, cv2_img):
        success, buffer = cv2.imencode('.jpg', cv2_img)
        if not success:
            raise ValueError("无法将图像编码为 JPEG 格式")
        base64_data = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"

    def model_infer(self, media_path, mode):
        resized_cv2_image = self._resize_image_if_needed(media_path, max_width=960, max_height=540)
        image_data_uri = self._image_to_base64_data_uri_from_cv2_img(resized_cv2_image)
        with open(role_path, 'r', encoding="utf-8") as model_config:
            vl_model_role = json.load(model_config)
        LLMPrompt = vl_model_role[mode]
        messages = [
            {"role": "system", "content": LLMPrompt["system"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_data_uri,
                    },
                    {"type": "text", "text": LLMPrompt["user"]},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=VLModelConfig.top_p,
                                            top_k=VLModelConfig.top_k, temperature=VLModelConfig.temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


if __name__ == "__main__":
    model_path = r"E:\Artificial-Intelligence\AI-large-language-model\模型和权重\Qwen\Qwen2.5-VL-7B-Instruct"
    role_path = "../../configs/roles/vl_model/vl_roles.json"

    qwen_vl_model = VLModel(model_path)
    vl_model_role = "image_description"
    while True:
        image_path_input = input("图片的路径：").strip("\"")
        output_text = qwen_vl_model.model_infer(image_path_input, mode=vl_model_role)
        print(output_text)
