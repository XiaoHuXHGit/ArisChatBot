# config.py
import json
from pathlib import Path
from dataclasses import dataclass, fields

CONFIG_DIR = Path(__file__).parent / "configs"


@dataclass
class GlobalConfigs:
    """
    全局配置。
    log_level: 日志级别
        可选值：DEBUG, INFO
    """
    log_level: str


@dataclass
class MainWindowConfigs:
    """
    主窗口配置。
    window_scale: 窗口缩放比例
        值域：0 ~ 1
    """
    window_scale: float


@dataclass
class VLConfigs:
    model_path: str
    device: str


@dataclass
class ASRConfigs:
    """
    ASR模型配置。
    model_path: 模型路径
    device: 运行设备
        可选值：cpu, cuda, auto
    """
    model_path: str
    device: str
    vad_model_path: str


@dataclass
class LLMConfigs:
    model_path: str
    device: str

    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    length_penalty: float
    no_repeat_ngram_size: int
    bad_words_ids: list
    num_return_sequences: int


@dataclass
class WebsocketConfigs:
    """
    Websocket配置。
    host: 主机名
    port: 端口号
    """
    server_host: str
    server_port: int


class ConfigManager:
    # 显式映射：配置类 ↔ JSON 文件名
    global_config: GlobalConfigs
    main_window_config: MainWindowConfigs
    asr_config: ASRConfigs
    websocket_config: WebsocketConfigs

    _CONFIG_MAP = {
        'global_config': ("global_configs.json", GlobalConfigs),
        'main_window_config': ("main_window.json", MainWindowConfigs),
        # 'vl_config': ("vl_config.json", VLConfigs),
        'asr_config': ("asr_model.json", ASRConfigs),
        'websocket_config': ("websockets.json", WebsocketConfigs)
    }

    def __init__(self, config_load: None | str = None):
        if config_load is None:
            self._load_all()
        else:
            self._load_single(config_load)


    def _load_json(self, filename: str) -> dict:
        path = CONFIG_DIR / filename
        if not path.exists():
            print(f"[Config] Warning: {path} not found, using defaults.")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_all(self):
        for attr_name, (filename, cls) in self._CONFIG_MAP.items():
            data = self._load_json(filename)
            # print(f"[DEBUG] Raw data from {filename}: {data}")
            field_names = {f.name for f in fields(cls)}
            valid_data = {k: v for k, v in data.items() if k in field_names}
            # print(f"[DEBUG] Valid data for {cls.__name__}: {valid_data}")
            setattr(self, attr_name, cls(**valid_data))

    def _load_single(self, attr_name: str):
        filename, cls = self._CONFIG_MAP[attr_name]
        data = self._load_json(filename)
        field_names = {f.name for f in fields(cls)}
        valid_data = {k: v for k, v in data.items() if k in field_names}
        setattr(self, attr_name, cls(**valid_data))


    def handle_update(self) -> bool:
        """重新加载所有配置。返回是否成功。"""
        try:
            self._load_all()
            print("[Config] Reloaded successfully.")
            return True
        except Exception as e:
            print(f"[Config] Reload failed: {e}")
            return False

