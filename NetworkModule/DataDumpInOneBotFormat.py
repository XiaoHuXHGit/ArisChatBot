import os
import re
import base64


class MessageElement:
    """
    表示单条消息元素的数据结构。
    """

    def __init__(self, content: str, content_type: int):
        """
        初始化消息元素。

        Args:
            content (str): 消息内容
            content_type (int): 消息内容类型，对应 DataDumpInOneBotFormat 的内容类型枚举
        """
        self.content = content
        self.content_type = content_type

    def get_content(self) -> tuple[str, int]:
        """
        获取消息内容和类型。

        Returns:
            tuple[str, int]: 消息内容和内容类型
        """
        return self.content, self.content_type


class DataDumpInOneBotFormat:
    """
    将内部消息数据结构转换为 OneBot 标准格式的数据转储类。
    该类负责将 GroupMessage 或 PrivateMessage 对象转换为符合 OneBot API 规范的消息格式，
    以便与 OneBot 协议兼容的聊天机器人框架进行交互。
    """

    # 消息类型枚举：区分私聊消息和群组消息
    PRIVATE_MESSAGE = 0
    GROUP_MESSAGE = 1
    FRIEND_POKE_MESSAGE = 2
    GROUP_POKE_MESSAGE = 3

    # 消息内容类型枚举：定义消息中可能包含的多媒体内容类型
    TEXT_MESSAGE = 0
    VOICE_MESSAGE = 1
    IMAGE_MESSAGE = 2
    VIDEO_MESSAGE = 3
    FILE_MESSAGE = 4
    AT_MESSAGE = 5
    FACE_MESSAGE = 6
    JSON_MESSAGE = 7
    POKE_MESSAGE = 8

    def __init__(self, user_id: str, group_id: str | None = None, message_type: int = 0):
        """
        初始化数据转储实例。

        Args:
            user_id (str | None): 用户 ID，不可为空
            group_id (str | None): 群组 ID，对于需要群组 ID 的消息类型(GROUP_MESSAGE, GROUP_POKE_MESSAGE)不可为空
            message_type (int): 消息类型标识，用于区分私聊消息和群组消息
        """
        self.user_id = user_id
        self.group_id = group_id
        self.message_type = message_type

    def process_message(self, message_elements: list[MessageElement] | None = None) -> dict:
        """
        处理消息数据，将其转换为 OneBot 标准格式并进行 JSON 序列化。

        Args:
            message_elements (list[MessageElement] | None): 消息元素列表，戳一戳消息可为 None

        Returns:
            str: JSON 序列化后的 OneBot 标准格式消息数据
        """
        # 验证必需的 ID
        self._validate_required_ids()

        # 对于戳一戳类型的消息，直接构建特殊格式
        if self.message_type == self.FRIEND_POKE_MESSAGE:
            return {
                "action": "friend_poke",
                "params": {
                    "user_id": self.user_id
                }
            }
        elif self.message_type == self.GROUP_POKE_MESSAGE:
            return {
                "action": "group_poke",
                "params": {
                    "group_id": self.group_id,
                    "user_id": self.user_id
                }
            }

        # 对于非戳一戳类型的消息，message_elements 不应为 None
        if message_elements is None:
            raise ValueError(f"非 poke 类型的 message_elements 不能为 None")

        # 构建 OneBot 格式的消息内容
        message_content = self._build_message_content(message_elements)
        # 根据消息类型构建 OneBot API 请求格式
        if self.message_type == self.GROUP_MESSAGE:
            # 群组消息格式
            onebot_format_data = {
                "action": "send_group_msg",
                "params": {
                    "group_id": self.group_id,
                    "message": message_content
                }
            }
        else:
            # 私聊消息格式
            onebot_format_data = {
                "action": "send_private_msg",
                "params": {
                    "user_id": self.user_id,
                    "message": message_content
                }
            }
        return onebot_format_data

    def _validate_required_ids(self):
        """
        验证必需的 ID 是否存在
        """
        if not self.user_id:
            raise ValueError(f"user_id 不可为空")
        elif self.message_type in [self.GROUP_MESSAGE, self.GROUP_POKE_MESSAGE]:
            # 群组消息和群组戳一戳消息需要 user_id 和 group_id
            if not self.group_id:
                raise ValueError(f"消息类型 {self.message_type} 需要提供 group_id")

    def _build_message_content(self, message_elements: list[MessageElement]) -> list[dict]:
        """
        根据消息元素列表构建 OneBot 格式的 message 内容列表。

        Args:
            message_elements (list[MessageElement]): 消息元素列表

        Returns:
            list[dict]: 符合 OneBot 格式的 message 内容列表
        """
        message_content = []

        for element in message_elements:
            content, content_type_enum = element.get_content()

            # 获取消息类型字符串
            content_type_str = self._get_content_type_string_by_enum(content_type_enum)

            # 获取数据类型字符串
            data_type_str = self._get_data_type_string_by_enum(content_type_enum)

            # 根据内容类型处理不同的格式检测
            if content_type_enum in [self.IMAGE_MESSAGE, self.VOICE_MESSAGE, self.VIDEO_MESSAGE]:
                data_value = self._detect_and_format_content(content, content_type_enum)
            else:
                data_value = content

            # 构建消息内容字典
            message_item = {
                "type": content_type_str,
                "data": {
                    data_type_str: data_value,
                    "summary": "(。>ᗜ<)_θ"
                }
            }

            message_content.append(message_item)

        return message_content

    def _detect_and_format_content(self, content: str, content_type_enum: int) -> str:
        """
        自动检测内容格式并返回合适的格式字符串。

        Args:
            content (str): 内容字符串
            content_type_enum (int): 内容类型枚举

        Returns:
            str: 格式化后的内容字符串
        """
        # 检测是否为网络地址
        if content.startswith("http"):
            return content

        # 检测是否为本地文件
        if os.path.exists(content):
            return f"file://{content}"

        # 对于语音和视频类型，如果网络和本地都检测失败，则报错
        if content_type_enum in [self.VOICE_MESSAGE, self.VIDEO_MESSAGE]:
            raise ValueError(f"语音或视频文件路径无效：{content}，既不是有效的网络地址也不是本地文件路径")

        # 对于图片类型，还需要检测是否为base64格式
        if content_type_enum == self.IMAGE_MESSAGE:
            # 检测是否为base64格式
            base64_pattern = r'^[A-Za-z0-9+/]*={0,2}$'
            if re.match(base64_pattern, content.replace('\n', '').replace('\r', '')) and len(content) > 0:
                # 验证base64是否有效
                try:
                    base64.b64decode(content, validate=True)
                    return f"base64://{content}"
                except Exception:
                    raise ValueError(f"图片内容格式无效：{content}，不是有效的base64编码")
            else:
                raise ValueError(f"图片内容格式无效：{content}，既不是有效的网络地址、本地文件路径也不是base64编码")

        return content

    def _get_content_type_string_by_enum(self, content_type_enum: int) -> str:
        """
        根据内容类型枚举值获取对应的内容类型字符串表示。

        Args:
            content_type_enum (int): 内容类型枚举值

        Returns:
            str: 内容类型的字符串表示
        """
        content_type_map = {
            self.TEXT_MESSAGE: "text",
            self.VOICE_MESSAGE: "record",
            self.IMAGE_MESSAGE: "image",
            self.VIDEO_MESSAGE: "video",
            self.FILE_MESSAGE: "file",
            self.AT_MESSAGE: "at",
            self.FACE_MESSAGE: "face",
            self.JSON_MESSAGE: "json",
            self.POKE_MESSAGE: "poke"
        }
        return content_type_map.get(content_type_enum, "text")

    def _get_data_type_string_by_enum(self, content_type_enum: int) -> str:
        """
        根据内容类型枚举值获取对应的数据类型字符串表示。

        Args:
            content_type_enum (int): 内容类型枚举值

        Returns:
            str: 数据类型的字符串表示
        """
        data_type_map = {
            self.TEXT_MESSAGE: "text",
            self.VOICE_MESSAGE: "file",
            self.IMAGE_MESSAGE: "file",
            self.VIDEO_MESSAGE: "file",
            self.FILE_MESSAGE: "file",
            self.AT_MESSAGE: "qq",
            self.FACE_MESSAGE: "id",
            self.JSON_MESSAGE: "data",
            self.POKE_MESSAGE: "id"
        }
        return data_type_map.get(content_type_enum, "text")