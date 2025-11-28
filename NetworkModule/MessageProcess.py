import os
import random
import re
import logging

from NetworkModule.DataStructure.ChatMessageDataStructer import GroupMessage, PrivateMessage
from DataAccessObject.DatabaseOperate.CharacterChatDataset import CharacterChatDataset
from NetworkModule.ScriptResponse import ScriptResponse
from NetworkModule.DataDumpInOneBotFormat import DataDumpInOneBotFormat, MessageElement


class MessageProcess:
    # 全局共享的响应引擎（按角色隔离）
    _responder = None

    def __init__(self, data: dict):
        self.raw_data = data
        self.return_message = None
        # 初始化响应引擎（单例模式，避免重复加载）
        if MessageProcess._responder is None:
            db = CharacterChatDataset(character_name="alice")  # 角色名固定为 "alice"
            MessageProcess._responder = ScriptResponse(db_interface=db, similarity_threshold=0.3)
        self.responder = MessageProcess._responder

    async def message_process(self, cse_data: PrivateMessage | GroupMessage) -> str | None:
        """处理消息内容，返回要回复的文本"""
        raw_text = cse_data.message.strip()

        # 检查是否为学习指令：/learn 输入 -> 输出
        learn_pattern = r'^/learn\s+(.+?)\s*->\s*(.+)$'
        match = re.match(learn_pattern, raw_text, re.DOTALL)
        if match:
            user_input = match.group(1).strip()
            bot_output = match.group(2).strip()
            if user_input and bot_output:
                self.responder.learn(user_input, bot_output)
                logging.info(f"Learned: '{user_input}' -> '{bot_output}'")
                return f"好的！我已经学会了：当你说「{user_input}」，我就回复「{bot_output}」～"
            else:
                return "指令格式错误！请使用：/learn [用户输入] -> [机器人回复]"

        # 正常对话模式
        if raw_text:
            response = self.responder.get_response(raw_text)
            return response
        return None

    async def group_message_process(self, cse_data: GroupMessage) -> list[dict] | None:
        """处理群消息，只要消息中包含“爱丽丝”就响应"""
        message = cse_data.message

        # 检查是否包含“爱丽丝”（中文字符）
        if "爱丽丝" in message:
            # 移除所有“爱丽丝”字样（可多次出现）
            cleaned_msg = message.replace("爱丽丝", "").strip()

            # 可选：进一步清理多余空格或标点（比如“，”、“：”等）
            # 例如：把“爱丽丝，你好” → “你好”
            cleaned_msg = re.sub(r'^[，。！？:\s]+', '', cleaned_msg)  # 去掉开头的标点/空格

            # 如果清理后为空，则不处理（避免纯“爱丽丝”触发无意义回复）
            if not cleaned_msg:
                return None

            # 临时替换消息内容用于处理
            original_message = cse_data.message
            cse_data.message = cleaned_msg

            reply_text = await self.message_process(cse_data)

            # 恢复原始消息（良好实践）
            cse_data.message = original_message

            if reply_text:
                reply_data = DataDumpInOneBotFormat(
                    user_id=cse_data.user_id,
                    group_id=cse_data.group_id,
                    message_type=DataDumpInOneBotFormat.GROUP_MESSAGE
                )
                return [
                    reply_data.process_message([
                        MessageElement(content=reply_text, content_type=DataDumpInOneBotFormat.TEXT_MESSAGE)
                    ])
                ]
        return None

    async def _parse_message(self, data: dict) -> list[dict] | None:
        """
        仅处理有效的 message 事件，过滤 API 响应、心跳等无关数据
        """
        # 排除 API 响应（包含 status/retcode/echo 等字段）
        if "status" in data or "retcode" in data or "echo" in data:
            logging.debug(f"忽略 API 响应或无效数据: {data.get('message', '')}")
            return None

        # 必须有 post_type 且为 "message"
        post_type = data.get("post_type")
        if post_type != "message":
            logging.debug(f"忽略除了 poke 外非 message 事件: post_type={post_type}")
            if data.get("sub_type") == "poke" and data.get("user_id") != data.get("self_id") and data.get("target_id") == data.get("self_id"):
                poke_reply = DataDumpInOneBotFormat(
                    user_id=data.get("user_id"),
                    group_id=data.get("group_id"),
                    message_type=DataDumpInOneBotFormat.GROUP_POKE_MESSAGE
                )
                image_reply = DataDumpInOneBotFormat(
                    user_id=data.get("user_id"),
                    group_id=data.get("group_id"),
                    message_type=DataDumpInOneBotFormat.GROUP_MESSAGE
                )
                image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "images", "poke_images", "alice")
                image_path = os.path.join(image_path, random.choice(os.listdir(image_path)))
                return [
                    poke_reply.process_message(),
                    image_reply.process_message([
                        MessageElement(content=image_path, content_type=DataDumpInOneBotFormat.IMAGE_MESSAGE)
                    ])
                ]
            return None

        # 必须有 message_type（group / private）
        message_type = data.get("message_type")
        if message_type not in ("group", "private"):
            logging.warning(f"未知 message_type: {message_type}")
            return None

        try:
            if message_type == "group":
                cse_data = GroupMessage.from_dict(data)
                return await self.group_message_process(cse_data)
            elif message_type == "private":
                cse_data = PrivateMessage.from_dict(data)
                reply_text = await self.message_process(cse_data)
                if reply_text:
                    reply_data = DataDumpInOneBotFormat(
                        user_id=cse_data.user_id,
                        message_type=DataDumpInOneBotFormat.PRIVATE_MESSAGE
                    )
                    return [
                        reply_data.process_message([
                            MessageElement(content=reply_text, content_type=DataDumpInOneBotFormat.TEXT_MESSAGE)
                        ])
                    ]
        except Exception as e:
            logging.error(f"处理消息时出错: {e}", exc_info=True)
        return None

    async def process(self) -> list[dict] | None:
        """主入口：解析并处理消息"""
        return await self._parse_message(self.raw_data)


if __name__ == "__main__":
    pass
