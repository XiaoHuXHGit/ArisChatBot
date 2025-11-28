"""
data structure for private chat message
{
    "Message": {
        "self_id": "string",           // 发送者的机器人ID
        "user_id": "string",           // 消息发送者的ID
        "time": "string",              // 消息发送时间戳
        "message_id": "string",        // 消息ID
        "message_seq": "string",       // 消息序列号
        "real_id": "string",           // 真实消息ID（可能用于某些特殊情况）
        "message_type": "string",      // 消息类型，如"private"、"group"等
        "sender": {                    // 消息发送者信息
            "user_id": "string",       // 用户ID
            "nickname": "string",      // 用户昵称
            "card": "string"           // 用户名片信息（如群名片）
        },
        "raw_message": "string",       // 原始消息内容
        "font": "int",                 // 消息字体
        "sub_type": "string",          // 子类型，如"normal"、"anonymous"等
        "message": "string",           // 处理过的消息内容
        "message_format": "string",    // 消息格式
        "post_type": "string",         // 消息投递类型，如"message"、"notice"等
        "target_id": "string"          // 目标ID，根据消息类型的不同，可能是群ID、好友ID等
    }
}
"""


class Sender:
    def __init__(self, user_id: str = None, nickname: str = None, card: str = None):
        self.user_id = user_id
        self.nickname = nickname
        self.card = card


class Message:
    def __init__(self, self_id: str = None, user_id: str = None, time: str = None,
                 message_id: str = None, message_seq: str = None, real_id: str = None,
                 message_type: str = None, sender: Sender = None, raw_message: str = None,
                 font: int = None, sub_type: str = None, message: str = None,
                 message_format: str = None, post_type: str = None):
        self.self_id = self_id
        self.user_id = user_id
        self.time = time
        self.message_id = message_id
        self.message_seq = message_seq
        self.real_id = real_id
        self.message_type = message_type
        self.sender = sender
        self.raw_message = raw_message
        self.font = font
        self.sub_type = sub_type
        self.message = message
        self.message_format = message_format
        self.post_type = post_type

    @classmethod
    def from_dict(cls, data: dict):
        # 创建Sender实例
        sender_data = data.get("sender", {})
        sender = Sender(user_id=sender_data.get("user_id"),
                        nickname=sender_data.get("nickname"),
                        card=sender_data.get("card"))

        # 创建Message实例并赋值
        message = cls(
            self_id=data.get("self_id"),
            user_id=data.get("user_id"),
            time=data.get("time"),
            message_id=data.get("message_id"),
            message_seq=data.get("message_seq"),
            real_id=data.get("real_id"),
            message_type=data.get("message_type"),
            sender=sender,
            raw_message=data.get("raw_message"),
            font=data.get("font"),
            sub_type=data.get("sub_type"),
            message=data.get("message"),
            message_format=data.get("message_format"),
            post_type=data.get("post_type"),
        )
        return message


class PrivateMessage(Message):
    def __init__(self, self_id: str = None, user_id: str = None, time: str = None,
                 message_id: str = None, message_seq: str = None, real_id: str = None,
                 message_type: str = None, sender: Sender = None, raw_message: str = None,
                 font: int = None, sub_type: str = None, message: str = None,
                 message_format: str = None, post_type: str = None, target_id: str = None):
        super().__init__(self_id=self_id, user_id=user_id, time=time,
                         message_id=message_id, message_seq=message_seq, real_id=real_id,
                         message_type=message_type, sender=sender, raw_message=raw_message,
                         font=font, sub_type=sub_type, message=message,
                         message_format=message_format, post_type=post_type)
        self.target_id = target_id

    @classmethod
    def from_dict(cls, data: dict):
        # 调用父类的from_dict方法
        message = super().from_dict(data)
        # 处理子类特有的属性
        message.target_id = data.get("target_id")
        return message


class GroupMessage(Message):
    def __init__(self, self_id: str = None, user_id: str = None, time: str = None,
                 message_id: str = None, message_seq: str = None, real_id: str = None,
                 message_type: str = None, sender: Sender = None, raw_message: str = None,
                 font: int = None, sub_type: str = None, message: str = None,
                 message_format: str = None, post_type: str = None, group_id: str = None):
        super().__init__(self_id=self_id, user_id=user_id, time=time,
                         message_id=message_id, message_seq=message_seq, real_id=real_id,
                         message_type=message_type, sender=sender, raw_message=raw_message,
                         font=font, sub_type=sub_type, message=message,
                         message_format=message_format, post_type=post_type)
        self.group_id = group_id

    @classmethod
    def from_dict(cls, data: dict):
        # 调用父类的from_dict方法
        message = super().from_dict(data)
        # 处理子类特有的属性
        message.group_id = data.get("group_id")
        return message

