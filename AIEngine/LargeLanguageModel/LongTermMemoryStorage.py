from mem0 import Memory
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime


class LongTermMemoryStorage:
    """
    基于Mem0框架的大模型长时记忆存储类
    用于管理用户对话历史、偏好和重要信息的向量化存储与检索
    支持本地模型部署
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化长时记忆存储

        Args:
            config: Mem0配置参数字典，可选
        """
        # 默认配置，针对本地模型优化
        default_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": "./qdrant_data",  # 本地Qdrant存储路径
                }
            },
            "llm": {
                "provider": "ollama",  # 使用Ollama本地模型
                "config": {
                    "model": "qwen3-32b-uncensored",  # 本地Qwen3模型
                    "temperature": 0.1,
                    "base_url": "http://localhost:11434"  # Ollama默认端口
                }
            },
            "embedder": {
                "provider": "huggingface",  # 本地HuggingFace嵌入模型
                "config": {
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                }
            }
        }

        # 如果提供了配置，则合并
        if config:
            self._merge_config(default_config, config)

        try:
            self.memory_client = Memory.from_config(default_config)
            self.logger = logging.getLogger(__name__)
            self.logger.info("LongTermMemoryStorage initialized successfully with local models")
        except Exception as e:
            self.logger.error(f"Failed to initialize LongTermMemoryStorage: {e}")
            raise

    def _merge_config(self, default: Dict, override: Dict):
        """
        合并默认配置和覆盖配置

        Args:
            default: 默认配置字典
            override: 覆盖配置字典
        """
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value

    def add_memory(self, user_input: str, user_id: str = "default_user",
                   additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        添加记忆到向量存储

        Args:
            user_input: 用户输入的文本
            user_id: 用户标识符
            additional_metadata: 额外的元数据信息

        Returns:
            包含添加结果的字典
        """
        try:
            # 构建记忆数据
            memory_data = {
                "input": user_input,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id
            }

            if additional_metadata:
                memory_data.update(additional_metadata)

            # 添加到Mem0记忆系统
            result = self.memory_client.add(
                user_input,
                user_id=user_id,
                metadata=additional_metadata or {}
            )

            self.logger.info(f"Memory added for user {user_id}: {user_input[:50]}...")
            return result

        except Exception as e:
            self.logger.error(f"Error adding memory: {e}")
            return {"error": str(e), "success": False}

    def search_memories(self, query: str, user_id: str = "default_user",
                        limit: int = 5) -> List[Dict[str, Any]]:
        """
        检索相关记忆

        Args:
            query: 查询文本
            user_id: 用户标识符
            limit: 返回结果数量限制

        Returns:
            相关记忆列表
        """
        try:
            # 从Mem0检索相关记忆
            results = self.memory_client.search(
                query,
                user_id=user_id,
                limit=limit
            )

            self.logger.info(f"Found {len(results)} memories for user {user_id} with query: {query[:30]}...")
            return results

        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
            return []

    def get_all_memories(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """
        获取特定用户的所有记忆

        Args:
            user_id: 用户标识符

        Returns:
            用户所有记忆的列表
        """
        try:
            results = self.memory_client.get_all(user_id=user_id)
            self.logger.info(f"Retrieved {len(results)} memories for user {user_id}")
            return results

        except Exception as e:
            self.logger.error(f"Error getting all memories: {e}")
            return []

    def update_memory(self, memory_id: str, new_content: str,
                      user_id: str = "default_user") -> Dict[str, Any]:
        """
        更新现有记忆

        Args:
            memory_id: 要更新的记忆ID
            new_content: 新的记忆内容
            user_id: 用户标识符

        Returns:
            更新结果字典
        """
        try:
            result = self.memory_client.update(
                new_content,
                memory_id=memory_id,
                user_id=user_id
            )
            self.logger.info(f"Memory {memory_id} updated for user {user_id}")
            return result

        except Exception as e:
            self.logger.error(f"Error updating memory {memory_id}: {e}")
            return {"error": str(e), "success": False}

    def delete_memory(self, memory_id: str, user_id: str = "default_user") -> Dict[str, Any]:
        """
        删除特定记忆

        Args:
            memory_id: 要删除的记忆ID
            user_id: 用户标识符

        Returns:
            删除结果字典
        """
        try:
            result = self.memory_client.delete(memory_id=memory_id, user_id=user_id)
            self.logger.info(f"Memory {memory_id} deleted for user {user_id}")
            return result

        except Exception as e:
            self.logger.error(f"Error deleting memory {memory_id}: {e}")
            return {"error": str(e), "success": False}

    def clear_user_memories(self, user_id: str) -> Dict[str, Any]:
        """
        清除特定用户的所有记忆

        Args:
            user_id: 用户标识符

        Returns:
            清除结果字典
        """
        try:
            result = self.memory_client.delete_all(user_id=user_id)
            self.logger.info(f"All memories cleared for user {user_id}")
            return result

        except Exception as e:
            self.logger.error(f"Error clearing memories for user {user_id}: {e}")
            return {"error": str(e), "success": False}

    def get_relevant_memories(self, query: str, user_id: str = "default_user",
                              limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取与查询最相关的记忆

        Args:
            query: 查询文本
            user_id: 用户标识符
            limit: 返回结果数量限制

        Returns:
            相关记忆列表
        """
        try:
            results = self.memory_client.get(
                query=query,
                user_id=user_id,
                limit=limit
            )
            return results

        except Exception as e:
            self.logger.error(f"Error getting relevant memories: {e}")
            return []

    def add_multiple_memories(self, inputs: List[str], user_id: str = "default_user") -> List[Dict[str, Any]]:
        """
        批量添加多个记忆

        Args:
            inputs: 输入文本列表
            user_id: 用户标识符

        Returns:
            添加结果列表
        """
        results = []
        for input_text in inputs:
            result = self.add_memory(input_text, user_id)
            results.append(result)

        self.logger.info(f"Added {len(inputs)} memories in batch for user {user_id}")
        return results

    def close(self):
        """
        关闭记忆存储连接，释放资源
        """
        try:
            # Mem0通常不需要显式关闭，但可以添加清理逻辑
            self.logger.info("LongTermMemoryStorage closed")
        except Exception as e:
            self.logger.error(f"Error closing LongTermMemoryStorage: {e}")


# 使用示例
if __name__ == "__main__":
    # 配置本地模型参数
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": "./qdrant_data",  # 本地存储路径
            }
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "qwen3-32b-uncensored",  # 本地Qwen3模型
                "temperature": 0.1,
                "base_url": "http://localhost:11434"
            }
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            }
        }
    }

    # 创建记忆存储实例
    memory_storage = LongTermMemoryStorage(config=config)

    # 添加记忆
    memory_storage.add_memory("用户喜欢科幻小说和编程", user_id="user001")
    memory_storage.add_memory("用户最近在学习Python机器学习", user_id="user001")

    # 检索记忆
    results = memory_storage.search_memories("用户兴趣爱好", user_id="user001")
    print("检索结果:", results)

    # 获取所有记忆
    all_memories = memory_storage.get_all_memories(user_id="user001")
    print("所有记忆:", len(all_memories))
