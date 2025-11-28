import sqlite3
import os
from typing import List, Tuple


class CharacterChatDataset:
    database_name = "CharacterProfile.sqlite"

    def __init__(self, character_name: str = "default"):
        if not character_name or not isinstance(character_name, str):
            raise ValueError("character_name must be a non-empty string.")
        # SQLite 表名不能包含特殊字符，简单过滤（实际可更严格）
        self.table_name = "".join(c for c in character_name if c.isalnum() or c in ("_", "-"))
        if not self.table_name:
            raise ValueError("character_name resulted in invalid table name.")

        self.conn = self._get_connection()
        self._initialize_table()

    def _get_connection(self):
        """获取数据库连接，如果数据库文件不存在则自动创建"""
        project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        database_dir = os.path.join(project_path, "DataAccessObject/DataStorage")
        os.makedirs(database_dir, exist_ok=True)  # 确保目录存在
        database_path = os.path.join(database_dir, self.database_name)
        return sqlite3.connect(database_path)

    def _initialize_table(self):
        """初始化表结构，如果表不存在则创建"""
        with self.conn:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS "{self.table_name}" (
                    id INTEGER PRIMARY KEY,
                    user_input TEXT NOT NULL,
                    expected_response TEXT NOT NULL
                )
            """)

    def add_qa(self, input_text: str, output_text: str):
        """添加新的问答对到数据库"""
        with self.conn:
            self.conn.execute(
                f'INSERT INTO "{self.table_name}" (user_input, expected_response) VALUES (?, ?)',
                (input_text, output_text)
            )

    def get_all_qa(self) -> List[Tuple[str, str]]:
        """获取所有问答对（用于 ScriptResponse 重建索引等）"""
        cursor = self.conn.cursor()
        cursor.execute(f'SELECT user_input, expected_response FROM "{self.table_name}"')
        return cursor.fetchall()

    def get_qa_count(self) -> int:
        """获取问答对数量"""
        cursor = self.conn.cursor()
        cursor.execute(f'SELECT COUNT(*) FROM "{self.table_name}"')
        return cursor.fetchone()[0]

    def check_connection(self):
        """检查数据库连接是否有效"""
        try:
            self.conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()