"""
TodayWife - 每日角色分配系统

此模块提供完整的每日角色分配管理解决方案，包括角色存储、用户数据管理和每日角色选择功能。
使用SQLite数据库存储角色信息和用户分配数据。

系统包含两个主要类：
- TodayWife: 供主程序使用的只读访问（仅查询操作）
- TodayWifeOperator: 供初始化和管理使用的完全访问（增删改查操作）

文件结构：
- TodayWifeStorage.sqlite: 存储角色数据（WifeStorage, HusbandStorage表）
- user_data.sqlite: 存储用户分配数据（UserTable表）

数据表：
- WifeStorage/HusbandStorage: id (主键, 自增), name (非空),
  work (非空), image_name (非空), description
- UserTable: id (主键, 自增), account (非空),
  character_type (非空), character_data (非空), date

使用方法：
    # 供主程序使用（只读操作）
    today_wife = TodayWife(db_path="TodayWifeStorage.sqlite")
    daily_character = today_wife.get_today_wife("user123")

    # 供初始化和管理使用
    operator = TodayWifeOperator()
    operator.add_wife("Kurumi", "Date A Live", "kurumi.png", "Time spirit")
    operator.set_today_wife("user123", "WifeStorage", 1)
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Tuple
import os

class TodayWife:
    """
    今日老婆数据访问类，仅提供查询权限
    """

    def __init__(self, db_path: str = "TodayWifeStorage.sqlite"):
        """
        初始化数据库连接

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")

    def _get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)

    def get_wife_by_id(self, wife_id: int) -> Optional[Tuple]:
        """
        根据ID获取老婆角色信息

        Args:
            wife_id: 角色ID

        Returns:
            角色信息元组 (id, name, work, image_name, description) 或 None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, work, image_name, description FROM WifeStorage WHERE id = ?",
                (wife_id,)
            )
            result = cursor.fetchone()
            return result

    def get_husband_by_id(self, husband_id: int) -> Optional[Tuple]:
        """
        根据ID获取老公角色信息

        Args:
            husband_id: 角色ID

        Returns:
            角色信息元组 (id, name, work, image_name, description) 或 None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, work, image_name, description FROM HusbandStorage WHERE id = ?",
                (husband_id,)
            )
            result = cursor.fetchone()
            return result

    def get_random_wife(self) -> Optional[Tuple]:
        """
        随机获取一个老婆角色（使用SQL内置随机函数）

        Returns:
            角色信息元组 (id, name, work, image_name, description) 或 None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, work, image_name, description FROM WifeStorage ORDER BY RANDOM() LIMIT 1"
            )
            result = cursor.fetchone()
            return result

    def get_random_husband(self) -> Optional[Tuple]:
        """
        随机获取一个老公角色（使用SQL内置随机函数）

        Returns:
            角色信息元组 (id, name, work, image_name, description) 或 None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, work, image_name, description FROM HusbandStorage ORDER BY RANDOM() LIMIT 1"
            )
            result = cursor.fetchone()
            return result

    def get_today_wife(self, account: str) -> Optional[Tuple]:
        """
        获取用户今日角色

        Args:
            account: 用户账号

        Returns:
            今日角色信息元组或 None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT t.id, t.account, t.character_type, t.character_data, t.date,
                          CASE 
                              WHEN t.character_type = 'WifeStorage' THEN w.name
                              WHEN t.character_type = 'HusbandStorage' THEN h.name
                          END as name,
                          CASE 
                              WHEN t.character_type = 'WifeStorage' THEN w.work
                              WHEN t.character_type = 'HusbandStorage' THEN h.work
                          END as work,
                          CASE 
                              WHEN t.character_type = 'WifeStorage' THEN w.image_name
                              WHEN t.character_type = 'HusbandStorage' THEN h.image_name
                          END as image_name,
                          CASE 
                              WHEN t.character_type = 'WifeStorage' THEN w.description
                              WHEN t.character_type = 'HusbandStorage' THEN h.description
                          END as description
                   FROM UserTable t
                   LEFT JOIN WifeStorage w ON (t.character_type = 'WifeStorage' AND t.character_data = w.id)
                   LEFT JOIN HusbandStorage h ON (t.character_type = 'HusbandStorage' AND t.character_data = h.id)
                   WHERE t.account = ? AND t.date = ?""",
                (account, datetime.now().strftime("%Y-%m-%d"))
            )
            result = cursor.fetchone()
            return result

    def get_character_by_type_and_id(self, char_type: str, char_id: int) -> Optional[Tuple]:
        """
        根据类型和ID获取角色信息

        Args:
            char_type: 角色类型 ('WifeStorage' 或 'HusbandStorage')
            char_id: 角色ID

        Returns:
            角色信息元组或 None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            table_name = char_type
            cursor.execute(
                f"SELECT id, name, work, image_name, description FROM {table_name} WHERE id = ?",
                (char_id,)
            )
            result = cursor.fetchone()
            return result


class TodayWifeOperator:
    """
    今日老婆数据库操作类，拥有增删改查权限
    """

    def __init__(self, db_path: str = "TodayWifeStorage.sqlite", user_db_path: str = "user_data.sqlite"):
        """
        初始化数据库连接和表结构

        Args:
            db_path: 今日老婆数据库文件路径
            user_db_path: 用户数据数据库文件路径
        """
        self.db_path = db_path
        self.user_db_path = user_db_path
        self._ensure_database_exists()
        self._ensure_tables_exist()

    def _get_connection(self, db_path: str = None):
        """获取数据库连接"""
        path = db_path or self.db_path
        return sqlite3.connect(path)

    def _ensure_database_exists(self):
        """确保数据库文件存在"""
        # 创建今日老婆数据库
        if not os.path.exists(self.db_path):
            with self._get_connection() as conn:
                conn.close()

        # 创建用户数据数据库
        if not os.path.exists(self.user_db_path):
            with self._get_connection(self.user_db_path) as conn:
                conn.close()

    def _ensure_tables_exist(self):
        """确保表结构存在"""
        # 创建今日老婆数据库中的表
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 创建WifeStorage表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS WifeStorage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    work TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    description TEXT
                )
            ''')

            # 创建HusbandStorage表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS HusbandStorage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    work TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    description TEXT
                )
            ''')

            conn.commit()

        # 创建用户数据数据库中的表
        with self._get_connection(self.user_db_path) as conn:
            cursor = conn.cursor()

            # 创建UserTable
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS UserTable (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account TEXT NOT NULL,
                    character_type TEXT NOT NULL,
                    character_data INTEGER NOT NULL,
                    date TEXT NOT NULL
                )
            ''')

            conn.commit()

    def add_wife(self, name: str, work: str, image_name: str, description: str = None) -> int:
        """
        添加老婆角色

        Args:
            name: 角色名
            work: 所属作品
            image_name: 图片名
            description: 角色简介

        Returns:
            新增记录的ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO WifeStorage (name, work, image_name, description) VALUES (?, ?, ?, ?)",
                (name, work, image_name, description)
            )
            conn.commit()
            return cursor.lastrowid

    def add_husband(self, name: str, work: str, image_name: str, description: str = None) -> int:
        """
        添加老公角色

        Args:
            name: 角色名
            work: 所属作品
            image_name: 图片名
            description: 角色简介

        Returns:
            新增记录的ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO HusbandStorage (name, work, image_name, description) VALUES (?, ?, ?, ?)",
                (name, work, image_name, description)
            )
            conn.commit()
            return cursor.lastrowid

    def update_wife(self, wife_id: int, name: str = None, work: str = None,
                   image_name: str = None, description: str = None) -> bool:
        """
        更新老婆角色信息

        Args:
            wife_id: 角色ID
            name: 角色名
            work: 所属作品
            image_name: 图片名
            description: 角色简介

        Returns:
            是否更新成功
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            updates = []
            params = []

            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if work is not None:
                updates.append("work = ?")
                params.append(work)
            if image_name is not None:
                updates.append("image_name = ?")
                params.append(image_name)
            if description is not None:
                updates.append("description = ?")
                params.append(description)

            if not updates:
                return False

            params.append(wife_id)
            query = f"UPDATE WifeStorage SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0

    def update_husband(self, husband_id: int, name: str = None, work: str = None,
                      image_name: str = None, description: str = None) -> bool:
        """
        更新老公角色信息

        Args:
            husband_id: 角色ID
            name: 角色名
            work: 所属作品
            image_name: 图片名
            description: 角色简介

        Returns:
            是否更新成功
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            updates = []
            params = []

            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if work is not None:
                updates.append("work = ?")
                params.append(work)
            if image_name is not None:
                updates.append("image_name = ?")
                params.append(image_name)
            if description is not None:
                updates.append("description = ?")
                params.append(description)

            if not updates:
                return False

            params.append(husband_id)
            query = f"UPDATE HusbandStorage SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0

    def delete_wife(self, wife_id: int) -> bool:
        """
        删除老婆角色

        Args:
            wife_id: 角色ID

        Returns:
            是否删除成功
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM WifeStorage WHERE id = ?", (wife_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_husband(self, husband_id: int) -> bool:
        """
        删除老公角色

        Args:
            husband_id: 角色ID

        Returns:
            是否删除成功
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM HusbandStorage WHERE id = ?", (husband_id,))
            conn.commit()
            return cursor.rowcount > 0

    def set_today_wife(self, account: str, char_type: str, char_id: int) -> bool:
        """
        设置用户今日角色

        Args:
            account: 用户账号
            char_type: 角色类型 ('WifeStorage' 或 'HusbandStorage')
            char_id: 角色ID

        Returns:
            是否设置成功
        """
        if char_type not in ['WifeStorage', 'HusbandStorage']:
            raise ValueError("character_type 必须是 'WifeStorage' 或 'HusbandStorage'")

        with self._get_connection(self.user_db_path) as conn:
            cursor = conn.cursor()

            # 先删除当天的记录（如果存在）
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute(
                "DELETE FROM UserTable WHERE account = ? AND date = ?",
                (account, today)
            )

            # 插入新的记录
            cursor.execute(
                "INSERT INTO UserTable (account, character_type, character_data, date) VALUES (?, ?, ?, ?)",
                (account, char_type, char_id, today)
            )
            conn.commit()
            return True

    def get_user_today_wife(self, account: str) -> Optional[Tuple]:
        """
        获取用户今日角色信息（从用户数据表）

        Args:
            account: 用户账号

        Returns:
            今日角色信息元组或 None
        """
        with self._get_connection(self.user_db_path) as conn:
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute(
                "SELECT id, account, character_type, character_data, date FROM UserTable WHERE account = ? AND date = ?",
                (account, today)
            )
            return cursor.fetchone()

    def clear_today_wife(self, account: str) -> bool:
        """
        清除用户今日角色设置

        Args:
            account: 用户账号

        Returns:
            是否清除成功
        """
        with self._get_connection(self.user_db_path) as conn:
            cursor = conn.cursor()
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute(
                "DELETE FROM UserTable WHERE account = ? AND date = ?",
                (account, today)
            )
            conn.commit()
            return cursor.rowcount > 0