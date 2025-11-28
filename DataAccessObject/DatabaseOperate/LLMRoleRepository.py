import sqlite3
import os

class LLMRoleRepository:
    database_name = "AIPromptRepository.sqlite"
    table_name = "LLMRoles"

    def __init__(self) -> None:
        # 自动初始化数据库连接和表结构
        self.conn = self._get_connection()
        self._initialize_table()

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接，如果数据库文件不存在则自动创建"""
        project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        database_path = os.path.join(project_path, f"DataAccessObject/DataStorage/{self.database_name}")
        return sqlite3.connect(database_path)

    def _initialize_table(self) -> None:
        """初始化表结构，如果表不存在则创建"""
        with self.conn:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    role_name TEXT PRIMARY KEY,
                    role_setting TEXT NOT NULL
                )
            """)

    def check_connection(self) -> bool:
        """检查数据库连接是否有效"""
        try:
            self.conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def get_role(self, role_name: str) -> str | None:
        """根据角色名获取角色设定"""
        cursor = self.conn.execute(
            f"SELECT role_setting FROM {self.table_name} WHERE role_name = ?",
            (role_name,)
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def set_role(self, role_name: str, role_setting: str) -> None:
        """设置或更新角色设定"""
        with self.conn:
            self.conn.execute(
                f"INSERT OR REPLACE INTO {self.table_name} (role_name, role_setting) VALUES (?, ?)",
                (role_name, role_setting)
            )

    def delete_role(self, role_name: str) -> bool:
        """删除指定角色，返回是否删除成功（即是否存在该角色）"""
        cursor = self.conn.execute(
            f"DELETE FROM {self.table_name} WHERE role_name = ?",
            (role_name,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def list_roles(self) -> list[str]:
        """列出所有角色名"""
        cursor = self.conn.execute(f"SELECT role_name FROM {self.table_name}")
        return [row[0] for row in cursor.fetchall()]

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    print("=== LLM 角色设定管理器 ===")
    print("1: 创建/更新角色")
    print("2: 查看角色设定")
    print("3: 删除角色")
    print("4: 列出所有角色")
    print("0: 退出")
    print("-" * 30)

    with LLMRoleRepository() as repo:
        while True:
            choice = input("请选择操作 (0-4): ").strip()
            if choice == "0":
                print("再见！")
                break
            elif choice == "1":
                role_name = input("请输入角色名: ").strip()
                if not role_name:
                    print("角色名不能为空！")
                    continue
                role_setting = input("请输入角色设定: ").strip()
                if not role_setting:
                    print("角色设定不能为空！")
                    continue
                repo.set_role(role_name, role_setting)
                print(f"✅ 角色 '{role_name}' 已保存。")
            elif choice == "2":
                role_name = input("请输入要查询的角色名: ").strip()
                if not role_name:
                    print("角色名不能为空！")
                    continue
                setting = repo.get_role(role_name)
                if setting is not None:
                    print(f"📖 角色 '{role_name}' 的设定：\n{setting}")
                else:
                    print(f"❌ 未找到角色 '{role_name}'。")
            elif choice == "3":
                role_name = input("请输入要删除的角色名: ").strip()
                if not role_name:
                    print("角色名不能为空！")
                    continue
                if repo.delete_role(role_name):
                    print(f"🗑️ 角色 '{role_name}' 已删除。")
                else:
                    print(f"❌ 角色 '{role_name}' 不存在，无法删除。")
            elif choice == "4":
                roles = repo.list_roles()
                if roles:
                    print("📋 所有角色：")
                    for name in roles:
                        print(f" - {name}")
                else:
                    print("📭 暂无任何角色。")
            else:
                print("⚠️ 无效选项，请输入 0-4。")
