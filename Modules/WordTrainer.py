import sqlite3
import random
from datetime import datetime
from typing import List, Tuple, Optional


class VocabularyDB:
    def __init__(self, db_path: str = "vocabulary.sqlite"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """初始化数据库，创建三张表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 单词表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL UNIQUE,
                translation TEXT NOT NULL
            )
        ''')

        # 统计表（记录出现次数、错误次数、最后出错时间）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stats (
                word TEXT PRIMARY KEY,
                appear_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                last_error_time TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def add_word(self, word: str, translation: str) -> bool:
        """添加单词及其翻译，同时初始化统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 插入单词表
            cursor.execute(
                "INSERT INTO words (word, translation) VALUES (?, ?)",
                (word, translation)
            )

            # 初始化统计表
            cursor.execute(
                "INSERT INTO stats (word, appear_count, error_count, last_error_time) VALUES (?, 0, 0, NULL)",
                (word,)
            )

            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            # 单词已存在
            conn.close()
            return False

    def search_words(self, query: str) -> List[Tuple[int, str, str]]:
        """根据单词（精确）或翻译（模糊）查询"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 先尝试精确匹配单词
        cursor.execute("SELECT id, word, translation FROM words WHERE word = ?", (query,))
        results = cursor.fetchall()

        if not results:
            # 模糊匹配翻译
            cursor.execute("SELECT id, word, translation FROM words WHERE translation LIKE ?", (f"%{query}%",))
            results = cursor.fetchall()

        conn.close()
        return results

    def get_review_words(self, limit: int = 50) -> List[str]:
        """获取用于复习的单词列表，按优先级排序"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 按出现次数升序、错误次数降序、最后出错时间升序排序
        cursor.execute('''
            SELECT word 
            FROM stats 
            ORDER BY 
                appear_count ASC, 
                error_count DESC,
                last_error_time ASC NULLS LAST
            LIMIT ?
        ''', (limit,))

        words = [row[0] for row in cursor.fetchall()]
        conn.close()
        return words

    def update_stats(self, word: str, is_correct: bool):
        """更新单词的统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 更新出现次数
        cursor.execute(
            "UPDATE stats SET appear_count = appear_count + 1 WHERE word = ?",
            (word,)
        )

        if not is_correct:
            # 更新错误次数和最后出错时间
            now = datetime.now().isoformat()
            cursor.execute(
                "UPDATE stats SET error_count = error_count + 1, last_error_time = ? WHERE word = ?",
                (now, word)
            )

        conn.commit()
        conn.close()

    def get_translation(self, word: str) -> Optional[str]:
        """获取单词的翻译"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT translation FROM words WHERE word = ?", (word,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None


class VocabularyTrainer:
    def __init__(self, db_path: str = "vocabulary.sqlite"):
        self.db = VocabularyDB(db_path)

    def add_words_interactive(self):
        """交互式添加单词"""
        print("=== 添加单词 ===")
        print("输入单词，输入 /0 结束")

        while True:
            word = input("单词: ").strip()
            if word == "/0":
                break
            if not word:
                continue

            translation = input("翻译: ").strip()
            if self.db.add_word(word, translation):
                print(f"✅ 已添加: {word} -> {translation}")
            else:
                print(f"⚠️  单词 '{word}' 已存在")

    def search_interactive(self):
        """交互式查询单词"""
        print("=== 查询单词 ===")
        query = input("请输入单词或翻译关键词: ").strip()
        results = self.db.search_words(query)

        if results:
            for wid, word, trans in results:
                print(f"{word}\n{trans.replace(" \\n ", "\n")}")
        else:
            print("未找到相关单词")

    def review_words(self):
        """单词测试"""
        print("=== 单词测试 ===")
        words = self.db.get_review_words(50)
        if not words:
            print("暂无单词可供测试")
            return

        random.shuffle(words)

        for word in words:
            translation = self.db.get_translation(word)
            if translation is None:
                continue

            print(f"\n单词: {word}")
            choice = input("0. 不认识  1. 认识  e. 退出测试\n请选择: ").strip()

            if choice == 'e':
                break
            elif choice == '1':
                print(f"✅ 翻译: {translation}")
                self.db.update_stats(word, is_correct=True)
            elif choice == '0':
                print(f"❌ 翻译: {translation}")
                self.db.update_stats(word, is_correct=False)
            else:
                print("无效输入，跳过此单词")

    def run(self):
        """主控制台循环"""
        while True:
            print("\n=== 背单词系统 ===")
            print("1. 添加单词")
            print("2. 查询单词")
            print("3. 单词测试")
            print("0. 退出")

            choice = input("请选择功能: ").strip()
            if choice == '1':
                self.add_words_interactive()
            elif choice == '2':
                self.search_interactive()
            elif choice == '3':
                self.review_words()
            elif choice == '0':
                print("再见！")
                break
            else:
                print("无效选项")


if __name__ == "__main__":
    trainer = VocabularyTrainer(
        db_path=r"E:\programs\aris_chatbot_local\aris_chatbot\DataAccessObject\DataStorage\vocabulary.sqlite")
    trainer.run()
