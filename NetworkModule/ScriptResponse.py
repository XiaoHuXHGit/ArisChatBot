import warnings
import random
import logging
import numpy as np

import jieba
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from DataAccessObject.DatabaseOperate.CharacterChatDataset import CharacterChatDataset

warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

# 假设停用词列表已加载
# STOP_WORDS = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
#               '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '里', '就是', '还是', '啊',
#               '呀', '吧', '呢', '哦', '嗯'}
STOP_WORDS = {}


# --- 核心响应引擎 ---
class ScriptResponse:
    def __init__(self, db_interface=None, character_name="default", similarity_threshold=0.3,
                 fallback_responses=None, temperature=0.7):
        """
        初始化聊天机器人回复引擎。
        :param db_interface: 可选，传入已有的 CharacterChatDataset 实例
        :param character_name: 若未提供 db_interface，则用此角色名创建新数据库表
        :param similarity_threshold: TF-IDF余弦相似度阈值
        :param fallback_responses: 默认回复列表
        :param temperature: 温度参数，控制随机性，值越大越随机，越小越偏向高相似度
        """
        if db_interface is not None:
            self.db = db_interface
        else:
            self.db = CharacterChatDataset(character_name=character_name)

        self.similarity_threshold = similarity_threshold
        self.temperature = temperature  # 温度参数控制随机性
        self.fallback_responses = fallback_responses or [
            "爱丽丝在网上就是爹",
            "把你们群最牛逼的叫出来，爱丽丝要把他大卸八块",
            "爱丽丝想干什么就干什么"
        ]
        # 内部用于处理和匹配的变量
        self.vectorizer = None
        self.tfidf_matrix = None
        self.processed_inputs = []  # 存储预处理后的输入文本
        # 初始加载数据库内容
        self._rebuild_index()

    def _preprocess(self, text):
        """使用jieba分词并去除停用词"""
        words = jieba.lcut(text.strip())
        filtered = [w for w in words if w not in STOP_WORDS and len(w.strip()) > 0]
        return " ".join(filtered)

    def _rebuild_index(self):
        """根据数据库当前内容，重建TF-IDF向量索引和预处理列表"""
        all_qa_pairs = self.db.get_all_qa()

        if not all_qa_pairs:
            self.processed_inputs = []
            self.vectorizer = None
            self.tfidf_matrix = None
            logging.debug("Index rebuilt: No QA pairs in DB.")
            return

        raw_inputs = [qa[0] for qa in all_qa_pairs]
        self.processed_inputs = [self._preprocess(text) for text in raw_inputs]

        # 关键防护：检查是否所有预处理后的文档都为空（仅含停用词或空白）
        non_empty_docs = [doc for doc in self.processed_inputs if doc.strip()]
        if not non_empty_docs:
            logging.warning(
                "All user inputs become empty after preprocessing (likely only stop words). "
                "Disabling TF-IDF matching; falling back to fuzzy string matching."
            )
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        try:
            self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_inputs)
            logging.debug(f"TF-IDF index rebuilt successfully with {len(all_qa_pairs)} QA pairs.")
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                logging.error(
                    "TF-IDF failed despite non-empty check — possible edge case. "
                    "Falling back to fuzzy matching."
                )
                self.vectorizer = None
                self.tfidf_matrix = None
            else:
                raise  # 其他错误仍抛出

    def _softmax_sampling(self, outputs, scores, temperature=0.7):
        """
        使用softmax和温度参数进行概率采样，确保所有候选都有机会被选中
        :param outputs: 候选回复列表
        :param scores: 对应的相似度分数列表
        :param temperature: 温度参数，控制随机性
        :return: 选中的回复
        """
        if not outputs:
            return None

        scores_array = np.array(scores)

        # 应用温度缩放
        scaled_scores = scores_array / temperature

        # 计算softmax概率
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))  # 减去最大值防止溢出
        probabilities = exp_scores / exp_scores.sum()

        # 根据概率随机选择
        chosen_idx = np.random.choice(len(outputs), p=probabilities)
        return outputs[chosen_idx]

    def find_best_match(self, input_text):
        """
        根据输入文本，使用TF-IDF和余弦相似度找到最佳匹配的问答对。
        现在会对所有超过阈值的匹配进行概率采样，确保每个匹配都有机会被选中
        :param input_text: 用户输入
        :return: (best_output, best_similarity_score) 或 (None, 0)
        """
        if not self.db.get_qa_count() or self.tfidf_matrix is None or self.vectorizer is None:
            return None, 0

        processed_input = self._preprocess(input_text)

        # 计算输入文本的TF-IDF向量
        input_vector = self.vectorizer.transform([processed_input])

        # 计算与所有已存储输入的余弦相似度
        similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()

        # 找到所有超过阈值的匹配项（关键改进：不限制数量）
        valid_indices = np.where(similarities >= self.similarity_threshold)[0]

        if len(valid_indices) == 0:
            # 如果TF-IDF没有找到足够相似的匹配，尝试fuzz.ratio匹配
            return self._fuzzy_fallback_match(input_text)

        # 获取所有有效匹配的输出和分数
        all_qa_pairs = self.db.get_all_qa()
        matched_outputs = [all_qa_pairs[idx][1] for idx in valid_indices]
        matched_scores = [similarities[idx] for idx in valid_indices]

        # 使用softmax采样选择最终输出（所有匹配项都有机会被选中）
        selected_output = self._softmax_sampling(matched_outputs, matched_scores, self.temperature)

        if selected_output:
            best_similarity = max(matched_scores)
            logging.debug(
                f"Softmax sampling selected: '{selected_output}' from {len(matched_outputs)} total candidates")
            return selected_output, best_similarity

        # 如果采样失败，回退到最高分匹配
        best_idx = valid_indices[np.argmax(similarities[valid_indices])]
        best_output = all_qa_pairs[best_idx][1]
        best_similarity = similarities[best_idx]
        return best_output, best_similarity

    def _fuzzy_fallback_match(self, input_text):
        """
        使用fuzz.ratio作为备选匹配方法，对所有高相似度匹配进行概率采样
        """
        all_qa_pairs = self.db.get_all_qa()
        if not all_qa_pairs:
            return None, 0

        # 收集所有高相似度的匹配（不限制数量）
        RATIO_THRESHOLD = 85
        high_similarity_matches = []

        for input_text_stored, output_text_stored in all_qa_pairs:
            ratio_score = fuzz.ratio(input_text, input_text_stored)
            if ratio_score >= RATIO_THRESHOLD:
                high_similarity_matches.append((output_text_stored, ratio_score))

        if not high_similarity_matches:
            return None, 0

        # 提取输出和分数
        outputs = [match[0] for match in high_similarity_matches]
        scores = [match[1] for match in high_similarity_matches]

        # 使用softmax采样
        selected_output = self._softmax_sampling(outputs, scores, self.temperature)

        if selected_output:
            best_ratio_score = max(scores)
            logging.debug(f"Fallback softmax sampling selected: '{selected_output}' from {len(outputs)} candidates")
            return selected_output, best_ratio_score / 100.0

        # 回退到最高分匹配
        best_match_idx = np.argmax(scores)
        best_match = high_similarity_matches[best_match_idx]
        return best_match[0], best_match[1] / 100.0

    def get_response(self, user_input):
        """
        主接口：根据用户输入返回最匹配的回复或默认回复。
        :param user_input: 用户原始输入字符串
        :return: 回复字符串
        """
        if not user_input.strip():
            return random.choice(self.fallback_responses)

        # 在数据库中查找最佳匹配
        best_output, score = self.find_best_match(user_input)

        if best_output:
            logging.info(f"Found match for '{user_input}' -> '{best_output}' (score: {score:.2f})")
            return best_output
        else:
            logging.info(f"No match found for '{user_input}'. Returning fallback response. (score: {score:.2f})")
            return random.choice(self.fallback_responses)

    def learn(self, input_text, output_text):
        """
        学习新的问答对。
        :param input_text: 用户的输入
        :param output_text: 期望的回复
        """
        self.db.add_qa(input_text, output_text)
        # 添加后需要重建索引
        self._rebuild_index()


# --- 测试代码 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 使用持久化数据库（角色名为 "alice"）
    with CharacterChatDataset(character_name="alice") as db:
        # 可以调整top_k和top_p参数来控制多样性
        responder = ScriptResponse(
            db_interface=db,
            similarity_threshold=0.3
        )

        print("--- 持久化聊天机器人测试 (SQLite 后端) ---")
        print("输入 'quit' 退出。")
        print("输入 'learn' 开始学习模式。")

        # 初始学习一些问答（仅当数据库为空时才添加，避免重复）
        if db.get_qa_count() == 0:
            print("\n--- 初始学习 ---")
            initial_qa = [
                ("你好", "你好呀！今天过得怎么样？"),
                ("你好", "嗨！很高兴见到你！"),  # 重复输入，不同回复
                ("你好", "你好世界！爱丽丝在这里等你呢！"),  # 重复输入，不同回复
                ("晚上好", "嗯嗯，晚上好，老师，要不要跟爱丽丝打一会游戏歇一歇？"),
                ("晚上好", "爱丽丝今天又通关了一款游戏呢！爱丽丝厉不厉害！"),
                ("晚上好", "晚上好呀，今天玩得开心吗？"),
                ("你会什么", "我会聊天、讲笑话，还能陪你玩文字游戏呢！")
            ]
            for q, a in initial_qa:
                responder.learn(q, a)
                print(f"已学习: {q} -> {a}")
        else:
            print(f"\n数据库已存在 {db.get_qa_count()} 条记录，跳过初始学习。")

        print("\n--- 自动测试 ---")
        test_inputs = [
            "你好",
            "晚上好",
            "你会什么",
            "讲个笑话",
            "随便聊聊",  # 不在初始学习中
        ]
        for user_input in test_inputs:
            print(f"\n用户输入: {user_input}")
            # 多次测试以展示随机性
            for i in range(3):
                response = responder.get_response(user_input)
                print(f"  回复 {i + 1}: {response}")

        print("\n--- 交互式测试 ---")
        while True:
            try:
                user_input = input("\n你: ")
                if user_input.lower() == 'quit':
                    print("再见！")
                    break
                elif user_input.lower() == 'learn':
                    new_input = input("请输入新的用户问题: ")
                    new_output = input("请输入你希望我如何回复: ")
                    responder.learn(new_input, new_output)
                    print(f"好的，我已经学会了 '{new_input}' -> '{new_output}'")
                    print(f"数据库当前有 {db.get_qa_count()} 条记录。")
                else:
                    response = responder.get_response(user_input)
                    print(f"机器人: {response}")
            except KeyboardInterrupt:
                print("\n再见！")
                break