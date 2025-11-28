# aris_chatbot.py
"""
AIRS Chatbot Local - Main Entry Point

This is the main entry script to launch the local AIRS Chatbot application.
It initializes and starts the core application in an object-oriented manner.

========================= 中文说明 =========================

ARIS Chatbot Local - 主入口

这是启动本地 AIRS Chatbot 应用程序的主入口脚本。
它以面向对象的方式初始化并启动核心应用程序。

日志级别：
- DEBUG：调试信息，包含所有信息，包括程序运行时的详细信息，包括变量的值、函数调用的堆栈信息等。
- INFO：一般信息，包含程序运行的大概情况，包括程序启动、结束、运行时间等。
- WARNING：警告信息，包含程序运行中可能出现的异常情况，但程序仍能正常运行。
- ERROR：错误信息，包含程序运行中出现的严重错误，程序无法继续运行。
- CRITICAL：严重错误信息，包含程序运行中出现的致命错误，程序崩溃退出。
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from UIWidgets.main_window import MainWindow
from configs import ConfigManager, ASRConfigs


class AIRSChatbotApp:
    """Main application class for AIRS Chatbot Local.
    AIRS Chatbot Local的主应用程序类"""

    def __init__(self):
        """Initialize the AIRS Chatbot application.
        初始化 AIRS Chatbot 应用程序"""
        self.app = QApplication(sys.argv)
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    def run(self):
        """Start the application event loop.
        启动应用程序事件循环"""
        print("Starting AIRS Chatbot Local...")
        self.main_window = MainWindow()
        self.main_window.show()
        sys.exit(self.app.exec())


def main():
    """Application entry point.
    应用程序入口"""
    config_manager = ConfigManager()
    config_manager.handle_update()
    print(ASRConfigs.model_path)
    bot = AIRSChatbotApp()
    bot.run()


if __name__ == "__main__":
    main()
