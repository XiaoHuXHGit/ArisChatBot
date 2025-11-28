# main.py
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow
from UIWidgets.UIStyles.main_window import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        # self.setWindowFlags(Qt.FramelessWindowHint)
        self.setupUi(self)  # 加载 UI

        # 🔗 绑定按钮点击事件
        self.windowMiniumButton.clicked.connect(self.minimize_window)
        self.windowMaxButton.clicked.connect(self.maximize_window)
        self.exitButton.clicked.connect(self.close_window)


    def minimize_window(self):
        print("Minimize clicked")
        # self.showMinimized()

    def maximize_window(self):
        # if self.isMaximized():
        #     self.showNormal()
        # else:
        #     self.showMaximized()
        print("Maximize clicked")

    def close_window(self):
        print("Exit clicked")
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())