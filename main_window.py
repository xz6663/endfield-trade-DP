# main_window.py
# PyQt5主界面入口

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from dp_gui import DpTab, TradeTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Endfield Trade DP')
        self.resize(1100, 800)
        self.tabs = QTabWidget()
        self.tabs.addTab(DpTab(), 'DP策略调参与测试')
        self.tabs.addTab(TradeTab(), '开始交易！')
        self.setCentralWidget(self.tabs)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
