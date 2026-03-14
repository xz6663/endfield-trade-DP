# main_window.py
# PyQt5主界面入口

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from dp_gui import DpTab, TradeTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Endfield Trade DP')
        self.resize(1500, 920)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F3F6FB;
            }
            QWidget {
                color: #1F2937;
                font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                font-size: 20px;
            }
            QTabWidget::pane {
                border: 1px solid #D8E0EC;
                background: #FFFFFF;
                border-radius: 10px;
                top: -1px;
            }
            QTabBar::tab {
                background: #EAF0F9;
                color: #334155;
                border: 1px solid #D8E0EC;
                padding: 8px 18px;
                margin-right: 6px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                color: #0F172A;
            }
            QGroupBox {
                background: #FFFFFF;
                border: 1px solid #DEE6F2;
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #334155;
            }
            QPushButton {
                background-color: #2563EB;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
                min-height: 18px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton:pressed {
                background-color: #1E40AF;
            }
            QPushButton:disabled {
                background-color: #93C5FD;
                color: #E5E7EB;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #F8FAFC;
                border: 1px solid #CBD5E1;
                border-radius: 8px;
                padding: 6px 8px;
                min-height: 18px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #2563EB;
                background: #FFFFFF;
            }
            QTableWidget {
                background: #FFFFFF;
                border: 1px solid #D8E0EC;
                border-radius: 8px;
                gridline-color: #E8EEF7;
                selection-background-color: #DBEAFE;
                selection-color: #0F172A;
            }
            QHeaderView::section {
                background: #EEF3FA;
                color: #334155;
                border: none;
                border-right: 1px solid #D8E0EC;
                border-bottom: 1px solid #D8E0EC;
                padding: 6px;
                font-weight: 600;
            }
            QLabel {
                color: #334155;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QSplitter::handle {
                background: #E2E8F0;
            }
            QSplitter::handle:horizontal {
                width: 6px;
            }
            QSplitter::handle:vertical {
                height: 6px;
            }
        """)
        self.tabs = QTabWidget()
        self.tabs.addTab(DpTab(), 'DP策略调参与测试')
        self.tabs.addTab(TradeTab(), '开始交易！')
        self.setCentralWidget(self.tabs)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
