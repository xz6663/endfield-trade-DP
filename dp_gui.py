# dp_gui.py
# DP策略调参与可视化Tab

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit, QTableWidget, QTableWidgetItem, QGroupBox, QMessageBox, QFileDialog, QHeaderView, QTabWidget, QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import dp
import pickle
import time

class TradeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.presets = {
            '四号谷地': {
                'product_names': ["锚点厨具货组","悬空鼷兽骨雕货组","巫术矿钻货组","天使罐头货组","谷地水培肉货组","团结牌口服液货组","塞什卡髀石货组","源石树幼苗货组","星体晶块货组","警戒者矿镐货组","边角料积木货组","硬脑壳头盔货组"],
                'shelf_cap': 960,
                'shelf_replenish_per_day': 320,
                'shelf_init': 320
            },
            '武陵': {
                'product_names': ["武陵冻梨货组","岳研避瘴茶货组","冬虫夏笋货组","武侠电影货组"],
                'shelf_cap': 100,
                'shelf_replenish_per_day': 50,
                'shelf_init': 50
            }
        }
        self.cfg = dp.Config(
            seed=42,
            product_names=self.presets['四号谷地']['product_names'],
            n_friends=20,
            horizon_days=7,
            shelf_cap=self.presets['四号谷地']['shelf_cap'],
            shelf_replenish_per_day=self.presets['四号谷地']['shelf_replenish_per_day'],
            shelf_init=self.presets['四号谷地']['shelf_init'],
            z_samples=200000,
            eval_episodes=20000,
        )
        self.theta = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        font = self.font(); font.setPointSize(12); self.setFont(font)
        # 预设选择
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel('选择配置:'))
        self.preset_combo = QComboBox(); self.preset_combo.addItems(list(self.presets.keys()))
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        layout.addLayout(preset_layout)
        # 参数区
        param_box = QGroupBox('参数调节')
        param_box.setFont(font)
        form = QFormLayout()
        self.param_widgets = {}
        param_names = {
            'product_names': '商品名称列表',
            'n_friends': '模拟好友数',
            'horizon_days': '周期天数',
            'shelf_cap': '货架容量',
            'shelf_replenish_per_day': '每日补货量',
            'shelf_init': '初始货架余量',
            'z_samples': 'z采样量',
            'eval_episodes': '评估路径数',
            'seed': '随机种子',
            'time': '用时间做种子'
        }
        for name, val in self.cfg.__dict__.items():
            if name in ['time']:
                continue
            label = param_names.get(name, name)
            if isinstance(val, int):
                w = QSpinBox(); w.setMaximum(1000000); w.setValue(val); w.setFont(font)
            elif isinstance(val, float):
                w = QDoubleSpinBox(); w.setDecimals(6); w.setMaximum(1e8); w.setValue(val); w.setFont(font)
            elif isinstance(val, list):
                w = QLineEdit(','.join(val)); w.setFont(font)
            else:
                w = QLineEdit(str(val)); w.setFont(font)
            self.param_widgets[name] = w
            form.addRow(label, w)
        # 新增time复选框
        self.time_checkbox = QCheckBox('用时间做种子'); self.time_checkbox.setFont(font)
        self.time_checkbox.setChecked(True)
        form.addRow(self.time_checkbox)
        param_box.setLayout(form)
        layout.addWidget(param_box)
        # 求解按钮
        self.btn_solve = QPushButton('求解DP表'); self.btn_solve.setFont(font)
        self.btn_solve.clicked.connect(self.on_solve)
        layout.addWidget(self.btn_solve)
        # 人工输入区
        manual_box = QGroupBox('用户输入决策'); manual_box.setFont(font)
        m_layout = QVBoxLayout()
        m_layout.addWidget(QLabel('当前天数:'))
        self.day_spin = QSpinBox(); self.day_spin.setMaximum(self.cfg.horizon_days); self.day_spin.setValue(1); self.day_spin.setFont(font)
        m_layout.addWidget(self.day_spin)
        m_layout.addWidget(QLabel('当前可购买余量S:'))
        self.shelf_spin = QSpinBox(); self.shelf_spin.setMaximum(self.cfg.shelf_cap); self.shelf_spin.setValue(self.cfg.shelf_init); self.shelf_spin.setFont(font)
        m_layout.addWidget(self.shelf_spin)
        self.price_table = QTableWidget(len(self.cfg.product_names), 2); self.price_table.setFont(font)
        self.price_table.setHorizontalHeaderLabels(['买入价', '好友最高价'])
        for i, name in enumerate(self.cfg.product_names):
            self.price_table.setVerticalHeaderItem(i, QTableWidgetItem(name))
        m_layout.addWidget(self.price_table)
        self.btn_decide = QPushButton('输出最优动作'); self.btn_decide.setFont(font)
        self.btn_decide.clicked.connect(self.on_decide)
        m_layout.addWidget(self.btn_decide)
        self.manual_result = QLabel(''); self.manual_result.setFont(font)
        m_layout.addWidget(self.manual_result)
        manual_box.setLayout(m_layout)
        layout.addWidget(manual_box)
        self.setLayout(layout)

    def on_preset_changed(self, preset):
        p = self.presets[preset]
        self.cfg.product_names = p['product_names']
        self.cfg.shelf_cap = p['shelf_cap']
        self.cfg.shelf_replenish_per_day = p['shelf_replenish_per_day']
        self.cfg.shelf_init = p['shelf_init']
        self.param_widgets['product_names'].setText(','.join(p['product_names']))
        self.param_widgets['shelf_cap'].setValue(p['shelf_cap'])
        self.param_widgets['shelf_replenish_per_day'].setValue(p['shelf_replenish_per_day'])
        self.param_widgets['shelf_init'].setValue(p['shelf_init'])
        self.price_table.setRowCount(len(p['product_names']))
        for i, name in enumerate(p['product_names']):
            self.price_table.setVerticalHeaderItem(i, QTableWidgetItem(name))
        self.shelf_spin.setMaximum(p['shelf_cap'])
        self.shelf_spin.setValue(p['shelf_init'])

    def on_solve(self):
        for name, w in self.param_widgets.items():
            if isinstance(w, QSpinBox):
                setattr(self.cfg, name, w.value())
            elif isinstance(w, QDoubleSpinBox):
                setattr(self.cfg, name, w.value())
            elif isinstance(w, QLineEdit):
                if name == 'product_names':
                    setattr(self.cfg, name, [x.strip() for x in w.text().split(',') if x.strip()])
                else:
                    setattr(self.cfg, name, w.text())
        self.cfg.time = self.time_checkbox.isChecked()
        self.btn_solve.setEnabled(False)
        self.btn_solve.setText('正在生成...')
        self.solver_thread = DPSolverThread(self.cfg, dp)
        self.solver_thread.finished.connect(self.on_solve_finished)
        self.solver_thread.start()
    def on_solve_finished(self, V, theta):
        self.theta = theta
        self.btn_solve.setEnabled(True)
        self.btn_solve.setText('求解DP表')
        QMessageBox.information(self, '完成', 'DP表生成完成！')

    def on_decide(self):
        if self.theta is None:
            QMessageBox.warning(self, '未生成DP表', '请先求解DP表！')
            return
        n = len(self.cfg.product_names)
        self_p = []
        best_p = []
        for i in range(n):
            buy_item = self.price_table.item(i, 0)
            sell_item = self.price_table.item(i, 1)
            try:
                buy = float(buy_item.text()) if buy_item else 0
                sell = float(sell_item.text()) if sell_item else 0
            except:
                buy, sell = 0, 0
            self_p.append(buy)
            best_p.append(sell)
        self_p = np.array(self_p)
        best_p = np.array(best_p)
        spread = best_p - self_p
        z = float(np.max(spread))
        t = self.day_spin.value() - 1
        S_after = self.shelf_spin.value()
        th = float(self.theta[t, S_after])
        if S_after == 0:
            msg = f'今日可用容量为0，无法操作。'
        elif z >= th:
            idx = int(np.argmax(spread))
            name = self.cfg.product_names[idx] if idx < len(self.cfg.product_names) else f'商品{idx+1}'
            S_end = 0
            msg = f'最优动作: 梭哈\n最大差价={z:.2f} >= 阈值{th:.2f}\n建议全部买入并卖出。\n商品:{name} 买入价:{self_p[idx]:.2f} 卖出价:{best_p[idx]:.2f}\n操作后货架余量:{S_end}'
        else:
            S_end = S_after
            msg = f'最优动作: 存着\n最大差价={z:.2f} < 阈值{th:.2f}\n建议观望。\n操作后货架余量:{S_end}'
        self.manual_result.setText(msg)

class DPSolverThread(QThread):
    finished = pyqtSignal(object, object)
    def __init__(self, cfg, dp_module):
        super().__init__()
        self.cfg = cfg
        self.dp = dp_module
    def run(self):
        z = self.dp.sample_z_distribution(self.cfg)
        V, theta = self.dp.compute_dp_threshold_policy(self.cfg, z)
        self.finished.emit(V, theta)

class DpTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cfg = dp.Config(
            seed=42,
            product_names=[f"商品{i+1}" for i in range(10)],
            n_friends=20,
            horizon_days=7,
            shelf_cap=500,
            shelf_replenish_per_day=80,
            shelf_init=200,
            z_samples=200000,
            eval_episodes=20000,
        )
        self.theta = None
        self.dp_table = None
        self.decision_details = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)
        # 参数区
        param_box = QGroupBox('DP参数调节')
        param_box.setFont(font)
        form = QFormLayout()
        self.param_widgets = {}
        self.product_names_widget = None
        for name, val in self.cfg.__dict__.items():
            if name in ['mean_price', 'log_sigma', 'time']:
                continue
            if isinstance(val, int):
                w = QSpinBox(); w.setMaximum(1000000); w.setValue(val); w.setFont(font)
            elif isinstance(val, float):
                w = QDoubleSpinBox(); w.setDecimals(6); w.setMaximum(1e8); w.setValue(val); w.setFont(font)
            elif isinstance(val, list):
                w = QLineEdit(','.join(val)); w.setFont(font)
                if name == 'product_names':
                    self.product_names_widget = w
                    w.editingFinished.connect(self.on_product_names_changed)
            else:
                w = QLineEdit(str(val)); w.setFont(font)
            self.param_widgets[name] = w
            form.addRow(name, w)
        # 新增time复选框
        self.time_checkbox = QCheckBox('用时间做种子'); self.time_checkbox.setFont(font)
        self.time_checkbox.setChecked(True)
        form.addRow(self.time_checkbox)
        param_box.setLayout(form)
        layout.addWidget(param_box)
        # 按钮区
        btn_layout = QHBoxLayout()
        self.btn_build = QPushButton('生成DP表并自动测试'); self.btn_build.setFont(font)
        self.btn_build.clicked.connect(self.on_build_dp)
        btn_layout.addWidget(self.btn_build)
        self.btn_save = QPushButton('保存DP表'); self.btn_save.setFont(font)
        self.btn_save.clicked.connect(self.on_save_dp)
        btn_layout.addWidget(self.btn_save)
        self.btn_load = QPushButton('导入DP表'); self.btn_load.setFont(font)
        self.btn_load.clicked.connect(self.on_load_dp)
        btn_layout.addWidget(self.btn_load)
        layout.addLayout(btn_layout)
        # DP表格区
        self.dp_table_widget = QTableWidget()
        self.dp_table_widget.setFont(font)
        layout.addWidget(QLabel('DP阈值表（theta）'))
        layout.addWidget(self.dp_table_widget)
        # 图表区
        self.fig = Figure(figsize=(6,4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        # 决策详情区
        self.detail_table = QTableWidget()
        self.detail_table.setFont(font)
        layout.addWidget(QLabel('每日决策详情'))
        layout.addWidget(self.detail_table)
        # 人工输入区
        self.manual_box = QGroupBox('人工输入每日价格并输出动作'); self.manual_box.setFont(font)
        m_layout = QVBoxLayout()
        self.day_spin = QSpinBox(); self.day_spin.setMaximum(self.cfg.horizon_days); self.day_spin.setValue(1); self.day_spin.setFont(font)
        m_layout.addWidget(QLabel('当前天数:'))
        m_layout.addWidget(self.day_spin)
        self.price_table = QTableWidget(len(self.cfg.product_names), 2)
        self.price_table.setFont(font)
        self.price_table.setHorizontalHeaderLabels(['买入价', '好友最高价'])
        for i, name in enumerate(self.cfg.product_names):
            self.price_table.setVerticalHeaderItem(i, QTableWidgetItem(name))
        m_layout.addWidget(self.price_table)
        self.btn_manual = QPushButton('输出最优动作'); self.btn_manual.setFont(font)
        self.btn_manual.clicked.connect(self.on_manual_action)
        m_layout.addWidget(self.btn_manual)
        self.btn_auto_fill = QPushButton('自动填入机器生成价格'); self.btn_auto_fill.setFont(font)
        self.btn_auto_fill.clicked.connect(self.on_auto_fill)
        m_layout.addWidget(self.btn_auto_fill)
        self.manual_result = QLabel(''); self.manual_result.setFont(font)
        m_layout.addWidget(self.manual_result)
        self.manual_box.setLayout(m_layout)
        layout.addWidget(self.manual_box)
        self.setLayout(layout)

    def on_build_dp(self):
        for name, w in self.param_widgets.items():
            if isinstance(w, QSpinBox):
                setattr(self.cfg, name, w.value())
            elif isinstance(w, QDoubleSpinBox):
                setattr(self.cfg, name, w.value())
            elif isinstance(w, QLineEdit):
                if name == 'product_names':
                    setattr(self.cfg, name, [x.strip() for x in w.text().split(',') if x.strip()])
                else:
                    setattr(self.cfg, name, w.text())
        self.cfg.time = self.time_checkbox.isChecked()
        self.btn_build.setEnabled(False)
        self.btn_build.setText('正在生成...')
        self.solver_thread = DPSolverThread(self.cfg, dp)
        self.solver_thread.finished.connect(self.on_build_dp_finished)
        self.solver_thread.start()
    def on_build_dp_finished(self, V, theta):
        self.theta = theta
        self.dp_table = theta.copy()
        self.show_dp_table()
        self.solver_thread.finished.disconnect()
        self.solver_thread = None
        self.btn_build.setText('自动测试中...')
        self.btn_build.setEnabled(False)
        self.auto_test_thread = DPAutoTestThread(self.cfg, self.theta, dp)
        self.auto_test_thread.finished.connect(self.on_auto_test_finished)
        self.auto_test_thread.start()

    def on_auto_test_finished(self, theta, decision_details, dp_profits, base_profits, price_pool):
        # 彻底防止死锁/卡死：先断开信号、销毁线程，再做UI更新
        self.auto_test_thread.finished.disconnect()
        self.auto_test_thread = None
        self.decision_details = decision_details
        self.show_decision_details()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(range(1, self.cfg.horizon_days+1), np.cumsum(dp_profits), marker='o', label='DP策略累计利润')
        ax.plot(range(1, self.cfg.horizon_days+1), np.cumsum(base_profits), marker='x', label='Baseline累计利润')
        ax.set_title('DP策略 vs Baseline 每日累计收益曲线')
        ax.set_xlabel('天数')
        ax.set_ylabel('累计利润')
        ax.legend()
        ax.grid(True)
        self.canvas.draw()
        self.btn_build.setEnabled(True)
        self.btn_build.setText('生成DP表并自动测试')
        QMessageBox.information(self, '完成', '阈值策略生成、自动测试与决策详情输出完成！')

    def show_dp_table(self):
        if self.dp_table is None:
            return
        H, cap = self.dp_table.shape
        self.dp_table_widget.setRowCount(H)
        self.dp_table_widget.setColumnCount(cap)
        self.dp_table_widget.setHorizontalHeaderLabels([str(i) for i in range(cap)])
        self.dp_table_widget.setVerticalHeaderLabels([str(i+1) for i in range(H)])

        # 设置固定列宽，避免自动调整列宽导致的卡顿
        header = self.dp_table_widget.horizontalHeader()
        header.setDefaultSectionSize(80)          # 可根据需要调整宽度
        header.setSectionResizeMode(QHeaderView.Fixed)  # 禁止用户拖动列宽（可选）

        self.dp_table_widget.setUpdatesEnabled(False)
        for t in range(H):
            for s in range(cap):
                item = QTableWidgetItem(f'{self.dp_table[t, s]:.2f}')
                item.setFont(self.font())
                self.dp_table_widget.setItem(t, s, item)
        self.dp_table_widget.setUpdatesEnabled(True)
    def on_save_dp(self):
        fname, _ = QFileDialog.getSaveFileName(self, '保存DP表', '', 'DP表文件 (*.dp)')
        if fname:
            with open(fname, 'wb') as f:
                pickle.dump(self.dp_table, f)
            QMessageBox.information(self, '保存成功', f'已保存到 {fname}')

    def on_load_dp(self):
        fname, _ = QFileDialog.getOpenFileName(self, '导入DP表', '', 'DP表文件 (*.dp)')
        if fname:
            with open(fname, 'rb') as f:
                self.dp_table = pickle.load(f)
            self.theta = self.dp_table.copy()
            self.show_dp_table()
            QMessageBox.information(self, '导入成功', f'已导入 {fname}')

    def show_decision_details(self):
        if not self.decision_details:
            return
        self.detail_table.setRowCount(len(self.decision_details))
        self.detail_table.setColumnCount(7)
        self.detail_table.setHorizontalHeaderLabels(['天数','z','阈值','最优动作','买卖商品','DP利润','Baseline利润'])
        for i, row in enumerate(self.decision_details):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setFont(self.font())
                self.detail_table.setItem(i, j, item)
        self.detail_table.resizeColumnsToContents()
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def get_action_detail(self, spread, self_p, best_p):
        # 返回买卖商品名和数量
        idx = int(np.argmax(spread))
        buy_price = self_p[idx]
        sell_price = best_p[idx]
        name = self.cfg.product_names[idx] if idx < len(self.cfg.product_names) else f'商品{idx+1}'
        return f'{name} 买入价:{buy_price:.2f} 卖出价:{sell_price:.2f}'

    def on_manual_action(self):
        if self.theta is None:
            QMessageBox.warning(self, '未生成阈值策略', '请先生成阈值策略！')
            return
        n = len(self.cfg.product_names)
        self_p = []
        best_p = []
        for i in range(n):
            buy_item = self.price_table.item(i, 0)
            sell_item = self.price_table.item(i, 1)
            try:
                buy = float(buy_item.text()) if buy_item else 0
                sell = float(sell_item.text()) if sell_item else 0
            except:
                buy, sell = 0, 0
            self_p.append(buy)
            best_p.append(sell)
        self_p = np.array(self_p)
        best_p = np.array(best_p)
        spread = best_p - self_p
        z = float(np.max(spread))
        t = self.day_spin.value() - 1
        shelf = self.cfg.shelf_init if t == 0 else 0
        S_after = min(self.cfg.shelf_cap, shelf + self.cfg.shelf_replenish_per_day)
        th = float(self.theta[t, S_after])
        if S_after == 0:
            msg = f'今日可用容量为0，无法操作。'
        elif z >= th:
            idx = int(np.argmax(spread))
            name = self.cfg.product_names[idx] if idx < len(self.cfg.product_names) else f'商品{idx+1}'
            msg = f'最优动作: ALL-IN（满仓做）\nspread最大值z={z:.2f} >= 阈值theta={th:.2f}\n建议全部买入并卖出套利。\n商品:{name} 买入价:{self_p[idx]:.2f} 卖出价:{best_p[idx]:.2f}'
        else:
            msg = f'最优动作: 不操作\nspread最大值z={z:.2f} < 阈值theta={th:.2f}\n建议观望。'
        self.manual_result.setText(msg)

    def on_auto_fill(self):
        n = len(self.cfg.product_names)
        agents = 1 + self.cfg.n_friends
        rng_seed = int(time.time()) if getattr(self.cfg, 'time', None) else self.cfg.seed
        rng = np.random.default_rng(rng_seed)
        price_pool = dp.load_price_data()
        prices = dp.sample_prices_with_noise(price_pool, agents * n, rng).reshape((agents, n))
        self_p = prices[0]
        best_p = prices.max(axis=0)
        for i in range(n):
            self.price_table.setItem(i, 0, QTableWidgetItem(f'{self_p[i]:.2f}'))
            self.price_table.setItem(i, 1, QTableWidgetItem(f'{best_p[i]:.2f}'))

    def on_product_names_changed(self):
        # 获取最新商品名列表
        w = self.product_names_widget
        names = [x.strip() for x in w.text().split(',') if x.strip()]
        if not names:
            names = [f'商品{i+1}' for i in range(10)]
        self.cfg.product_names = names
        # 刷新人工输入表格
        self.price_table.setRowCount(len(names))
        for i, name in enumerate(names):
            self.price_table.setVerticalHeaderItem(i, QTableWidgetItem(name))
        # 清空旧数据
        for i in range(len(names)):
            self.price_table.setItem(i, 0, QTableWidgetItem(''))
            self.price_table.setItem(i, 1, QTableWidgetItem(''))

class DPAutoTestThread(QThread):
    finished = pyqtSignal(object, list, list, list, object)
    def __init__(self, cfg, theta, dp_module):
        super().__init__()
        self.cfg = cfg
        self.theta = theta
        self.dp = dp_module
    def run(self):
        decision_details = []
        n = len(self.cfg.product_names)
        agents = 1 + self.cfg.n_friends
        rng_seed = int(time.time()) if getattr(self.cfg, 'time', None) else self.cfg.seed
        rng = np.random.default_rng(rng_seed)
        shelf_dp = int(self.cfg.shelf_init)
        shelf_base = int(self.cfg.shelf_init)
        dp_profits = []
        base_profits = []
        price_pool = self.dp.load_price_data()
        for t in range(self.cfg.horizon_days):
            shelf_dp = min(self.cfg.shelf_cap, shelf_dp + self.cfg.shelf_replenish_per_day)
            shelf_base = min(self.cfg.shelf_cap, shelf_base + self.cfg.shelf_replenish_per_day)
            prices = self.dp.sample_prices_with_noise(price_pool, agents * n, rng).reshape((agents, n))
            self_p = prices[0]
            best_p = prices.max(axis=0)
            spread = best_p - self_p
            z_val = float(spread.max())
            S_after = shelf_dp
            th = float(self.theta[t, S_after])
            if S_after > 0 and z_val >= th:
                profit_dp = float(S_after) * z_val
                shelf_dp = 0
                action = 'ALL-IN'
                idx = int(np.argmax(spread))
                buy_price = self_p[idx]
                sell_price = best_p[idx]
                name = self.cfg.product_names[idx] if idx < len(self.cfg.product_names) else f'商品{idx+1}'
                action_detail = f'{name} 买入价:{buy_price:.2f} 卖出价:{sell_price:.2f}'
            else:
                profit_dp = 0
                action = '观望'
                action_detail = ''
            dp_profits.append(profit_dp)
            if shelf_base > 0:
                profit_base = float(shelf_base) * z_val
                shelf_base = 0
            else:
                profit_base = 0
            base_profits.append(profit_base)
            decision_details.append([t+1, z_val, th, action, action_detail, profit_dp, profit_base])
        self.finished.emit(self.theta, decision_details, dp_profits, base_profits, price_pool)
