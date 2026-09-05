# -*- coding: utf-8 -*-
"""
PyYoloGUI - YOLO 检测训练工具 (UI 分离版本 v4)
基于 main_detect.ui 布局 + yolo_detect_professional_4.py 业务逻辑

文件结构:
    app_v4.py - 主程序（业务逻辑）
    main_detect.ui - UI 设计文件（可选，删除后自动使用代码构建）
"""

import os
import sys
import json
import time
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from PyQt5.QtGui import QPixmap, QImage, QKeyEvent
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QComboBox, QProgressBar, QTextEdit,
    QGroupBox, QFormLayout, QMessageBox, QSplitter, QSizePolicy, QScrollArea
)

# 尝试导入 ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("警告: 未安装 ultralytics 库，请使用 pip install ultralytics 安装")

# 尝试导入 torch，用于获取 CUDA 设备
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ------------------------- 全局样式 -------------------------

# 现代化扁平风格样式表
APP_STYLE = """
/* 全局样式 */
QWidget {
    background-color: #f5f5f5;
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    font-size: 10pt;
}

/* 主窗口 */
QMainWindow {
    background-color: #ffffff;
}

/* 标签页 */
QTabWidget::pane {
    border: 1px solid #e0e0e0;
    background-color: #ffffff;
    border-radius: 4px;
    margin-top: -1px;
}

QTabWidget::tab-bar {
    alignment: left;
}

QTabBar::tab {
    background-color: #f5f5f5;
    color: #666666;
    padding: 10px 50px;
    min-width: 80px;
    margin-right: 4px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 11pt;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #1976d2;
    font-weight: bold;
    border-bottom: 2px solid #1976d2;
}

QTabBar::tab:hover:!selected {
    background-color: #e3f2fd;
}

/* 分组框 */
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #333333;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #1976d2;
}

/* 按钮 */
QPushButton {
    background-color: #1976d2;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 10pt;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1565c0;
}

QPushButton:pressed {
    background-color: #0d47a1;
}

QPushButton:disabled {
    background-color: #bdbdbd;
    color: #757575;
}

/* 次要按钮 */
QPushButton[secondary="true"] {
    background-color: #ffffff;
    color: #1976d2;
    border: 1px solid #1976d2;
}

QPushButton[secondary="true"]:hover {
    background-color: #e3f2fd;
}

/* 危险按钮 */
QPushButton[danger="true"] {
    background-color: #f44336;
}

QPushButton[danger="true"]:hover {
    background-color: #d32f2f;
}

/* 输入框 */
QLineEdit {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 10pt;
}

QLineEdit:focus {
    border-color: #1976d2;
}

QLineEdit:disabled {
    background-color: #f5f5f5;
}

/* 文本编辑框 */
QTextEdit {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 6px;
    font-family: "Consolas", "Microsoft YaHei", monospace;
    font-size: 9pt;
}

QTextEdit:focus {
    border-color: #1976d2;
}

/* 下拉框 */
QComboBox {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 10pt;
}

QComboBox:hover {
    border-color: #1976d2;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    selection-background-color: #e3f2fd;
    selection-color: #1976d2;
}

/* 标签 */
QLabel {
    background-color: transparent;
    color: #333333;
    font-size: 10pt;
}

/* 单选按钮 */
QRadioButton {
    spacing: 6px;
    color: #333333;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #bdbdbd;
    border-radius: 8px;
    background-color: #ffffff;
}

QRadioButton::indicator:checked {
    background-color: #1976d2;
    border-color: #1976d2;
}

/* 复选框 */
QCheckBox {
    spacing: 6px;
    color: #333333;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #bdbdbd;
    border-radius: 3px;
    background-color: #ffffff;
}

QCheckBox::indicator:checked {
    background-color: #1976d2;
    border-color: #1976d2;
}

/* 进度条 */
QProgressBar {
    background-color: #e0e0e0;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    text-align: center;
    height: 20px;
    color: #ffffff;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #1976d2;
    border-radius: 3px;
}

/* 拆分器 */
QSplitter::handle {
    background-color: #e0e0e0;
}

QSplitter::handle:horizontal {
    width: 4px;
}

QSplitter::handle:vertical {
    height: 4px;
}

QSplitter::handle:hover {
    background-color: #1976d2;
}

/* 表单布局 */
QFormLayout {
    spacing: 10px;
}

QFormLayout label {
    color: #666666;
    font-weight: bold;
}

/* 状态栏 */
QStatusBar {
    background-color: #f5f5f5;
    color: #666666;
}

/* 滚动区域 */
QScrollArea {
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    background-color: #ffffff;
}

/* 工具提示 */
QToolTip {
    background-color: #333333;
    color: #ffffff;
    border: none;
    padding: 4px 8px;
}

/* 消息框 */
QMessageBox {
    background-color: #ffffff;
}

QMessageBox QLabel {
    color: #333333;
}

QMessageBox QPushButton {
    min-width: 80px;
}
"""

def get_available_devices():
    """返回可用设备列表，如 ['cpu', '0', '1']"""
    devices = ['cpu']
    if TORCH_AVAILABLE and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(str(i))
    return devices

def cv2_to_qpixmap(img_bgr):
    """将 OpenCV BGR 图像转换为 QPixmap"""
    if img_bgr is None:
        return QPixmap()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def draw_labels_on_image(img_bgr, label_path, class_names=None):
    """根据 YOLO 格式标签文件在图像上绘制标签 - 样式与模型检测一致"""
    if not os.path.exists(label_path):
        return img_bgr.copy()
    img = img_bgr.copy()
    h, w = img.shape[:2]
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        width = float(parts[3]) * w
        height = float(parts[4]) * h
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        # 使用与模型检测一致的样式
        cls_name = str(cls_id) if class_names is None else class_names.get(cls_id, str(cls_id))
        # 框颜色: 红色 (与模型检测默认颜色一致)
        color = (255, 0, 0)
        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
        # 绘制标签背景
        (text_width, text_height), baseline = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        # 绘制标签文字
        cv2.putText(img, cls_name, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    return img

def compute_iou(box1, box2):
    """计算两个框的 IoU，box 格式为 [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def compute_difference_score(results1, results2):
    """计算两个检测结果列表的差异分数（平均 IoU）"""
    if not results1 or not results2:
        return 0.0
    boxes1 = []
    boxes2 = []
    for r in results1:
        if r.boxes is not None:
            for box in r.boxes.xyxy.cpu().numpy():
                boxes1.append(box)
    for r in results2:
        if r.boxes is not None:
            for box in r.boxes.xyxy.cpu().numpy():
                boxes2.append(box)
    if not boxes1 or not boxes2:
        return 0.0
    matched = set()
    total_iou = 0.0
    count = 0
    for b1 in boxes1:
        best_iou = 0
        best_idx = -1
        for i, b2 in enumerate(boxes2):
            if i in matched:
                continue
            iou = compute_iou(b1, b2)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0:
            matched.add(best_idx)
            total_iou += best_iou
            count += 1
    return total_iou / count if count > 0 else 0.0


# ------------------------- 工作线程 -------------------------

class DetectWorker(QThread):
    """检测工作线程"""
    progress_update = pyqtSignal(int, int)
    image_update = pyqtSignal(object, object, int)
    info_update = pyqtSignal(str)
    finished_detect = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            model1_path = self.params.get('model1_path')
            model2_path = self.params.get('model2_path')
            img_dir = self.params.get('img_dir')
            label_dir = self.params.get('label_dir')
            batch_mode = self.params.get('batch_mode')
            detection_mode = self.params.get('detection_mode')
            conf = self.params.get('conf', 0.25)
            device = self.params.get('device', 'cpu')
            iou = self.params.get('iou', 0.45)
            imgsz = self.params.get('imgsz', 640)
            max_det = self.params.get('max_det', 300)
            classes = self.params.get('classes', None)
            quantize = self.params.get('quantize', 16)
            augment = self.params.get('augment', False)
            stream = self.params.get('stream', False)
            current_index = self.params.get('current_index', 0)

            model1 = None
            model2 = None
            if detection_mode in ['single', 'single_label', 'dual']:
                if not model1_path or not os.path.exists(model1_path):
                    raise Exception("模型1路径无效")
                model1 = YOLO(model1_path)
            if detection_mode == 'dual':
                if not model2_path or not os.path.exists(model2_path):
                    raise Exception("模型2路径无效")
                model2 = YOLO(model2_path)

            if not os.path.isdir(img_dir):
                raise Exception("图片路径无效")
            img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
            img_files.sort()
            if not img_files:
                raise Exception("图片目录中没有图片")
            total_images = len(img_files)

            if batch_mode:
                indices = range(total_images)
            else:
                if current_index >= total_images:
                    current_index = 0
                indices = [current_index]

            results_cache = []
            for idx, img_idx in enumerate(indices):
                if not self._is_running:
                    break
                img_name = img_files[img_idx]
                img_path = os.path.join(img_dir, img_name)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    self.error_signal.emit(f"无法读取图片: {img_path}")
                    continue

                if model1 is not None:
                    t0 = time.time()
                    results1 = model1.predict(
                        source=img_bgr, conf=conf, device=device, iou=iou,
                        imgsz=imgsz, max_det=max_det, classes=classes,
                        quantize=quantize, augment=augment, stream=False, verbose=False
                    )
                    t1 = time.time()
                    elapsed = (t1 - t0) * 1000
                    if results1 and len(results1) > 0:
                        img1 = results1[0].plot()
                        boxes = results1[0].boxes
                        num_objects = len(boxes) if boxes is not None else 0
                        info_lines = []
                        if boxes is not None and num_objects > 0:
                            for box in boxes:
                                cls_id = int(box.cls[0])
                                conf_val = float(box.conf[0])
                                if hasattr(results1[0], 'names') and cls_id in results1[0].names:
                                    cls_name = results1[0].names[cls_id]
                                else:
                                    cls_name = str(cls_id)
                                info_lines.append(f"{cls_name} ({conf_val:.2%})")
                        info_text = f"图片名称: {img_name}\n检测耗时: {elapsed:.2f} ms\n检测到 {num_objects} 个目标\n" + "\n".join(info_lines)
                    else:
                        img1 = img_bgr.copy()
                        info_text = f"图片名称: {img_name}\n检测耗时: {elapsed:.2f} ms\n检测到 0 个目标"
                else:
                    img1 = img_bgr.copy()
                    info_text = f"图片名称: {img_name}\n未使用模型"

                img2 = None
                if detection_mode == 'dual' and model2 is not None:
                    t0 = time.time()
                    results2 = model2.predict(
                        source=img_bgr, conf=conf, device=device, iou=iou,
                        imgsz=imgsz, max_det=max_det, classes=classes,
                        quantize=quantize, augment=augment, stream=False, verbose=False
                    )
                    t1 = time.time()
                    elapsed2 = (t1 - t0) * 1000
                    if results2 and len(results2) > 0:
                        img2 = results2[0].plot()
                        diff_score = compute_difference_score(results1 if 'results1' in locals() else [], results2)
                        info_text += f"\n模型2耗时: {elapsed2:.2f} ms\n差异分数: {diff_score:.1f}"
                    else:
                        img2 = img_bgr.copy()
                elif detection_mode == 'single_label':
                    if label_dir and os.path.isdir(label_dir):
                        label_name = os.path.splitext(img_name)[0] + '.txt'
                        label_path = os.path.join(label_dir, label_name)
                        class_names = None
                        if model1 is not None and hasattr(model1, 'names'):
                            class_names = model1.names
                        img2 = draw_labels_on_image(img_bgr, label_path, class_names)
                    else:
                        img2 = img_bgr.copy()
                else:
                    img2 = img_bgr.copy()

                self.image_update.emit(img1, img2, img_idx)
                self.info_update.emit(info_text)
                self.progress_update.emit(idx + 1, total_images if batch_mode else 1)

                results_cache.append({
                    'img_name': img_name,
                    'img1': img1.copy(),
                    'img2': img2.copy(),
                    'info': info_text
                })

                if not batch_mode:
                    break

            self.finished_detect.emit(results_cache)
        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))


class TrainWorker(QThread):
    """训练工作线程"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    error_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            import io
            from contextlib import redirect_stdout, redirect_stderr

            model_path = self.params.get('model_path')
            data_yaml = self.params.get('data_yaml')
            project = self.params.get('project')
            name = self.params.get('name')
            epochs = self.params.get('epochs', 100)
            imgsz = self.params.get('imgsz', 640)
            workers = self.params.get('workers', 8)
            batch = self.params.get('batch', 16)
            device = self.params.get('device', 'cpu')
            hflip = self.params.get('hflip', 0.5)
            vflip = self.params.get('vflip', 0.0)
            degrees = self.params.get('degrees', 0.0)
            mosaic = self.params.get('mosaic', 1.0)
            mixup = self.params.get('mixup', 0.0)
            hsv_h = self.params.get('hsv_h', 0.015)
            hsv_s = self.params.get('hsv_s', 0.7)
            hsv_v = self.params.get('hsv_v', 0.4)
            box = self.params.get('box', 7.5)
            cls = self.params.get('cls', 0.5)
            dfl = self.params.get('dfl', 1.5)
            cache = self.params.get('cache', False)
            patience = self.params.get('patience', 100)

            train_args = {
                'model': model_path, 'data': data_yaml, 'epochs': epochs,
                'imgsz': imgsz, 'workers': workers, 'batch': batch,
                'device': device, 'hflip': hflip, 'vflip': vflip,
                'degrees': degrees, 'mosaic': mosaic, 'mixup': mixup,
                'hsv_h': hsv_h, 'hsv_s': hsv_s, 'hsv_v': hsv_v,
                'box': box, 'cls': cls, 'dfl': dfl, 'cache': cache,
                'patience': patience, 'project': project, 'name': name,
                'exist_ok': True, 'verbose': True,
            }

            config_dir = os.path.join(project, name)
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, 'train_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(train_args, f, ensure_ascii=False, indent=4)

            class StreamToLogger:
                def __init__(self, signal):
                    self.signal = signal
                def write(self, message):
                    if message and message.strip():
                        self.signal.emit(message)
                def flush(self):
                    pass

            logger = StreamToLogger(self.log_signal)
            with redirect_stdout(logger), redirect_stderr(logger):
                model = YOLO(model_path)
                model.train(**train_args)

            self.finished_signal.emit(True, "训练完成")
        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))
            self.finished_signal.emit(False, str(e))


class ExportWorker(QThread):
    """模型导出工作线程"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    error_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            import io
            from contextlib import redirect_stdout, redirect_stderr

            model_path = self.params.get('model_path')
            export_format = self.params.get('format', 'onnx')
            half = self.params.get('half', False)
            device = self.params.get('device', 'cpu')
            imgsz = self.params.get('imgsz', 640)

            class StreamToLogger:
                def __init__(self, signal):
                    self.signal = signal
                def write(self, message):
                    if message and message.strip():
                        self.signal.emit(message)
                def flush(self):
                    pass

            logger = StreamToLogger(self.log_signal)
            with redirect_stdout(logger), redirect_stderr(logger):
                model = YOLO(model_path)
                model.export(format=export_format, half=half, device=device, imgsz=imgsz)

            self.finished_signal.emit(True, "导出成功")
        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))
            self.finished_signal.emit(False, str(e))


# ------------------------- 检测标签页 -------------------------

class DetectTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_index = 0
        self.img_files = []
        self.results_cache = []
        self.detect_worker = None
        self.init_ui()
        self.connect_signals()
        self.setFocusPolicy(Qt.StrongFocus)

    def init_ui(self):
        """按照 main_detect.ui 布局创建界面"""
        main_layout = QHBoxLayout(self)

        # 使用 QSplitter 分割三个区域
        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # 左侧：模型1结果
        left_group = QGroupBox("模型1检测结果")
        left_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        left_layout = QVBoxLayout(left_group)
        self.label_img1 = QLabel()
        self.label_img1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.label_img1.setAlignment(Qt.AlignCenter)
        self.label_img1.setMinimumSize(400, 400)
        self.label_img1.setStyleSheet("border: 1px solid gray;")
        left_layout.addWidget(self.label_img1)
        splitter.addWidget(left_group)

        # 中间：模型2结果
        mid_group = QGroupBox("模型2检测结果/标签")
        mid_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        mid_layout = QVBoxLayout(mid_group)
        self.label_img2 = QLabel()
        self.label_img2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.label_img2.setAlignment(Qt.AlignCenter)
        self.label_img2.setMinimumSize(400, 400)
        self.label_img2.setStyleSheet("border: 1px solid gray;")
        mid_layout.addWidget(self.label_img2)
        splitter.addWidget(mid_group)

        # 右侧：参数面板
        right_panel = QWidget()
        right_panel.setObjectName("right_panel")
        right_panel.setMaximumWidth(450)
        right_layout = QVBoxLayout(right_panel)

        # 模型1路径
        model1_layout = QHBoxLayout()
        model1_layout.addWidget(QLabel("模型1路径:"))
        self.edit_model1 = QLineEdit()
        self.edit_model1.setPlaceholderText("未加载模型/模型地址")
        model1_layout.addWidget(self.edit_model1)
        self.btn_model1_select = QPushButton("选择模型")
        model1_layout.addWidget(self.btn_model1_select)
        right_layout.addLayout(model1_layout)

        # 模型2路径
        model2_layout = QHBoxLayout()
        model2_layout.addWidget(QLabel("模型2路径:"))
        self.edit_model2 = QLineEdit()
        self.edit_model2.setPlaceholderText("未加载模型/模型地址")
        model2_layout.addWidget(self.edit_model2)
        self.btn_model2_select = QPushButton("选择模型")
        model2_layout.addWidget(self.btn_model2_select)
        right_layout.addLayout(model2_layout)

        # 图片路径
        img_layout = QHBoxLayout()
        img_layout.addWidget(QLabel("图片路径:"))
        self.edit_img_path = QLineEdit()
        self.edit_img_path.setPlaceholderText("未加载图片/图片地址")
        img_layout.addWidget(self.edit_img_path)
        self.btn_img_select = QPushButton("选择图片")
        img_layout.addWidget(self.btn_img_select)
        right_layout.addLayout(img_layout)

        # 标签路径
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("标签路径:"))
        self.edit_label_path = QLineEdit()
        self.edit_label_path.setPlaceholderText("未加载标签/标签地址")
        label_layout.addWidget(self.edit_label_path)
        self.btn_label_select = QPushButton("选择标签目录")
        label_layout.addWidget(self.btn_label_select)
        right_layout.addLayout(label_layout)

        # 检测模式单选
        self.radio_batch = QRadioButton("批量检测")
        self.radio_single = QRadioButton("单张检测")
        self.radio_single.setChecked(True)
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(self.radio_batch)
        batch_layout.addWidget(self.radio_single)
        right_layout.addLayout(batch_layout)

        # 检查模式单选
        self.radio_mode_single = QRadioButton("单模型检测")
        self.radio_mode_single_label = QRadioButton("单模型＋标签")
        self.radio_mode_dual = QRadioButton("双模型检测")
        self.radio_mode_single.setChecked(True)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.radio_mode_single)
        mode_layout.addWidget(self.radio_mode_single_label)
        mode_layout.addWidget(self.radio_mode_dual)
        right_layout.addLayout(mode_layout)

        # 参数网格
        form_layout = QFormLayout()
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        form_layout.addRow("置信度", self.spin_conf)

        self.combo_device = QComboBox()
        for dev in get_available_devices():
            self.combo_device.addItem(dev)
        form_layout.addRow("设备:", self.combo_device)

        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.0, 1.0)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.45)
        form_layout.addRow("IOU:", self.spin_iou)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 4096)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        form_layout.addRow("imgsz:", self.spin_imgsz)

        self.spin_max_det = QSpinBox()
        self.spin_max_det.setRange(1, 1000)
        self.spin_max_det.setValue(300)
        form_layout.addRow("max_det:", self.spin_max_det)

        self.edit_classes = QLineEdit()
        self.edit_classes.setPlaceholderText("例如: 0,1,2 (留空表示所有)")
        form_layout.addRow("classes:", self.edit_classes)

        self.spin_quantize = QSpinBox()
        self.spin_quantize.setRange(0, 255)
        self.spin_quantize.setValue(0)
        self.spin_quantize.setToolTip("量化参数值 (0表示不量化)")
        form_layout.addRow("quantize:", self.spin_quantize)

        self.check_augment = QCheckBox("augment")
        self.check_stream = QCheckBox("stream")
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.check_augment)
        checkbox_layout.addWidget(self.check_stream)
        form_layout.addRow("选项:", checkbox_layout)

        right_layout.addLayout(form_layout)

        # 开始检测按钮
        self.btn_start_detect = QPushButton("开始检测")
        right_layout.addWidget(self.btn_start_detect)

        # 上一页/下一页
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("上一页")
        self.btn_next = QPushButton("下一页")
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        right_layout.addLayout(nav_layout)

        # 进度条
        self.label_progress = QLabel("0/0")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(self.label_progress)
        progress_layout.addWidget(self.progress_bar)
        right_layout.addLayout(progress_layout)

        # 检测结果信息
        right_layout.addWidget(QLabel("检测结果"))
        self.text_info = QTextEdit()
        self.text_info.setReadOnly(True)
        right_layout.addWidget(self.text_info)

        right_panel.setLayout(right_layout)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 400, 350])
        main_layout.addWidget(splitter)

    def connect_signals(self):
        """连接信号和槽"""
        self.btn_model1_select.clicked.connect(lambda: self.select_model(1))
        self.btn_model2_select.clicked.connect(lambda: self.select_model(2))
        self.btn_img_select.clicked.connect(self.select_image_dir)
        self.btn_label_select.clicked.connect(self.select_label_dir)
        self.btn_start_detect.clicked.connect(self.start_detection)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)

        # 连接按钮组
        self.batch_group = QButtonGroup(self)
        self.batch_group.addButton(self.radio_batch)
        self.batch_group.addButton(self.radio_single)
        
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.radio_mode_single)
        self.mode_group.addButton(self.radio_mode_single_label)
        self.mode_group.addButton(self.radio_mode_dual)

    def select_model(self, model_num):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch模型 (*.pt);;所有文件 (*)")
        if file_path:
            if model_num == 1:
                self.edit_model1.setText(file_path)
            else:
                self.edit_model2.setText(file_path)

    def select_image_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if dir_path:
            self.edit_img_path.setText(dir_path)
            self.load_image_list()

    def select_label_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择标签目录")
        if dir_path:
            self.edit_label_path.setText(dir_path)

    def load_image_list(self):
        img_dir = self.edit_img_path.text()
        if not img_dir or not os.path.isdir(img_dir):
            return
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
        self.img_files.sort()
        self.current_index = 0
        self.results_cache = []
        self.update_progress_label()
        if self.img_files:
            self.show_image_at_index(0)

    def update_progress_label(self):
        total = len(self.img_files)
        self.label_progress.setText(f"{self.current_index + 1 if self.img_files else 0}/{total}")

    def show_image_at_index(self, index):
        if not self.img_files or index < 0 or index >= len(self.img_files):
            return
        self.current_index = index
        img_path = os.path.join(self.edit_img_path.text(), self.img_files[index])
        img_bgr = cv2.imread(img_path)
        if img_bgr is not None:
            pixmap = cv2_to_qpixmap(img_bgr)
            self.label_img1.setPixmap(pixmap.scaled(self.label_img1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.label_img2.setPixmap(pixmap.scaled(self.label_img2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.update_progress_label()

    def prev_image(self):
        if self.img_files:
            if self.current_index > 0:
                if self.results_cache:
                    idx = max(0, self.current_index - 1)
                    if idx < len(self.results_cache):
                        self.on_image_update(self.results_cache[idx]['img1'], self.results_cache[idx]['img2'], idx)
                        self.on_info_update(self.results_cache[idx]['info'])
                        return
                self.show_image_at_index(self.current_index - 1)
            else:
                # 第一张时跳到最后一张
                if self.results_cache:
                    idx = len(self.results_cache) - 1
                    if idx >= 0:
                        self.on_image_update(self.results_cache[idx]['img1'], self.results_cache[idx]['img2'], idx)
                        self.on_info_update(self.results_cache[idx]['info'])
                        return
                self.show_image_at_index(len(self.img_files) - 1)

    def next_image(self):
        if self.img_files:
            if self.current_index < len(self.img_files) - 1:
                if self.results_cache:
                    idx = min(len(self.img_files)-1, self.current_index + 1)
                    if idx < len(self.results_cache):
                        self.on_image_update(self.results_cache[idx]['img1'], self.results_cache[idx]['img2'], idx)
                        self.on_info_update(self.results_cache[idx]['info'])
                        return
                self.show_image_at_index(self.current_index + 1)
            else:
                # 最后一张时回到第一张
                if self.results_cache:
                    idx = 0
                    if idx < len(self.results_cache):
                        self.on_image_update(self.results_cache[idx]['img1'], self.results_cache[idx]['img2'], idx)
                        self.on_info_update(self.results_cache[idx]['info'])
                        return
                self.show_image_at_index(0)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key_Right:
            self.next_image()
        else:
            super().keyPressEvent(event)

    def get_detection_params(self):
        params = {
            'model1_path': self.edit_model1.text().strip(),
            'model2_path': self.edit_model2.text().strip(),
            'img_dir': self.edit_img_path.text().strip(),
            'label_dir': self.edit_label_path.text().strip(),
            'batch_mode': self.radio_batch.isChecked(),
            'conf': self.spin_conf.value(),
            'device': self.combo_device.currentText(),
            'iou': self.spin_iou.value(),
            'imgsz': self.spin_imgsz.value(),
            'max_det': self.spin_max_det.value(),
            'classes': None if not self.edit_classes.text().strip() else [int(c) for c in self.edit_classes.text().split(',') if c.strip().isdigit()],
            'quantize': self.spin_quantize.value(),
            'augment': self.check_augment.isChecked(),
            'stream': self.check_stream.isChecked(),
            'current_index': self.current_index
        }
        if self.radio_mode_single.isChecked():
            params['detection_mode'] = 'single'
        elif self.radio_mode_single_label.isChecked():
            params['detection_mode'] = 'single_label'
        elif self.radio_mode_dual.isChecked():
            params['detection_mode'] = 'dual'
        else:
            params['detection_mode'] = 'single'
        return params

    def start_detection(self):
        if self.detect_worker and self.detect_worker.isRunning():
            QMessageBox.warning(self, "警告", "检测正在进行中")
            return
        params = self.get_detection_params()
        if not params['img_dir'] or not os.path.isdir(params['img_dir']):
            QMessageBox.warning(self, "警告", "请选择有效的图片目录")
            return
        if params['detection_mode'] in ['single', 'single_label', 'dual'] and not params['model1_path']:
            QMessageBox.warning(self, "警告", "请选择模型1路径")
            return
        if params['detection_mode'] == 'dual' and not params['model2_path']:
            QMessageBox.warning(self, "警告", "请选择模型2路径")
            return
        if params['detection_mode'] == 'single_label' and not params['label_dir']:
            QMessageBox.warning(self, "警告", "请选择标签目录")
            return

        self.results_cache = []
        self.text_info.clear()
        self.progress_bar.setValue(0)

        self.detect_worker = DetectWorker(params)
        self.detect_worker.image_update.connect(self.on_image_update)
        self.detect_worker.info_update.connect(self.on_info_update)
        self.detect_worker.progress_update.connect(self.on_progress_update)
        self.detect_worker.finished_detect.connect(self.on_detect_finished)
        self.detect_worker.error_signal.connect(self.on_detect_error)
        self.detect_worker.start()
        self.btn_start_detect.setEnabled(False)

    def on_image_update(self, img1, img2, index):
        if img1 is not None:
            pix1 = cv2_to_qpixmap(img1)
            self.label_img1.setPixmap(pix1.scaled(self.label_img1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if img2 is not None:
            pix2 = cv2_to_qpixmap(img2)
            self.label_img2.setPixmap(pix2.scaled(self.label_img2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.current_index = index
        self.update_progress_label()

    def on_info_update(self, info):
        self.text_info.setPlainText(info)

    def on_progress_update(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.label_progress.setText(f"{current}/{total}")

    def on_detect_finished(self, results_cache):
        self.results_cache = results_cache
        self.btn_start_detect.setEnabled(True)
        if results_cache:
            idx = min(self.current_index, len(results_cache)-1)
            self.on_image_update(results_cache[idx]['img1'], results_cache[idx]['img2'], idx)
            self.on_info_update(results_cache[idx]['info'])

    def on_detect_error(self, error_msg):
        self.btn_start_detect.setEnabled(True)
        QMessageBox.critical(self, "错误", f"检测出错: {error_msg}")


# ------------------------- 训练标签页 -------------------------

class TrainTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.train_worker = None
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """按照 main_detect.ui 布局创建界面"""
        main_layout = QVBoxLayout(self)

        # 模型文件
        model_group = QGroupBox("模型文件")
        model_layout = QHBoxLayout(model_group)
        model_layout.addWidget(QLabel("模型路径:"))
        self.edit_model_path_train = QLineEdit()
        self.edit_model_path_train.setPlaceholderText("未加载模型/模型地址")
        model_layout.addWidget(self.edit_model_path_train)
        self.btn_train_model_select = QPushButton("选择模型")
        model_layout.addWidget(self.btn_train_model_select)
        main_layout.addWidget(model_group)

        # 数据集配置
        data_group = QGroupBox("数据集配置")
        data_layout = QHBoxLayout(data_group)
        data_layout.addWidget(QLabel("yaml配置:"))
        self.edit_yaml_path = QLineEdit()
        self.edit_yaml_path.setPlaceholderText("未加载配置/yaml地址")
        data_layout.addWidget(self.edit_yaml_path)
        self.btn_yaml_select = QPushButton("选择配置")
        data_layout.addWidget(self.btn_yaml_select)
        main_layout.addWidget(data_group)

        # 项目设置 - 添加滚动条
        project_group = QGroupBox("项目设置")
        project_form = QFormLayout(project_group)
        self.edit_project_dir = QLineEdit()
        self.edit_project_dir.setPlaceholderText("未加载目录/目录地址")
        btn_project_dir = QPushButton("选择目录")
        btn_project_dir.clicked.connect(self.select_project_dir)
        proj_dir_layout = QHBoxLayout()
        proj_dir_layout.addWidget(self.edit_project_dir)
        proj_dir_layout.addWidget(btn_project_dir)
        project_form.addRow("项目目录:", proj_dir_layout)

        self.edit_experiment_name = QLineEdit()
        self.edit_experiment_name.setText("exp")
        project_form.addRow("实验名称:", self.edit_experiment_name)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(100)
        project_form.addRow("训练轮数:", self.spin_epochs)

        self.spin_imgsz_train = QSpinBox()
        self.spin_imgsz_train.setRange(32, 4096)
        self.spin_imgsz_train.setValue(640)
        project_form.addRow("图片尺寸:", self.spin_imgsz_train)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 64)
        self.spin_workers.setValue(8)
        project_form.addRow("工作进程:", self.spin_workers)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 512)
        self.spin_batch.setValue(16)
        project_form.addRow("批次大小:", self.spin_batch)

        self.combo_device_train = QComboBox()
        for dev in get_available_devices():
            self.combo_device_train.addItem(dev)
        project_form.addRow("设备:", self.combo_device_train)
        
        # 为项目设置添加滚动条
        project_scroll = QScrollArea()
        project_scroll.setWidget(project_group)
        project_scroll.setWidgetResizable(True)
        project_scroll.setMaximumHeight(300)
        main_layout.addWidget(project_scroll)

        # 数据增强 - 添加滚动条
        aug_group = QGroupBox("数据增强")
        aug_form = QFormLayout(aug_group)
        self.spin_hflip = QDoubleSpinBox()
        self.spin_hflip.setRange(0.0, 1.0)
        self.spin_hflip.setSingleStep(0.05)
        self.spin_hflip.setValue(0.5)
        aug_form.addRow("水平翻转概率:", self.spin_hflip)

        self.spin_vflip = QDoubleSpinBox()
        self.spin_vflip.setRange(0.0, 1.0)
        self.spin_vflip.setSingleStep(0.05)
        self.spin_vflip.setValue(0.0)
        aug_form.addRow("垂直翻转概率:", self.spin_vflip)

        self.spin_degrees = QDoubleSpinBox()
        self.spin_degrees.setRange(0.0, 180.0)
        self.spin_degrees.setSingleStep(1.0)
        self.spin_degrees.setValue(0.0)
        aug_form.addRow("旋转角度:", self.spin_degrees)

        self.spin_mosaic = QDoubleSpinBox()
        self.spin_mosaic.setRange(0.0, 1.0)
        self.spin_mosaic.setSingleStep(0.1)
        self.spin_mosaic.setValue(1.0)
        aug_form.addRow("mosaic增强:", self.spin_mosaic)

        self.spin_mixup = QDoubleSpinBox()
        self.spin_mixup.setRange(0.0, 1.0)
        self.spin_mixup.setSingleStep(0.05)
        self.spin_mixup.setValue(0.0)
        aug_form.addRow("mixup增强:", self.spin_mixup)
        
        # 为数据增强添加滚动条
        aug_scroll = QScrollArea()
        aug_scroll.setWidget(aug_group)
        aug_scroll.setWidgetResizable(True)
        aug_scroll.setMaximumHeight(200)
        main_layout.addWidget(aug_scroll)

        # 颜色空间增强 - 添加滚动条
        color_group = QGroupBox("颜色空间增强")
        color_form = QFormLayout(color_group)
        self.spin_hsv_h = QDoubleSpinBox()
        self.spin_hsv_h.setRange(0.0, 1.0)
        self.spin_hsv_h.setSingleStep(0.005)
        self.spin_hsv_h.setValue(0.01)
        color_form.addRow("hsv-hue偏移:", self.spin_hsv_h)

        self.spin_hsv_s = QDoubleSpinBox()
        self.spin_hsv_s.setRange(0.0, 1.0)
        self.spin_hsv_s.setSingleStep(0.05)
        self.spin_hsv_s.setValue(0.7)
        color_form.addRow("hsv-saturation偏移:", self.spin_hsv_s)

        self.spin_hsv_v = QDoubleSpinBox()
        self.spin_hsv_v.setRange(0.0, 1.0)
        self.spin_hsv_v.setSingleStep(0.05)
        self.spin_hsv_v.setValue(0.4)
        color_form.addRow("hsv-value偏移:", self.spin_hsv_v)
        
        # 为颜色空间增强添加滚动条
        color_scroll = QScrollArea()
        color_scroll.setWidget(color_group)
        color_scroll.setWidgetResizable(True)
        color_scroll.setMaximumHeight(150)
        main_layout.addWidget(color_scroll)

        # 损失权重 - 添加滚动条
        loss_group = QGroupBox("损失权重")
        loss_form = QFormLayout(loss_group)
        self.spin_box_loss = QDoubleSpinBox()
        self.spin_box_loss.setRange(0.0, 100.0)
        self.spin_box_loss.setValue(7.5)
        loss_form.addRow("box损失权重:", self.spin_box_loss)

        self.spin_cls_loss = QDoubleSpinBox()
        self.spin_cls_loss.setRange(0.0, 100.0)
        self.spin_cls_loss.setValue(0.5)
        loss_form.addRow("class损失权重:", self.spin_cls_loss)

        self.spin_dfl_loss = QDoubleSpinBox()
        self.spin_dfl_loss.setRange(0.0, 100.0)
        self.spin_dfl_loss.setValue(1.5)
        loss_form.addRow("DFL损失权重:", self.spin_dfl_loss)
        
        # 为损失权重添加滚动条
        loss_scroll = QScrollArea()
        loss_scroll.setWidget(loss_group)
        loss_scroll.setWidgetResizable(True)
        loss_scroll.setMaximumHeight(150)
        main_layout.addWidget(loss_scroll)

        # 其他设置
        other_group = QGroupBox("其他设置")
        other_layout = QHBoxLayout(other_group)
        self.check_cache = QCheckBox("缓存数据")
        other_layout.addWidget(self.check_cache)
        other_layout.addWidget(QLabel("耐心轮数:"))
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(1, 1000)
        self.spin_patience.setValue(100)
        other_layout.addWidget(self.spin_patience)
        other_layout.addStretch()
        main_layout.addWidget(other_group)

        # 快速预设按钮
        preset_layout = QHBoxLayout()
        self.btn_preset_default = QPushButton("默认")
        self.btn_preset_aggressive = QPushButton("激进增强")
        self.btn_preset_light = QPushButton("轻量训练")
        preset_layout.addWidget(self.btn_preset_default)
        preset_layout.addWidget(self.btn_preset_aggressive)
        preset_layout.addWidget(self.btn_preset_light)
        preset_layout.addStretch()
        main_layout.addLayout(preset_layout)

        # 训练日志
        main_layout.addWidget(QLabel("训练日志:"))
        self.text_log_train = QTextEdit()
        self.text_log_train.setReadOnly(True)
        main_layout.addWidget(self.text_log_train)

        # 按钮组
        btn_layout = QHBoxLayout()
        self.btn_start_train = QPushButton("开始训练")
        self.btn_stop_train = QPushButton("停止训练")
        self.btn_stop_train.setEnabled(False)
        self.btn_export_config = QPushButton("导出训练配置")
        self.btn_import_config = QPushButton("导入训练配置")
        self.btn_clear_log = QPushButton("清空日志")
        btn_layout.addWidget(self.btn_start_train)
        btn_layout.addWidget(self.btn_stop_train)
        btn_layout.addWidget(self.btn_export_config)
        btn_layout.addWidget(self.btn_import_config)
        btn_layout.addWidget(self.btn_clear_log)
        main_layout.addLayout(btn_layout)

    def connect_signals(self):
        """连接信号和槽"""
        self.btn_train_model_select.clicked.connect(self.select_model)
        self.btn_yaml_select.clicked.connect(self.select_yaml)
        self.btn_start_train.clicked.connect(self.start_training)
        self.btn_stop_train.clicked.connect(self.stop_training)
        self.btn_export_config.clicked.connect(self.export_config)
        self.btn_import_config.clicked.connect(self.import_config)
        self.btn_clear_log.clicked.connect(self.clear_log)
        self.btn_preset_default.clicked.connect(self.preset_default)
        self.btn_preset_aggressive.clicked.connect(self.preset_aggressive)
        self.btn_preset_light.clicked.connect(self.preset_light)

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch模型 (*.pt);;所有文件 (*)")
        if file_path:
            self.edit_model_path_train.setText(file_path)

    def select_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择YAML配置文件", "", "YAML文件 (*.yaml *.yml);;所有文件 (*)")
        if file_path:
            self.edit_yaml_path.setText(file_path)

    def select_project_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择项目目录")
        if dir_path:
            self.edit_project_dir.setText(dir_path)

    def get_train_params(self):
        return {
            'model_path': self.edit_model_path_train.text().strip(),
            'data_yaml': self.edit_yaml_path.text().strip(),
            'project': self.edit_project_dir.text().strip(),
            'name': self.edit_experiment_name.text().strip(),
            'epochs': self.spin_epochs.value(),
            'imgsz': self.spin_imgsz_train.value(),
            'workers': self.spin_workers.value(),
            'batch': self.spin_batch.value(),
            'device': self.combo_device_train.currentText(),
            'hflip': self.spin_hflip.value(),
            'vflip': self.spin_vflip.value(),
            'degrees': self.spin_degrees.value(),
            'mosaic': self.spin_mosaic.value(),
            'mixup': self.spin_mixup.value(),
            'hsv_h': self.spin_hsv_h.value(),
            'hsv_s': self.spin_hsv_s.value(),
            'hsv_v': self.spin_hsv_v.value(),
            'box': self.spin_box_loss.value(),
            'cls': self.spin_cls_loss.value(),
            'dfl': self.spin_dfl_loss.value(),
            'cache': self.check_cache.isChecked(),
            'patience': self.spin_patience.value()
        }

    def start_training(self):
        if self.train_worker and self.train_worker.isRunning():
            QMessageBox.warning(self, "警告", "训练正在进行中")
            return
        params = self.get_train_params()
        if not params['model_path']:
            QMessageBox.warning(self, "警告", "请选择模型文件")
            return
        if not params['data_yaml']:
            QMessageBox.warning(self, "警告", "请选择数据集yaml文件")
            return
        if not params['project']:
            QMessageBox.warning(self, "警告", "请选择项目目录")
            return

        self.text_log_train.clear()
        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)

        self.train_worker = TrainWorker(params)
        self.train_worker.log_signal.connect(self.append_log)
        self.train_worker.finished_signal.connect(self.on_train_finished)
        self.train_worker.error_signal.connect(self.on_train_error)
        self.train_worker.start()

    def stop_training(self):
        if self.train_worker and self.train_worker.isRunning():
            QMessageBox.information(self, "提示", "训练一旦开始无法中途停止，请等待当前epoch完成或强制关闭程序")
        else:
            self.btn_stop_train.setEnabled(False)

    def export_config(self):
        params = self.get_train_params()
        project = params['project']
        name = params['name']
        if not project or not name:
            QMessageBox.warning(self, "警告", "请先填写项目目录和实验名称")
            return
        config_dir = os.path.join(project, name)
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, 'train_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=4)
        QMessageBox.information(self, "成功", f"配置已导出到: {config_path}")

    def import_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "JSON文件 (*.json);;所有文件 (*)")
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            self.edit_model_path_train.setText(params.get('model_path', ''))
            self.edit_yaml_path.setText(params.get('data_yaml', ''))
            self.edit_project_dir.setText(params.get('project', ''))
            self.edit_experiment_name.setText(params.get('name', 'exp'))
            self.spin_epochs.setValue(params.get('epochs', 100))
            self.spin_imgsz_train.setValue(params.get('imgsz', 640))
            self.spin_workers.setValue(params.get('workers', 8))
            self.spin_batch.setValue(params.get('batch', 16))
            dev = params.get('device', 'cpu')
            index = self.combo_device_train.findText(dev)
            if index >= 0:
                self.combo_device_train.setCurrentIndex(index)
            self.spin_hflip.setValue(params.get('hflip', 0.5))
            self.spin_vflip.setValue(params.get('vflip', 0.0))
            self.spin_degrees.setValue(params.get('degrees', 0.0))
            self.spin_mosaic.setValue(params.get('mosaic', 1.0))
            self.spin_mixup.setValue(params.get('mixup', 0.0))
            self.spin_hsv_h.setValue(params.get('hsv_h', 0.015))
            self.spin_hsv_s.setValue(params.get('hsv_s', 0.7))
            self.spin_hsv_v.setValue(params.get('hsv_v', 0.4))
            self.spin_box_loss.setValue(params.get('box', 7.5))
            self.spin_cls_loss.setValue(params.get('cls', 0.5))
            self.spin_dfl_loss.setValue(params.get('dfl', 1.5))
            self.check_cache.setChecked(params.get('cache', False))
            self.spin_patience.setValue(params.get('patience', 100))
            QMessageBox.information(self, "成功", "配置导入成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败: {str(e)}")

    def clear_log(self):
        self.text_log_train.clear()

    def append_log(self, message):
        self.text_log_train.append(message)

    def on_train_finished(self, success, message):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        if success:
            self.append_log("=== 训练完成 ===")
        else:
            self.append_log(f"=== 训练失败: {message} ===")

    def on_train_error(self, error_msg):
        self.append_log(f"错误: {error_msg}")

    def preset_default(self):
        self.spin_hflip.setValue(0.5)
        self.spin_vflip.setValue(0.0)
        self.spin_degrees.setValue(0.0)
        self.spin_mosaic.setValue(1.0)
        self.spin_mixup.setValue(0.0)
        self.spin_hsv_h.setValue(0.015)
        self.spin_hsv_s.setValue(0.7)
        self.spin_hsv_v.setValue(0.4)
        self.spin_box_loss.setValue(7.5)
        self.spin_cls_loss.setValue(0.5)
        self.spin_dfl_loss.setValue(1.5)

    def preset_aggressive(self):
        self.spin_hflip.setValue(0.8)
        self.spin_vflip.setValue(0.3)
        self.spin_degrees.setValue(15.0)
        self.spin_mosaic.setValue(1.0)
        self.spin_mixup.setValue(0.2)
        self.spin_hsv_h.setValue(0.03)
        self.spin_hsv_s.setValue(0.9)
        self.spin_hsv_v.setValue(0.6)

    def preset_light(self):
        self.spin_hflip.setValue(0.0)
        self.spin_vflip.setValue(0.0)
        self.spin_degrees.setValue(0.0)
        self.spin_mosaic.setValue(0.0)
        self.spin_mixup.setValue(0.0)
        self.spin_hsv_h.setValue(0.0)
        self.spin_hsv_s.setValue(0.0)
        self.spin_hsv_v.setValue(0.0)
        self.spin_box_loss.setValue(5.0)
        self.spin_cls_loss.setValue(0.3)
        self.spin_dfl_loss.setValue(1.0)


# ------------------------- 格式转换标签页 -------------------------

class FormatTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.export_worker = None
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """按照 main_detect.ui 布局创建界面"""
        main_layout = QVBoxLayout(self)

        # 模型路径
        model_layout = QHBoxLayout()
        self.label_format_model_path = QLabel("模型路径:")
        model_layout.addWidget(self.label_format_model_path)
        self.edit_model_path_format = QLineEdit()
        self.edit_model_path_format.setPlaceholderText("未加载模型/模型地址")
        model_layout.addWidget(self.edit_model_path_format)
        self.btn_format_model_select = QPushButton("选择模型")
        model_layout.addWidget(self.btn_format_model_select)
        main_layout.addLayout(model_layout)

        # 导出格式
        format_group = QGroupBox("导出格式")
        format_layout = QHBoxLayout(format_group)
        self.radio_onnx = QRadioButton("onnx")
        self.radio_engine = QRadioButton("engine")
        self.radio_onnx.setChecked(True)
        format_layout.addWidget(self.radio_onnx)
        format_layout.addWidget(self.radio_engine)
        main_layout.addWidget(format_group)

        # 参数设置
        param_layout = QFormLayout()
        self.label_format_half = QLabel("half:")
        self.check_half = QCheckBox()
        param_layout.addRow(self.label_format_half, self.check_half)

        self.label_format_device = QLabel("device:")
        self.combo_device_format = QComboBox()
        for dev in get_available_devices():
            self.combo_device_format.addItem(dev)
        param_layout.addRow(self.label_format_device, self.combo_device_format)

        self.label_format_imgsz = QLabel("imgsz:")
        self.spin_imgsz_format = QSpinBox()
        self.spin_imgsz_format.setRange(32, 4096)
        self.spin_imgsz_format.setValue(640)
        param_layout.addRow(self.label_format_imgsz, self.spin_imgsz_format)
        main_layout.addLayout(param_layout)

        # 开始转换按钮
        self.btn_export = QPushButton("转换开始")
        main_layout.addWidget(self.btn_export)

        # 转换日志
        self.label_export_log = QLabel("转换日志:")
        main_layout.addWidget(self.label_export_log)
        self.text_log_format = QTextEdit()
        self.text_log_format.setReadOnly(True)
        main_layout.addWidget(self.text_log_format)

    def connect_signals(self):
        """连接信号和槽"""
        self.btn_format_model_select.clicked.connect(self.select_model)
        self.btn_export.clicked.connect(self.start_export)

        self.format_group_btn = QButtonGroup(self)
        self.format_group_btn.addButton(self.radio_onnx)
        self.format_group_btn.addButton(self.radio_engine)

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch模型 (*.pt);;所有文件 (*)")
        if file_path:
            self.edit_model_path_format.setText(file_path)

    def get_export_params(self):
        return {
            'model_path': self.edit_model_path_format.text().strip(),
            'format': 'onnx' if self.radio_onnx.isChecked() else 'engine',
            'half': self.check_half.isChecked(),
            'device': self.combo_device_format.currentText(),
            'imgsz': self.spin_imgsz_format.value()
        }

    def start_export(self):
        if self.export_worker and self.export_worker.isRunning():
            QMessageBox.warning(self, "警告", "转换正在进行中")
            return
        params = self.get_export_params()
        if not params['model_path']:
            QMessageBox.warning(self, "警告", "请选择模型文件")
            return
        self.text_log_format.clear()
        self.btn_export.setEnabled(False)
        self.export_worker = ExportWorker(params)
        self.export_worker.log_signal.connect(self.append_log)
        self.export_worker.finished_signal.connect(self.on_export_finished)
        self.export_worker.error_signal.connect(self.on_export_error)
        self.export_worker.start()

    def append_log(self, message):
        self.text_log_format.append(message)

    def on_export_finished(self, success, message):
        self.btn_export.setEnabled(True)
        if success:
            self.append_log("=== 导出成功 ===")
        else:
            self.append_log(f"=== 导出失败: {message} ===")

    def on_export_error(self, error_msg):
        self.append_log(f"错误: {error_msg}")


# ------------------------- 主窗口 -------------------------

class MainWindow(QMainWindow):
    CONFIG_FILE = "app_config.json"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO26权重训练-测试工具")
        self.resize(1400, 983)
        
        # 应用全局样式
        app = QApplication.instance()
        app.setStyleSheet(APP_STYLE)

        # 创建标签页
        self.tabs = QTabWidget()
        self.detect_tab = DetectTab()
        self.train_tab = TrainTab()
        self.format_tab = FormatTab()
        
        self.tabs.addTab(self.detect_tab, "detect")
        self.tabs.addTab(self.train_tab, "train")
        self.tabs.addTab(self.format_tab, "format")
        
        self.setCentralWidget(self.tabs)

        # 加载配置
        self.load_config()

    def closeEvent(self, event):
        self.save_config()
        if (hasattr(self, 'detect_tab') and self.detect_tab.detect_worker and self.detect_tab.detect_worker.isRunning()) or \
           (hasattr(self, 'train_tab') and self.train_tab.train_worker and self.train_tab.train_worker.isRunning()) or \
           (hasattr(self, 'format_tab') and self.format_tab.export_worker and self.format_tab.export_worker.isRunning()):
            reply = QMessageBox.question(self, '确认退出', '有任务正在进行，确定要退出吗？',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def save_config(self):
        if hasattr(self, 'detect_tab') and hasattr(self, 'train_tab') and hasattr(self, 'format_tab'):
            config = {
                'detect': {
                    'model1': self.detect_tab.edit_model1.text(),
                    'model2': self.detect_tab.edit_model2.text(),
                    'img_dir': self.detect_tab.edit_img_path.text(),
                    'label_dir': self.detect_tab.edit_label_path.text(),
                    'conf': self.detect_tab.spin_conf.value(),
                    'iou': self.detect_tab.spin_iou.value(),
                    'imgsz': self.detect_tab.spin_imgsz.value(),
                    'max_det': self.detect_tab.spin_max_det.value(),
                    'classes': self.detect_tab.edit_classes.text(),
                    'quantize': self.detect_tab.spin_quantize.value(),
                    'augment': self.detect_tab.check_augment.isChecked(),
                    'stream': self.detect_tab.check_stream.isChecked(),
                    'batch_mode': self.detect_tab.radio_batch.isChecked(),
                    'mode_single': self.detect_tab.radio_mode_single.isChecked(),
                    'mode_single_label': self.detect_tab.radio_mode_single_label.isChecked(),
                    'mode_dual': self.detect_tab.radio_mode_dual.isChecked(),
                    'device': self.detect_tab.combo_device.currentText(),
                },
                'train': self.train_tab.get_train_params(),
                'format': {
                    'model_path': self.format_tab.edit_model_path_format.text(),
                    'format_onnx': self.format_tab.radio_onnx.isChecked(),
                    'half': self.format_tab.check_half.isChecked(),
                    'device': self.format_tab.combo_device_format.currentText(),
                    'imgsz': self.format_tab.spin_imgsz_format.value(),
                }
            }
            try:
                with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"保存配置失败: {e}")

    def load_config(self):
        if not os.path.exists(self.CONFIG_FILE):
            return
        try:
            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if hasattr(self, 'detect_tab'):
                detect_cfg = config.get('detect', {})
                self.detect_tab.edit_model1.setText(detect_cfg.get('model1', ''))
                self.detect_tab.edit_model2.setText(detect_cfg.get('model2', ''))
                self.detect_tab.edit_img_path.setText(detect_cfg.get('img_dir', ''))
                self.detect_tab.edit_label_path.setText(detect_cfg.get('label_dir', ''))
                self.detect_tab.spin_conf.setValue(detect_cfg.get('conf', 0.25))
                self.detect_tab.spin_iou.setValue(detect_cfg.get('iou', 0.45))
                self.detect_tab.spin_imgsz.setValue(detect_cfg.get('imgsz', 640))
                self.detect_tab.spin_max_det.setValue(detect_cfg.get('max_det', 300))
                self.detect_tab.edit_classes.setText(detect_cfg.get('classes', ''))
                self.detect_tab.spin_quantize.setValue(detect_cfg.get('quantize', 0))
                self.detect_tab.check_augment.setChecked(detect_cfg.get('augment', False))
                self.detect_tab.check_stream.setChecked(detect_cfg.get('stream', False))
                self.detect_tab.radio_batch.setChecked(detect_cfg.get('batch_mode', False))
                self.detect_tab.radio_single.setChecked(not detect_cfg.get('batch_mode', False))
                self.detect_tab.radio_mode_single.setChecked(detect_cfg.get('mode_single', True))
                self.detect_tab.radio_mode_single_label.setChecked(detect_cfg.get('mode_single_label', False))
                self.detect_tab.radio_mode_dual.setChecked(detect_cfg.get('mode_dual', False))
                dev = detect_cfg.get('device', 'cpu')
                idx = self.detect_tab.combo_device.findText(dev)
                if idx >= 0:
                    self.detect_tab.combo_device.setCurrentIndex(idx)
                if self.detect_tab.edit_img_path.text():
                    self.detect_tab.load_image_list()

            if hasattr(self, 'train_tab'):
                train_cfg = config.get('train', {})
                if train_cfg:
                    self.train_tab.edit_model_path_train.setText(train_cfg.get('model_path', ''))
                    self.train_tab.edit_yaml_path.setText(train_cfg.get('data_yaml', ''))
                    self.train_tab.edit_project_dir.setText(train_cfg.get('project', ''))
                    self.train_tab.edit_experiment_name.setText(train_cfg.get('name', 'exp'))
                    self.train_tab.spin_epochs.setValue(train_cfg.get('epochs', 100))
                    self.train_tab.spin_imgsz_train.setValue(train_cfg.get('imgsz', 640))
                    self.train_tab.spin_workers.setValue(train_cfg.get('workers', 8))
                    self.train_tab.spin_batch.setValue(train_cfg.get('batch', 16))
                    dev = train_cfg.get('device', 'cpu')
                    idx = self.train_tab.combo_device_train.findText(dev)
                    if idx >= 0:
                        self.train_tab.combo_device_train.setCurrentIndex(idx)
                    self.train_tab.spin_hflip.setValue(train_cfg.get('hflip', 0.5))
                    self.train_tab.spin_vflip.setValue(train_cfg.get('vflip', 0.0))
                    self.train_tab.spin_degrees.setValue(train_cfg.get('degrees', 0.0))
                    self.train_tab.spin_mosaic.setValue(train_cfg.get('mosaic', 1.0))
                    self.train_tab.spin_mixup.setValue(train_cfg.get('mixup', 0.0))
                    self.train_tab.spin_hsv_h.setValue(train_cfg.get('hsv_h', 0.015))
                    self.train_tab.spin_hsv_s.setValue(train_cfg.get('hsv_s', 0.7))
                    self.train_tab.spin_hsv_v.setValue(train_cfg.get('hsv_v', 0.4))
                    self.train_tab.spin_box_loss.setValue(train_cfg.get('box', 7.5))
                    self.train_tab.spin_cls_loss.setValue(train_cfg.get('cls', 0.5))
                    self.train_tab.spin_dfl_loss.setValue(train_cfg.get('dfl', 1.5))
                    self.train_tab.check_cache.setChecked(train_cfg.get('cache', False))
                    self.train_tab.spin_patience.setValue(train_cfg.get('patience', 100))

            if hasattr(self, 'format_tab'):
                format_cfg = config.get('format', {})
                self.format_tab.edit_model_path_format.setText(format_cfg.get('model_path', ''))
                self.format_tab.radio_onnx.setChecked(format_cfg.get('format_onnx', True))
                self.format_tab.radio_engine.setChecked(not format_cfg.get('format_onnx', True))
                self.format_tab.check_half.setChecked(format_cfg.get('half', False))
                dev = format_cfg.get('device', 'cpu')
                idx = self.format_tab.combo_device_format.findText(dev)
                if idx >= 0:
                    self.format_tab.combo_device_format.setCurrentIndex(idx)
                self.format_tab.spin_imgsz_format.setValue(format_cfg.get('imgsz', 640))
        except Exception as e:
            print(f"加载配置失败: {e}")


# ------------------------- 程序入口 -------------------------

if __name__ == '__main__':
    app = QApplication(sys.argv)
    if not ULTRALYTICS_AVAILABLE:
        QMessageBox.warning(None, "警告", "未检测到 ultralytics 库，请先安装: pip install ultralytics")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
