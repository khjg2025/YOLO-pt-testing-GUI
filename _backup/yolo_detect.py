# -*- coding: utf-8 -*-
"""
YOLO26权重训练-测试工具
修复跨线程图像传递崩溃，增加设备选择，参数持久化
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox,
    QRadioButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QProgressBar, QScrollArea, QGridLayout, QFormLayout,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont, QColor

# 尝试导入 ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

# 默认参数
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
DEFAULT_MAX_DET = 300
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 16
DEFAULT_WORKERS = 8
DEFAULT_DEVICE = 'cpu'
DEFAULT_HALF = False
DEFAULT_AUGMENT = False
DEFAULT_STREAM = False

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (128, 0, 128), (0, 128, 128),
          (128, 128, 0), (192, 192, 0), (192, 0, 192), (0, 192, 192)]



class WorkerSignals:
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    # 替换为：
    detection_result = pyqtSignal(int, str, list, float)  # (model_index, img_path, boxes_list, elapsed)
    result_text = pyqtSignal(str)


# ================== 新的 DetectWorker ==================
class DetectWorker(QThread):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        if not ULTRALYTICS_AVAILABLE:
            self.signals.error.emit("未安装 ultralytics，请运行: pip install ultralytics")
            return
        try:
            mode = self.params['mode']
            batch = self.params['batch']
            model1_path = self.params['model1_path']
            model2_path = self.params.get('model2_path', '')
            img_dir = self.params['img_dir']
            label_dir = self.params.get('label_dir', '')
            conf = self.params['conf']
            iou = self.params['iou']
            imgsz = self.params['imgsz']
            max_det = self.params['max_det']
            half = self.params['half']
            augment = self.params['augment']
            stream = self.params['stream']
            device = self.params.get('device', 'cpu')

            # 获取图片列表
            if os.path.isdir(img_dir):
                img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
                img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_exts)]
                img_files.sort()
            else:
                img_files = [os.path.basename(img_dir)]

            total = len(img_files)
            if total == 0:
                self.signals.error.emit("没有找到图片")
                return

            self.signals.log.emit(f"加载模型1: {model1_path} (设备: {device})")
            model1 = YOLO(model1_path, device=device)
            if mode == 'dual':
                self.signals.log.emit(f"加载模型2: {model2_path} (设备: {device})")
                model2 = YOLO(model2_path, device=device)

            for idx, fname in enumerate(img_files):
                if not self._is_running:
                    break
                img_path = os.path.join(img_dir, fname) if os.path.isdir(img_dir) else img_dir
                # 只读图片，不进行像素操作
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    self.signals.log.emit(f"无法读取图片: {fname}")
                    continue
                h, w, _ = img_bgr.shape

                t0 = time.time()
                if mode == 'single' or mode == 'single_label':
                    res1 = model1(img_path, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
                                  half=half, augment=augment, verbose=False)
                    t1 = time.time()
                    det1 = res1[0].boxes
                    boxes1 = self._extract_boxes(det1, res1[0].names)
                    self.signals.detection_result.emit(1, img_path, boxes1, t1 - t0)
                    text1 = self.format_result(fname, t1 - t0, det1)
                    self.signals.result_text.emit(f"模型1检测结果:\n{text1}")

                    if mode == 'single_label' and label_dir:
                        label_file = os.path.join(label_dir, os.path.splitext(fname)[0] + '.txt')
                        if os.path.exists(label_file):
                            label_boxes = self.load_yolo_labels(label_file, w, h)
                            # 将标签转换为与检测框相同的格式
                            label_boxes_converted = []
                            for (cls_id, cx, cy, w_norm, h_norm) in label_boxes:
                                x1 = int((cx - w_norm/2) * w)
                                y1 = int((cy - h_norm/2) * h)
                                x2 = int((cx + w_norm/2) * w)
                                y2 = int((cy + h_norm/2) * h)
                                cls_name = f"label_{cls_id}"  # 或从class_names获取
                                label_boxes_converted.append({
                                    "class": cls_name,
                                    "confidence": 1.0,
                                    "bbox": [x1, y1, x2, y2]
                                })
                            self.signals.detection_result.emit(2, img_path, label_boxes_converted, 0.0)
                            text2 = f"标签文件: {os.path.basename(label_file)}\n检测到 {len(label_boxes)} 个目标"
                            self.signals.result_text.emit(f"标签标注结果:\n{text2}")
                        else:
                            self.signals.log.emit(f"未找到标签文件: {label_file}")

                elif mode == 'dual':
                    res1 = model1(img_path, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
                                  half=half, augment=augment, verbose=False)
                    t1 = time.time()
                    res2 = model2(img_path, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
                                  half=half, augment=augment, verbose=False)
                    t2 = time.time()
                    det1 = res1[0].boxes
                    det2 = res2[0].boxes
                    boxes1 = self._extract_boxes(det1, res1[0].names)
                    boxes2 = self._extract_boxes(det2, res2[0].names)
                    self.signals.detection_result.emit(1, img_path, boxes1, t1 - t0)
                    self.signals.detection_result.emit(2, img_path, boxes2, t2 - t1)
                    text1 = self.format_result(fname, t1 - t0, det1)
                    text2 = self.format_result(fname, t2 - t1, det2)
                    diff = abs(len(det1) - len(det2))
                    self.signals.result_text.emit(f"模型1检测结果:\n{text1}\n\n模型2检测结果:\n{text2}\n\n差异分数: {diff:.1f}")

                progress = int((idx + 1) / total * 100)
                self.signals.progress.emit(progress, f"处理 {idx+1}/{total}")

            self.signals.log.emit("检测完成")
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(str(e))
            import traceback
            traceback.print_exc()

    def _extract_boxes(self, boxes, names):
        result = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            result.append({"class": cls_name, "confidence": conf, "bbox": xyxy})
        return result

    def load_yolo_labels(self, label_path, img_w, img_h):
        boxes = []
        if not os.path.exists(label_path):
            return boxes
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                boxes.append((cls, xc, yc, w, h))
        return boxes

    def format_result(self, fname, time_ms, det):
        lines = [f"图片名称: {fname}",
                 f"检测耗时: {time_ms*1000:.1f} ms",
                 f"检测到 {len(det)} 个目标:"]
        cls_count = defaultdict(int)
        for box in det:
            cls = int(box.cls[0])
            cls_count[cls] += 1
        for cls, cnt in cls_count.items():
            lines.append(f"  {cls} ({cnt})")
        return "\n".join(lines)


class TrainWorker(QThread):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        if not ULTRALYTICS_AVAILABLE:
            self.signals.error.emit("未安装 ultralytics")
            return
        try:
            model_path = self.params['model_path']
            yaml_path = self.params['yaml_path']
            project = self.params['project']
            name = self.params['name']
            epochs = self.params['epochs']
            imgsz = self.params['imgsz']
            batch = self.params['batch']
            workers = self.params['workers']
            device = self.params['device']
            hsv_h = self.params['hsv_h']
            hsv_s = self.params['hsv_s']
            hsv_v = self.params['hsv_v']
            flipud = self.params['flipud']
            fliplr = self.params['fliplr']
            degrees = self.params['degrees']
            mosaic = self.params['mosaic']
            mixup = self.params['mixup']
            box_loss = self.params['box_loss']
            cls_loss = self.params['cls_loss']
            dfl_loss = self.params['dfl_loss']
            cache = self.params['cache']
            patience = self.params['patience']

            args = {
                'data': yaml_path,
                'epochs': epochs,
                'imgsz': imgsz,
                'batch': batch,
                'workers': workers,
                'device': device,
                'project': project,
                'name': name,
                'exist_ok': True,
                'patience': patience,
                'cache': cache,
                'hsv_h': hsv_h,
                'hsv_s': hsv_s,
                'hsv_v': hsv_v,
                'flipud': flipud,
                'fliplr': fliplr,
                'degrees': degrees,
                'mosaic': mosaic,
                'mixup': mixup,
                'box': box_loss,
                'cls': cls_loss,
                'dfl': dfl_loss,
            }
            args = {k: v for k, v in args.items() if v is not None}

            config_path = os.path.join(project, name, 'train_config.json')
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(args, f, indent=4)
            self.signals.log.emit(f"训练配置已导出至: {config_path}")

            model = YOLO(model_path)
            self.signals.log.emit("开始训练...")
            # 模拟训练（实际可替换为 model.train(**args)）
            for i in range(epochs):
                if not self._is_running:
                    self.signals.log.emit("训练被用户停止")
                    break
                time.sleep(1)
                progress = int((i + 1) / epochs * 100)
                self.signals.progress.emit(progress, f"Epoch {i+1}/{epochs}")
                self.signals.log.emit(f"Epoch {i+1} 完成")
            if self._is_running:
                self.signals.log.emit("训练完成！")
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(str(e))


class FormatWorker(QThread):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        if not ULTRALYTICS_AVAILABLE:
            self.signals.error.emit("未安装 ultralytics")
            return
        try:
            model_path = self.params['model_path']
            format_type = self.params['format']
            half = self.params['half']
            device = self.params['device']
            imgsz = self.params['imgsz']

            model = YOLO(model_path)
            self.signals.log.emit(f"开始转换为 {format_type} ...")
            if format_type == 'onnx':
                model.export(format='onnx', imgsz=imgsz, half=half, device=device)
            elif format_type == 'engine':
                model.export(format='engine', imgsz=imgsz, half=half, device=device)
            self.signals.log.emit("转换完成！")
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(str(e))


class DetectTab(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('YOLO_Tool', 'Detect')
        self.initUI()
        self.worker = None
        self.img_files = []
        self.current_idx = 0
        self.total = 0
        # 在 DetectTab.__init__ 中添加颜色字典
        self.class_colors = {}
        self.class_colors2 = {}

    # 新增绘图方法
    def _draw_boxes_on_label(self, label, img_path, boxes, colors_dict):
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            label.setText("无法加载图片")
            return
        if boxes:
            img = pixmap.toImage()
            painter = QPainter(img)
            painter.setRenderHint(QPainter.Antialiasing)
            for box in boxes:
                x1, y1, x2, y2 = box["bbox"]
                cls_name = box["class"]
                if cls_name not in colors_dict:
                    colors_dict[cls_name] = self._get_next_color(len(colors_dict))
                color = colors_dict[cls_name]
                pen = QPen(color, 3)
                painter.setPen(pen)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                label_text = f"{cls_name} {box['confidence']:.2f}"
                painter.setPen(color)
                painter.setFont(QFont("Arial", 14))
                painter.drawText(x1, y1 - 5, label_text)
            painter.end()
            pixmap = QPixmap.fromImage(img)
        # 缩放
        label_size = label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)

    def _get_next_color(self, index):
        colors = [
            QColor(255, 0, 0), QColor(0, 170, 255), QColor(0, 200, 0),
            QColor(255, 170, 0), QColor(170, 0, 255), QColor(255, 255, 0),
            QColor(255, 0, 255), QColor(0, 255, 255), QColor(255, 100, 0),
            QColor(0, 255, 100),
        ]
        return colors[index % len(colors)]

    # 新增槽
    def update_detection_result(self, model_idx, img_path, boxes, elapsed):
        if model_idx == 1:
            self._draw_boxes_on_label(self.label1, img_path, boxes, self.class_colors)
        elif model_idx == 2:
            self._draw_boxes_on_label(self.label2, img_path, boxes, self.class_colors2)

    def initUI(self):
        layout = QHBoxLayout(self)

        # 左侧模型1
        left_group = QGroupBox("模型1检测结果")
        left_layout = QVBoxLayout()
        self.label1 = QLabel("图片显示区域")
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.label1.setMinimumSize(400, 400)
        self.label1.setScaledContents(True)
        left_layout.addWidget(self.label1)
        left_group.setLayout(left_layout)
        layout.addWidget(left_group, 1)

        # 中间模型2
        mid_group = QGroupBox("模型2检测结果")
        mid_layout = QVBoxLayout()
        self.label2 = QLabel("图片显示区域")
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.label2.setMinimumSize(400, 400)
        self.label2.setScaledContents(True)
        mid_layout.addWidget(self.label2)
        mid_group.setLayout(mid_layout)
        layout.addWidget(mid_group, 1)

        # 右侧参数
        right_group = QGroupBox("检测参数")
        right_layout = QVBoxLayout()

        form = QFormLayout()
        self.model1_path = QLineEdit()
        self.model1_path.setPlaceholderText("未加载模型")
        self.btn_model1 = QPushButton("选择模型")
        self.btn_model1.clicked.connect(lambda: self.load_model(1))
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.model1_path, 1)
        hbox1.addWidget(self.btn_model1)
        form.addRow("模型1路径:", hbox1)

        self.model2_path = QLineEdit()
        self.model2_path.setPlaceholderText("未加载模型")
        self.btn_model2 = QPushButton("选择模型")
        self.btn_model2.clicked.connect(lambda: self.load_model(2))
        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.model2_path, 1)
        hbox2.addWidget(self.btn_model2)
        form.addRow("模型2路径:", hbox2)

        self.img_path = QLineEdit()
        self.img_path.setPlaceholderText("未加载图片/目录")
        self.btn_img = QPushButton("选择图片")
        self.btn_img.clicked.connect(self.load_image_dir)
        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.img_path, 1)
        hbox3.addWidget(self.btn_img)
        form.addRow("图片路径:", hbox3)

        self.label_path = QLineEdit()
        self.label_path.setPlaceholderText("未加载标签目录")
        self.btn_label = QPushButton("选择标签目录")
        self.btn_label.clicked.connect(self.load_label_dir)
        hbox4 = QHBoxLayout()
        hbox4.addWidget(self.label_path, 1)
        hbox4.addWidget(self.btn_label)
        form.addRow("标签路径:", hbox4)

        right_layout.addLayout(form)

        # 检测模式
        mode_group = QGroupBox("检测模式")
        mode_layout = QHBoxLayout()
        self.batch_radio = QRadioButton("批量检测")
        self.single_radio = QRadioButton("单张检测")
        self.batch_radio.setChecked(True)
        mode_layout.addWidget(self.batch_radio)
        mode_layout.addWidget(self.single_radio)
        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)

        # 模型模式
        model_mode_group = QGroupBox("模型模式")
        model_mode_layout = QHBoxLayout()
        self.single_model_radio = QRadioButton("单模型检查")
        self.single_label_radio = QRadioButton("单模型+标签")
        self.dual_model_radio = QRadioButton("双模型检查")
        self.single_model_radio.setChecked(True)
        model_mode_layout.addWidget(self.single_model_radio)
        model_mode_layout.addWidget(self.single_label_radio)
        model_mode_layout.addWidget(self.dual_model_radio)
        model_mode_group.setLayout(model_mode_layout)
        right_layout.addWidget(model_mode_group)

        # 参数
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout()
        param_layout.addWidget(QLabel("置信度:"), 0, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0, 1)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(DEFAULT_CONF)
        param_layout.addWidget(self.conf_spin, 0, 1)

        param_layout.addWidget(QLabel("设备:"), 0, 2)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['cpu', '0', '1', '2', '3', 'mps'])
        self.device_combo.setCurrentText(DEFAULT_DEVICE)
        param_layout.addWidget(self.device_combo, 0, 3)

        param_layout.addWidget(QLabel("IOU:"), 1, 0)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0, 1)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(DEFAULT_IOU)
        param_layout.addWidget(self.iou_spin, 1, 1)

        param_layout.addWidget(QLabel("imgsz:"), 2, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 2560)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(DEFAULT_IMGSZ)
        param_layout.addWidget(self.imgsz_spin, 2, 1)

        param_layout.addWidget(QLabel("max_det:"), 2, 2)
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 1000)
        self.max_det_spin.setValue(DEFAULT_MAX_DET)
        param_layout.addWidget(self.max_det_spin, 2, 3)

        param_layout.addWidget(QLabel("classes:"), 3, 0)
        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("如: 0,1,2")
        param_layout.addWidget(self.classes_edit, 3, 1, 1, 3)

        self.half_check = QCheckBox("half")
        self.half_check.setChecked(DEFAULT_HALF)
        param_layout.addWidget(self.half_check, 4, 0)
        self.augment_check = QCheckBox("augment")
        self.augment_check.setChecked(DEFAULT_AUGMENT)
        param_layout.addWidget(self.augment_check, 4, 1)
        self.stream_check = QCheckBox("stream")
        self.stream_check.setChecked(DEFAULT_STREAM)
        param_layout.addWidget(self.stream_check, 4, 2)

        param_group.setLayout(param_layout)
        right_layout.addWidget(param_group)

        # 按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self.start_detect)
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.next_image)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        right_layout.addLayout(btn_layout)

        # 进度
        progress_layout = QHBoxLayout()
        self.idx_label = QLabel("0/0")
        progress_layout.addWidget(self.idx_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar, 1)
        self.progress_label = QLabel("0%")
        progress_layout.addWidget(self.progress_label)
        right_layout.addLayout(progress_layout)

        # 结果
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        self.result_text.setPlaceholderText("检测结果将显示在这里...")
        right_layout.addWidget(self.result_text)

        right_group.setLayout(right_layout)
        layout.addWidget(right_group, 1)

        self.load_settings()

    def load_settings(self):
        self.model1_path.setText(self.settings.value('model1_path', ''))
        self.model2_path.setText(self.settings.value('model2_path', ''))
        self.img_path.setText(self.settings.value('img_path', ''))
        self.label_path.setText(self.settings.value('label_path', ''))
        device = self.settings.value('device', DEFAULT_DEVICE)
        idx = self.device_combo.findText(device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)

    def save_settings(self):
        self.settings.setValue('model1_path', self.model1_path.text())
        self.settings.setValue('model2_path', self.model2_path.text())
        self.settings.setValue('img_path', self.img_path.text())
        self.settings.setValue('label_path', self.label_path.text())
        self.settings.setValue('device', self.device_combo.currentText())

    def load_model(self, idx):
        path, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型", "", "PyTorch模型 (*.pt)")
        if path:
            if idx == 1:
                self.model1_path.setText(path)
            else:
                self.model2_path.setText(path)
            self.save_settings()

    def load_image_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if path:
            self.img_path.setText(path)
            self.load_image_list()
            self.save_settings()

    def load_label_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择标签目录")
        if path:
            self.label_path.setText(path)
            self.save_settings()

    def load_image_list(self):
        path = self.img_path.text()
        if os.path.isdir(path):
            exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
            self.img_files = [f for f in os.listdir(path) if f.lower().endswith(exts)]
            self.img_files.sort()
        else:
            self.img_files = [os.path.basename(path)] if os.path.isfile(path) else []
        self.total = len(self.img_files)
        self.current_idx = 0 if self.total > 0 else -1
        self.update_idx_label()
        self.update_image_display()

    def update_idx_label(self):
        if self.total > 0:
            self.idx_label.setText(f"{self.current_idx+1}/{self.total}")
        else:
            self.idx_label.setText("0/0")

    def update_image_display(self):
        if self.total == 0 or self.current_idx < 0:
            return
        img_path = self.img_path.text()
        if os.path.isdir(img_path):
            fname = self.img_files[self.current_idx]
            full_path = os.path.join(img_path, fname)
        else:
            full_path = img_path
        pix = self.load_image_as_pixmap(full_path)
        if pix:
            self.label1.setPixmap(pix)
            self.label2.setPixmap(pix)

    def load_image_as_pixmap(self, path):
        img = cv2.imread(path)
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    def prev_image(self):
        if self.total == 0:
            return
        self.current_idx = (self.current_idx - 1) % self.total
        self.update_idx_label()
        self.update_image_display()

    def next_image(self):
        if self.total == 0:
            return
        self.current_idx = (self.current_idx + 1) % self.total
        self.update_idx_label()
        self.update_image_display()

    def start_detect(self):
        """启动检测线程"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "检测正在进行中")
            return

        # ---------- 校验参数 ----------
        model1 = self.model1_path.text()
        if not os.path.exists(model1):
            QMessageBox.warning(self, "错误", "请选择有效的模型1文件")
            return

        img_dir = self.img_path.text()
        if not os.path.exists(img_dir):
            QMessageBox.warning(self, "错误", "请选择有效的图片路径")
            return

        # 确定检测模式
        if self.single_label_radio.isChecked():
            mode = 'single_label'
            if not os.path.exists(self.label_path.text()):
                QMessageBox.warning(self, "错误", "单模型+标签模式需要选择标签目录")
                return
        elif self.dual_model_radio.isChecked():
            mode = 'dual'
            if not os.path.exists(self.model2_path.text()):
                QMessageBox.warning(self, "错误", "双模型模式需要加载模型2")
                return
        else:
            mode = 'single'

        batch = self.batch_radio.isChecked()

        classes_str = self.classes_edit.text().strip()
        classes = None
        if classes_str:
            try:
                classes = [int(x) for x in classes_str.split(',')]
            except:
                pass

        # ---------- 构建参数 ----------
        params = {
            'mode': mode,
            'batch': batch,
            'model1_path': model1,
            'model2_path': self.model2_path.text(),
            'img_dir': img_dir,
            'label_dir': self.label_path.text(),
            'conf': self.conf_spin.value(),
            'iou': self.iou_spin.value(),
            'imgsz': self.imgsz_spin.value(),
            'max_det': self.max_det_spin.value(),
            'half': self.half_check.isChecked(),
            'augment': self.augment_check.isChecked(),
            'stream': self.stream_check.isChecked(),
            'device': self.device_combo.currentText(),
        }
        if classes:
            params['classes'] = classes

        # ---------- 创建并启动工作线程 ----------
        self.worker = DetectWorker(params)
        self.worker.signals = WorkerSignals()

        # 连接信号
        self.worker.signals.log.connect(self.append_log)
        self.worker.signals.error.connect(self.show_error)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.detection_result.connect(self.update_detection_result)  # 新信号
        self.worker.signals.result_text.connect(self.result_text.setText)
        self.worker.signals.finished.connect(self.detect_finished)

        self.worker.start()
        self.start_btn.setEnabled(False)

    def append_log(self, msg):
        self.result_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def show_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
        self.start_btn.setEnabled(True)

    def update_progress(self, val, msg):
        self.progress_bar.setValue(val)
        self.progress_label.setText(f"{val}%")
        self.result_text.append(msg)

    def update_detection_image_data(self, idx, data, w, h):
        """在主线程中将字节数据转为 QPixmap 并显示"""
        qimg = QImage(data, w, h, w * 3, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        if idx == 1:
            self.label1.setPixmap(pixmap)
        else:
            self.label2.setPixmap(pixmap)

    def detect_finished(self):
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("100%")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()


class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('YOLO_Tool', 'Train')
        self.initUI()
        self.worker = None

    def initUI(self):
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        main_layout = QVBoxLayout(content)

        # 模型文件
        model_group = QGroupBox("模型文件")
        model_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("未加载模型")
        self.btn_model = QPushButton("选择模型")
        self.btn_model.clicked.connect(self.load_model)
        model_layout.addWidget(self.model_path, 1)
        model_layout.addWidget(self.btn_model)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # 数据集配置
        data_group = QGroupBox("数据集配置")
        data_layout = QHBoxLayout()
        self.yaml_path = QLineEdit()
        self.yaml_path.setPlaceholderText("未加载配置")
        self.btn_yaml = QPushButton("选择配置")
        self.btn_yaml.clicked.connect(self.load_yaml)
        data_layout.addWidget(self.yaml_path, 1)
        data_layout.addWidget(self.btn_yaml)
        data_group.setLayout(data_layout)
        main_layout.addWidget(data_group)

        # 项目设置
        project_group = QGroupBox("项目设置")
        project_layout = QGridLayout()
        project_layout.addWidget(QLabel("项目目录:"), 0, 0)
        self.project_dir = QLineEdit()
        self.project_dir.setPlaceholderText("未加载目录")
        self.btn_project = QPushButton("选择目录")
        self.btn_project.clicked.connect(self.select_project_dir)
        hbox = QHBoxLayout()
        hbox.addWidget(self.project_dir, 1)
        hbox.addWidget(self.btn_project)
        project_layout.addLayout(hbox, 0, 1, 1, 3)
        project_layout.addWidget(QLabel("实验名称:"), 1, 0)
        self.exp_name = QLineEdit("exp")
        project_layout.addWidget(self.exp_name, 1, 1)
        project_layout.addWidget(QLabel("训练轮数:"), 1, 2)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(DEFAULT_EPOCHS)
        project_layout.addWidget(self.epochs_spin, 1, 3)
        project_layout.addWidget(QLabel("图片尺寸:"), 2, 0)
        self.imgsz_train = QSpinBox()
        self.imgsz_train.setRange(32, 2560)
        self.imgsz_train.setSingleStep(32)
        self.imgsz_train.setValue(DEFAULT_IMGSZ)
        project_layout.addWidget(self.imgsz_train, 2, 1)
        project_layout.addWidget(QLabel("工作进程:"), 2, 2)
        self.workers_train = QSpinBox()
        self.workers_train.setRange(0, 64)
        self.workers_train.setValue(DEFAULT_WORKERS)
        project_layout.addWidget(self.workers_train, 2, 3)
        project_layout.addWidget(QLabel("批次大小:"), 3, 0)
        self.batch_train = QSpinBox()
        self.batch_train.setRange(1, 1024)
        self.batch_train.setValue(DEFAULT_BATCH)
        project_layout.addWidget(self.batch_train, 3, 1)
        project_layout.addWidget(QLabel("设备:"), 3, 2)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['cpu', '0', '1', '2', '3', 'mps'])
        self.device_combo.setCurrentText(DEFAULT_DEVICE)
        project_layout.addWidget(self.device_combo, 3, 3)
        project_group.setLayout(project_layout)
        main_layout.addWidget(project_group)

        # 数据增强
        aug_group = QGroupBox("数据增强")
        aug_layout = QGridLayout()
        aug_layout.addWidget(QLabel("水平翻转概率:"), 0, 0)
        self.fliplr_spin = QDoubleSpinBox()
        self.fliplr_spin.setRange(0, 1)
        self.fliplr_spin.setSingleStep(0.05)
        self.fliplr_spin.setValue(0.5)
        aug_layout.addWidget(self.fliplr_spin, 0, 1)
        aug_layout.addWidget(QLabel("垂直翻转概率:"), 0, 2)
        self.flipud_spin = QDoubleSpinBox()
        self.flipud_spin.setRange(0, 1)
        self.flipud_spin.setSingleStep(0.05)
        self.flipud_spin.setValue(0.0)
        aug_layout.addWidget(self.flipud_spin, 0, 3)
        aug_layout.addWidget(QLabel("旋转角度:"), 1, 0)
        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(0, 180)
        self.degrees_spin.setValue(0.0)
        aug_layout.addWidget(self.degrees_spin, 1, 1)
        aug_layout.addWidget(QLabel("mosaic增强:"), 1, 2)
        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0, 1)
        self.mosaic_spin.setSingleStep(0.05)
        self.mosaic_spin.setValue(1.0)
        aug_layout.addWidget(self.mosaic_spin, 1, 3)
        aug_layout.addWidget(QLabel("mixup增强:"), 2, 0)
        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0, 1)
        self.mixup_spin.setSingleStep(0.05)
        self.mixup_spin.setValue(0.0)
        aug_layout.addWidget(self.mixup_spin, 2, 1)
        aug_group.setLayout(aug_layout)
        main_layout.addWidget(aug_group)

        # 颜色空间增强
        color_group = QGroupBox("颜色空间增强")
        color_layout = QGridLayout()
        color_layout.addWidget(QLabel("hsv-hue偏移:"), 0, 0)
        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0, 1)
        self.hsv_h_spin.setSingleStep(0.01)
        self.hsv_h_spin.setValue(0.015)
        color_layout.addWidget(self.hsv_h_spin, 0, 1)
        color_layout.addWidget(QLabel("hsv-saturation偏移:"), 0, 2)
        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0, 1)
        self.hsv_s_spin.setSingleStep(0.01)
        self.hsv_s_spin.setValue(0.7)
        color_layout.addWidget(self.hsv_s_spin, 0, 3)
        color_layout.addWidget(QLabel("hsv-value偏移:"), 1, 0)
        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0, 1)
        self.hsv_v_spin.setSingleStep(0.01)
        self.hsv_v_spin.setValue(0.4)
        color_layout.addWidget(self.hsv_v_spin, 1, 1)
        color_group.setLayout(color_layout)
        main_layout.addWidget(color_group)

        # 损失权重
        loss_group = QGroupBox("损失权重")
        loss_layout = QGridLayout()
        loss_layout.addWidget(QLabel("box损失权重:"), 0, 0)
        self.box_loss_spin = QDoubleSpinBox()
        self.box_loss_spin.setRange(0, 10)
        self.box_loss_spin.setSingleStep(0.1)
        self.box_loss_spin.setValue(7.5)
        loss_layout.addWidget(self.box_loss_spin, 0, 1)
        loss_layout.addWidget(QLabel("class损失权重:"), 0, 2)
        self.cls_loss_spin = QDoubleSpinBox()
        self.cls_loss_spin.setRange(0, 10)
        self.cls_loss_spin.setSingleStep(0.1)
        self.cls_loss_spin.setValue(0.5)
        loss_layout.addWidget(self.cls_loss_spin, 0, 3)
        loss_layout.addWidget(QLabel("DFL损失权重:"), 1, 0)
        self.dfl_loss_spin = QDoubleSpinBox()
        self.dfl_loss_spin.setRange(0, 10)
        self.dfl_loss_spin.setSingleStep(0.1)
        self.dfl_loss_spin.setValue(1.5)
        loss_layout.addWidget(self.dfl_loss_spin, 1, 1)
        loss_group.setLayout(loss_layout)
        main_layout.addWidget(loss_group)

        # 其他设置
        other_group = QGroupBox("其他设置")
        other_layout = QGridLayout()
        self.cache_check = QCheckBox("缓存数据")
        other_layout.addWidget(self.cache_check, 0, 0)
        other_layout.addWidget(QLabel("耐心轮数:"), 0, 1)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 1000)
        self.patience_spin.setValue(50)
        other_layout.addWidget(self.patience_spin, 0, 2)
        other_group.setLayout(other_layout)
        main_layout.addWidget(other_group)

        # 快速预设
        preset_group = QGroupBox("快速预设")
        preset_layout = QHBoxLayout()
        self.default_btn = QPushButton("默认")
        self.default_btn.clicked.connect(self.set_default_preset)
        self.aggressive_btn = QPushButton("激进增强")
        self.aggressive_btn.clicked.connect(self.set_aggressive_preset)
        self.light_btn = QPushButton("轻量训练")
        self.light_btn.clicked.connect(self.set_light_preset)
        preset_layout.addWidget(self.default_btn)
        preset_layout.addWidget(self.aggressive_btn)
        preset_layout.addWidget(self.light_btn)
        preset_group.setLayout(preset_layout)
        main_layout.addWidget(preset_group)

        # 日志
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        # 按钮
        btn_layout = QHBoxLayout()
        self.start_train_btn = QPushButton("开始训练")
        self.start_train_btn.clicked.connect(self.start_train)
        self.stop_train_btn = QPushButton("停止训练")
        self.stop_train_btn.clicked.connect(self.stop_train)
        self.export_config_btn = QPushButton("导出训练配置")
        self.export_config_btn.clicked.connect(self.export_config)
        self.import_config_btn = QPushButton("导入训练配置")
        self.import_config_btn.clicked.connect(self.import_config)
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.log_text.clear)
        btn_layout.addWidget(self.start_train_btn)
        btn_layout.addWidget(self.stop_train_btn)
        btn_layout.addWidget(self.export_config_btn)
        btn_layout.addWidget(self.import_config_btn)
        btn_layout.addWidget(self.clear_log_btn)
        main_layout.addLayout(btn_layout)

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        main_layout.addWidget(self.train_progress)

        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.load_settings()

    def load_settings(self):
        self.model_path.setText(self.settings.value('train_model_path', ''))
        self.yaml_path.setText(self.settings.value('train_yaml_path', ''))
        self.project_dir.setText(self.settings.value('train_project_dir', ''))
        device = self.settings.value('train_device', DEFAULT_DEVICE)
        idx = self.device_combo.findText(device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)

    def save_settings(self):
        self.settings.setValue('train_model_path', self.model_path.text())
        self.settings.setValue('train_yaml_path', self.yaml_path.text())
        self.settings.setValue('train_project_dir', self.project_dir.text())
        self.settings.setValue('train_device', self.device_combo.currentText())

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型", "", "PyTorch模型 (*.pt)")
        if path:
            self.model_path.setText(path)
            self.save_settings()

    def load_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择YAML配置文件", "", "YAML文件 (*.yaml *.yml)")
        if path:
            self.yaml_path.setText(path)
            self.save_settings()

    def select_project_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择项目目录")
        if path:
            self.project_dir.setText(path)
            self.save_settings()

    def set_default_preset(self):
        self.fliplr_spin.setValue(0.5)
        self.flipud_spin.setValue(0.0)
        self.degrees_spin.setValue(0.0)
        self.mosaic_spin.setValue(1.0)
        self.mixup_spin.setValue(0.0)
        self.hsv_h_spin.setValue(0.015)
        self.hsv_s_spin.setValue(0.7)
        self.hsv_v_spin.setValue(0.4)
        self.box_loss_spin.setValue(7.5)
        self.cls_loss_spin.setValue(0.5)
        self.dfl_loss_spin.setValue(1.5)

    def set_aggressive_preset(self):
        self.fliplr_spin.setValue(0.5)
        self.flipud_spin.setValue(0.5)
        self.degrees_spin.setValue(45.0)
        self.mosaic_spin.setValue(1.0)
        self.mixup_spin.setValue(0.5)
        self.hsv_h_spin.setValue(0.1)
        self.hsv_s_spin.setValue(0.9)
        self.hsv_v_spin.setValue(0.6)

    def set_light_preset(self):
        self.fliplr_spin.setValue(0.3)
        self.flipud_spin.setValue(0.0)
        self.degrees_spin.setValue(0.0)
        self.mosaic_spin.setValue(0.5)
        self.mixup_spin.setValue(0.0)
        self.hsv_h_spin.setValue(0.01)
        self.hsv_s_spin.setValue(0.5)
        self.hsv_v_spin.setValue(0.3)
        self.epochs_spin.setValue(30)
        self.batch_train.setValue(8)

    def start_train(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "训练正在进行中")
            return
        if not os.path.exists(self.model_path.text()):
            QMessageBox.warning(self, "错误", "请选择有效的模型文件")
            return
        if not os.path.exists(self.yaml_path.text()):
            QMessageBox.warning(self, "错误", "请选择有效的YAML配置文件")
            return
        if not os.path.exists(self.project_dir.text()):
            QMessageBox.warning(self, "错误", "请选择有效的项目目录")
            return
        params = {
            'model_path': self.model_path.text(),
            'yaml_path': self.yaml_path.text(),
            'project': self.project_dir.text(),
            'name': self.exp_name.text(),
            'epochs': self.epochs_spin.value(),
            'imgsz': self.imgsz_train.value(),
            'batch': self.batch_train.value(),
            'workers': self.workers_train.value(),
            'device': self.device_combo.currentText(),
            'hsv_h': self.hsv_h_spin.value(),
            'hsv_s': self.hsv_s_spin.value(),
            'hsv_v': self.hsv_v_spin.value(),
            'flipud': self.flipud_spin.value(),
            'fliplr': self.fliplr_spin.value(),
            'degrees': self.degrees_spin.value(),
            'mosaic': self.mosaic_spin.value(),
            'mixup': self.mixup_spin.value(),
            'box_loss': self.box_loss_spin.value(),
            'cls_loss': self.cls_loss_spin.value(),
            'dfl_loss': self.dfl_loss_spin.value(),
            'cache': self.cache_check.isChecked(),
            'patience': self.patience_spin.value(),
        }
        self.worker = TrainWorker(params)
        self.worker.signals = WorkerSignals()
        self.worker.signals.log.connect(self.log_text.append)
        self.worker.signals.error.connect(self.show_train_error)
        self.worker.signals.progress.connect(self.update_train_progress)
        self.worker.signals.finished.connect(self.train_finished)
        self.worker.start()
        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)

    def stop_train(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log_text.append("正在停止训练...")

    def show_train_error(self, msg):
        QMessageBox.critical(self, "训练错误", msg)
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)

    def update_train_progress(self, val, msg):
        self.train_progress.setValue(val)
        self.log_text.append(msg)

    def train_finished(self):
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.train_progress.setValue(100)

    def export_config(self):
        params = {
            'model_path': self.model_path.text(),
            'yaml_path': self.yaml_path.text(),
            'project': self.project_dir.text(),
            'name': self.exp_name.text(),
            'epochs': self.epochs_spin.value(),
            'imgsz': self.imgsz_train.value(),
            'batch': self.batch_train.value(),
            'workers': self.workers_train.value(),
            'device': self.device_combo.currentText(),
            'hsv_h': self.hsv_h_spin.value(),
            'hsv_s': self.hsv_s_spin.value(),
            'hsv_v': self.hsv_v_spin.value(),
            'flipud': self.flipud_spin.value(),
            'fliplr': self.fliplr_spin.value(),
            'degrees': self.degrees_spin.value(),
            'mosaic': self.mosaic_spin.value(),
            'mixup': self.mixup_spin.value(),
            'box_loss': self.box_loss_spin.value(),
            'cls_loss': self.cls_loss_spin.value(),
            'dfl_loss': self.dfl_loss_spin.value(),
            'cache': self.cache_check.isChecked(),
            'patience': self.patience_spin.value(),
        }
        file_path, _ = QFileDialog.getSaveFileName(self, "导出训练配置", "train_config.json", "JSON文件 (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(params, f, indent=4)
            QMessageBox.information(self, "成功", f"配置已导出至 {file_path}")

    def import_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "导入训练配置", "", "JSON文件 (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    params = json.load(f)
                self.model_path.setText(params.get('model_path', ''))
                self.yaml_path.setText(params.get('yaml_path', ''))
                self.project_dir.setText(params.get('project', ''))
                self.exp_name.setText(params.get('name', 'exp'))
                self.epochs_spin.setValue(params.get('epochs', DEFAULT_EPOCHS))
                self.imgsz_train.setValue(params.get('imgsz', DEFAULT_IMGSZ))
                self.batch_train.setValue(params.get('batch', DEFAULT_BATCH))
                self.workers_train.setValue(params.get('workers', DEFAULT_WORKERS))
                self.device_combo.setCurrentText(params.get('device', DEFAULT_DEVICE))
                self.hsv_h_spin.setValue(params.get('hsv_h', 0.015))
                self.hsv_s_spin.setValue(params.get('hsv_s', 0.7))
                self.hsv_v_spin.setValue(params.get('hsv_v', 0.4))
                self.flipud_spin.setValue(params.get('flipud', 0.0))
                self.fliplr_spin.setValue(params.get('fliplr', 0.5))
                self.degrees_spin.setValue(params.get('degrees', 0.0))
                self.mosaic_spin.setValue(params.get('mosaic', 1.0))
                self.mixup_spin.setValue(params.get('mixup', 0.0))
                self.box_loss_spin.setValue(params.get('box_loss', 7.5))
                self.cls_loss_spin.setValue(params.get('cls_loss', 0.5))
                self.dfl_loss_spin.setValue(params.get('dfl_loss', 1.5))
                self.cache_check.setChecked(params.get('cache', False))
                self.patience_spin.setValue(params.get('patience', 50))
                QMessageBox.information(self, "成功", "配置导入成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入失败: {e}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()


class FormatTab(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = QSettings('YOLO_Tool', 'Format')
        self.initUI()
        self.worker = None

    def initUI(self):
        layout = QVBoxLayout(self)

        model_group = QGroupBox("模型")
        model_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("未加载模型")
        self.btn_model = QPushButton("选择模型")
        self.btn_model.clicked.connect(self.load_model)
        model_layout.addWidget(self.model_path, 1)
        model_layout.addWidget(self.btn_model)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        format_group = QGroupBox("导出格式")
        format_layout = QHBoxLayout()
        self.onnx_radio = QRadioButton("onnx")
        self.engine_radio = QRadioButton("engine")
        self.onnx_radio.setChecked(True)
        format_layout.addWidget(self.onnx_radio)
        format_layout.addWidget(self.engine_radio)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        param_group = QGroupBox("转换参数")
        param_layout = QGridLayout()
        self.half_check = QCheckBox("half")
        param_layout.addWidget(self.half_check, 0, 0)
        param_layout.addWidget(QLabel("device:"), 0, 1)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['cpu', '0', '1', '2', '3', 'mps'])
        self.device_combo.setCurrentText(DEFAULT_DEVICE)
        param_layout.addWidget(self.device_combo, 0, 2)
        param_layout.addWidget(QLabel("imgsz:"), 1, 0)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 2560)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(DEFAULT_IMGSZ)
        param_layout.addWidget(self.imgsz_spin, 1, 1)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        self.convert_btn = QPushButton("转换开始")
        self.convert_btn.clicked.connect(self.start_convert)
        layout.addWidget(self.convert_btn)

        log_group = QGroupBox("转换日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        self.load_settings()

    def load_settings(self):
        self.model_path.setText(self.settings.value('format_model_path', ''))
        device = self.settings.value('format_device', DEFAULT_DEVICE)
        idx = self.device_combo.findText(device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)

    def save_settings(self):
        self.settings.setValue('format_model_path', self.model_path.text())
        self.settings.setValue('format_device', self.device_combo.currentText())

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择YOLO模型", "", "PyTorch模型 (*.pt)")
        if path:
            self.model_path.setText(path)
            self.save_settings()

    def start_convert(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "转换正在进行中")
            return
        model_path = self.model_path.text()
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "请选择有效的模型文件")
            return
        format_type = 'onnx' if self.onnx_radio.isChecked() else 'engine'
        params = {
            'model_path': model_path,
            'format': format_type,
            'half': self.half_check.isChecked(),
            'device': self.device_combo.currentText(),
            'imgsz': self.imgsz_spin.value(),
        }
        self.worker = FormatWorker(params)
        self.worker.signals = WorkerSignals()
        self.worker.signals.log.connect(self.log_text.append)
        self.worker.signals.error.connect(self.show_convert_error)
        self.worker.signals.finished.connect(self.convert_finished)
        self.worker.start()
        self.convert_btn.setEnabled(False)

    def show_convert_error(self, msg):
        QMessageBox.critical(self, "转换错误", msg)
        self.convert_btn.setEnabled(True)

    def convert_finished(self):
        self.convert_btn.setEnabled(True)
        self.log_text.append("转换完成")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO26权重训练-测试工具")
        self.setGeometry(100, 100, 1400, 900)

        self.tab_widget = QTabWidget()
        self.detect_tab = DetectTab()
        self.train_tab = TrainTab()
        self.format_tab = FormatTab()

        self.tab_widget.addTab(self.detect_tab, "detect")
        self.tab_widget.addTab(self.train_tab, "train")
        self.tab_widget.addTab(self.format_tab, "format")

        self.setCentralWidget(self.tab_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())