"""
YOLO26 图片检测 GUI 程序（UI 分离版）
界面由 yolo_detector.ui 定义，逻辑代码在此文件中。
运行：python yolo_detector_gui.py
"""

import sys
import os
import glob
import time
import json
import csv
import uuid
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QGroupBox,
    QSpinBox, QProgressBar, QMenuBar,
    QListWidget, QListWidgetItem, QTextEdit, QTabWidget,
    QGroupBox as QG
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor, QKeySequence
from PyQt5.uic import loadUi

# ── 第三方库 ──────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


# ── 主窗口 ─────────────────────────────────────────────────
class YoloDetectorGUI(QMainWindow):

    def __init__(self):
        super().__init__()

        # 加载 .ui 文件（与脚本同目录）
        ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_detector.ui")
        loadUi(ui_path, self)

        # 状态变量
        self.model = None
        self.model_path = ""
        self.model2 = None  # 第二个模型
        self.model2_path = ""
        self.image_files = []
        self.current_index = 0
        self.detections = []  # 第一个模型的检测结果
        self.detections2 = []  # 第二个模型的检测结果
        self.result_image = None
        self.class_colors = {}
        self.class_colors2 = {}  # 第二个模型的类别颜色
        self.last_detection_info = ""
        self.history_records = []
        self.label_dir = ""
        self.label_index = {}  # {image_basename_no_ext: [(cls, cx, cy, w, h), ...]}
        self.class_names = {}  # {class_id: name}
        self.label_result_image = None  # 标签绘制后的图片
        self.diff_scores = {}  # {img_path: diff_score}
        self.sorted_indices = []  # 差异排序后的图片索引列表
        self.diff_sorted = False  # 是否已进行差异排序
        self.display_mode = "single"  # "single" or "dual"

        # 尝试加载默认模型
        self._try_load_model()

        # GPU 监控定时器
        self.gpu_timer = QTimer()
        self.gpu_timer.timeout.connect(self._update_gpu_status)
        self.gpu_timer.start(1000)

        # 连接菜单动作
        self.action_open_dir.triggered.connect(self._select_directory)
        self.action_exit.triggered.connect(self.close)
        self.action_prev.triggered.connect(self._prev_image)
        self.action_next.triggered.connect(self._next_image)
        self.action_single_detect.triggered.connect(self._detect_single)
        self.action_batch_detect.triggered.connect(self._detect_directory)
        self.action_export_json.triggered.connect(self._export_results_json)
        self.action_export_csv.triggered.connect(self._export_results_csv)
        self.action_clear_history.triggered.connect(self._clear_history)

        # 连接按钮
        self.model_btn.clicked.connect(self._select_model)
        self.dir_btn.clicked.connect(self._select_directory)
        self.label_dir_btn.clicked.connect(self._select_label_directory)
        self.detect_btn.clicked.connect(self._detect_directory)
        self.single_btn.clicked.connect(self._detect_single)
        self.diff_sort_btn.clicked.connect(self._diff_sort)
        self.prev_btn.clicked.connect(self._prev_image)
        self.next_btn.clicked.connect(self._next_image)

        # 连接历史记录
        self.history_list.itemClicked.connect(self._show_history_detail)

        # 注册快捷键
        self._setup_shortcuts()

        # 添加第二个模型选择下拉菜单
        self._add_model2_selector()

        # 连接显示模式切换
        self.display_mode_combo.currentIndexChanged.connect(self._on_display_mode_changed)

        # 右侧面板最小宽度固定
        self.control_group.setMinimumWidth(300)
        self.btn_group.setMinimumWidth(300)
        self.info_progress_group.setMinimumWidth(300)
        self.result_group.setMinimumWidth(300)

        # 加载上次保存的参数
        self._load_last_settings()

    def _add_model2_selector(self):
        """添加第二个模型选择器"""
        from PyQt5.QtWidgets import QComboBox
        # 在控制设置区添加第二个模型选择行
        model2_row = QHBoxLayout()
        model2_row.addWidget(QLabel("第二个模型:"))
        self.model2_input = QLabel("未加载" if not self.model2 else f"✅ {self.model2_path}")
        self.model2_input.setStyleSheet("color: red;" if not self.model2 else "color: green; font-weight: bold;")
        model2_row.addWidget(self.model2_input)
        self.model2_btn = QPushButton("选择模型文件")
        self.model2_btn.clicked.connect(self._select_model2)
        model2_row.addWidget(self.model2_btn)
        self.control_inner.addLayout(model2_row)

    # ── 快捷键 ───────────────────────────────────────────
    def _setup_shortcuts(self):
        pass  # keyPressEvent 直接处理，无需 QShortcut

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._prev_image()
        elif event.key() == Qt.Key_Right:
            self._next_image()
        elif event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
            self._detect_single()
        elif event.key() == Qt.Key_B and event.modifiers() & Qt.ControlModifier:
            self._detect_directory()
        elif event.key() == Qt.Key_Q and event.modifiers() & Qt.ControlModifier:
            self.close()
        elif event.key() == Qt.Key_O and event.modifiers() & Qt.ControlModifier:
            self._select_directory()
        else:
            super().keyPressEvent(event)

    # ── 模型 ──────────────────────────────────────────────
    def _try_load_model(self):
        if not YOLO_AVAILABLE:
            return
        for d in (".", "models"):
            if not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                if name.endswith(".pt"):
                    try:
                        self.model = YOLO(os.path.join(d, name))
                        self.model_path = os.path.join(d, name)
                        self.model_input.setText(f"✅ {self.model_path}")
                        self.model_input.setStyleSheet("color: green; font-weight: bold;")
                        return
                    except Exception:
                        continue

    def _select_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 模型文件", "", "模型文件 (*.pt)"
        )
        if not path:
            return
        if not YOLO_AVAILABLE:
            QMessageBox.critical(self, "错误", "未安装 ultralytics 库。\n请运行: pip install ultralytics")
            return
        try:
            self.model = YOLO(path)
            self.model_path = path
            self.model_input.setText(f"✅ {path}")
            self.model_input.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(self, "成功", f"模型加载成功！\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{e}")

    def _select_model2(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择第二个 YOLO 模型文件", "", "模型文件 (*.pt)"
        )
        if not path:
            return
        if not YOLO_AVAILABLE:
            QMessageBox.critical(self, "错误", "未安装 ultralytics 库。\n请运行: pip install ultralytics")
            return
        try:
            self.model2 = YOLO(path)
            self.model2_path = path
            self.model2_input.setText(f"✅ {path}")
            self.model2_input.setStyleSheet("color: green; font-weight: bold;")
            QMessageBox.information(self, "成功", f"第二个模型加载成功！\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"第二个模型加载失败:\n{e}")

    # ── 目录 ──────────────────────────────────────────────
    def _select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if not directory:
            return
        self.image_files = self._scan_images(directory)
        if not self.image_files:
            QMessageBox.warning(self, "警告", "目录中未找到支持的图片文件！")
            return
        self.dir_label.setText(directory)
        self.dir_label.setStyleSheet("color: #333; font-weight: bold;")
        self.current_index = 0
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self._update_info()
        self._show_current_image()
        QMessageBox.information(self, "成功", f"已找到 {len(self.image_files)} 张图片\n目录: {directory}")

    def _scan_images(self, directory: str) -> list:
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp")
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        files.sort()
        return files

    def _select_label_directory(self):
        """选择标签目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择标签目录")
        if not directory:
            return
        self.label_dir = directory
        self.label_index = self._load_labels(directory)
        self.label_dir_input.setText(directory)
        self.label_dir_input.setStyleSheet("color: #333; font-weight: bold;")
        # 如果有当前图片，刷新显示
        if self.image_files and self.current_index < len(self.image_files):
            self._show_current_image()
        QMessageBox.information(self, "成功", f"已加载 {len(self.label_index)} 个标签文件\n目录: {directory}")

    def _load_labels(self, label_dir: str) -> dict:
        """加载YOLO格式标签文件，返回 {image_no_ext: [(cls_id, cx, cy, w, h), ...]}"""
        label_index = {}
        if not os.path.isdir(label_dir):
            return label_index
        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue
            filepath = os.path.join(label_dir, fname)
            img_base = os.path.splitext(fname)[0]
            boxes = []
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append((cls_id, cx, cy, w, h))
                        # 记录类别名称
                        if cls_id not in self.class_names:
                            self.class_names[cls_id] = f"class_{cls_id}"
                if boxes:
                    label_index[img_base] = boxes
            except Exception:
                continue
        return label_index

    # ── 检测 ─────────────────────────────────────────────
    def _detect_single(self):
        if not self.model:
            QMessageBox.warning(self, "警告", "请先选择 YOLO 模型文件！")
            return
        if not self.image_files:
            QMessageBox.warning(self, "警告", "请先选择检测目录！")
            return
        if self.current_index >= len(self.image_files):
            QMessageBox.warning(self, "警告", "请先选择一张图片！")
            return

        img_path = self.image_files[self.current_index]
        try:
            t0 = time.time()
            results = self.model(img_path, conf=self.conf_spin.value() / 100.0,
                                 imgsz=self.img_size_spin.value())
            elapsed = time.time() - t0
            self._save_results(img_path, results)

            # 如果启用了第二个模型，也进行检测
            if self.display_mode == "dual" and self.model2:
                results2 = self.model2(img_path, conf=self.conf_spin.value() / 100.0,
                                       imgsz=self.img_size_spin.value())
                self._save_results2(img_path, results2)

            self._update_info()
            self._show_current_image()

            n_det = len(results[0].boxes) if results and results[0].boxes else 0
            conf_val = self.conf_spin.value() / 100.0
            self.last_detection_info = (
                f"单张检测耗时: {elapsed*1000:.1f} ms\n"
                f"置信度: {conf_val:.2f} | 检测到 {n_det} 个目标"
            )
            self.result_list.setText(self.last_detection_info)
            self._add_history("单张检测", img_path, elapsed, n_det, conf_val)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败:\n{e}")

    def _detect_directory(self):
        if not self.model:
            QMessageBox.warning(self, "警告", "请先选择 YOLO 模型文件！")
            return
        if not self.image_files:
            QMessageBox.warning(self, "警告", "请先选择检测目录！")
            return

        confirm = QMessageBox.question(
            self, "确认",
            f"将对 {len(self.image_files)} 张图片进行检测，是否继续？",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        self.detect_btn.setEnabled(False)
        self.progress.setValue(0)
        QTimer.singleShot(0, self._run_detection)

    def _run_detection(self):
        total = len(self.image_files)
        t0_total = time.time()
        for i, img_path in enumerate(self.image_files):
            try:
                t1 = time.time()
                results = self.model(img_path, conf=self.conf_spin.value() / 100.0,
                                     imgsz=self.img_size_spin.value())
                elapsed = time.time() - t1
                self._save_results(img_path, results)

                # 如果启用了第二个模型，也进行检测
                if self.display_mode == "dual" and self.model2:
                    results2 = self.model2(img_path, conf=self.conf_spin.value() / 100.0,
                                           imgsz=self.img_size_spin.value())
                    self._save_results2(img_path, results2)

                n_det = len(results[0].boxes) if results and results[0].boxes else 0
                conf_val = self.conf_spin.value() / 100.0
                self.result_list.setText(
                    f"图片 {i+1}/{total}: {elapsed*1000:.1f} ms\n"
                    f"置信度: {conf_val:.2f} | 检测到 {n_det} 个目标"
                )
                self._add_history("批量检测", img_path, elapsed, n_det, conf_val)
            except Exception as e:
                print(f"检测 {img_path} 失败: {e}")
            self.progress.setValue(int((i + 1) / total * 100))
            QApplication.processEvents()

        total_elapsed = time.time() - t0_total
        self.detect_btn.setEnabled(True)
        self.current_index = 0
        self._update_info()
        self._show_current_image()

        # 如果有标签目录，计算并显示差异分数
        if self.label_index:
            self._show_diff_scores()

        conf_val = self.conf_spin.value() / 100.0
        self.result_list.setText(
            f"批量检测完成！\n共{total} 张图片，总耗时 {total_elapsed:.2f} 秒\n"
            f"置信度: {conf_val:.2f}"
        )
        QMessageBox.information(self, "完成",
                                f"检测完成！共处理 {total} 张图片，总耗时 {total_elapsed:.2f} 秒。")

    def _show_diff_scores(self):
        """计算并显示差异分数"""
        self.diff_scores = {}
        for img_path in self.image_files:
            det_boxes = self._get_detection_boxes(img_path)
            label_boxes = self._get_label_boxes(img_path)
            score = self._calculate_diff_score(det_boxes, label_boxes)
            self.diff_scores[img_path] = score

        # 显示差异分数
        sorted_diffs = sorted(self.diff_scores.items(), key=lambda x: x[1], reverse=True)
        diff_text = "差异分数 (越大差异越大):\n"
        for img_path, score in sorted_diffs[:10]:
            img_name = os.path.basename(img_path)
            diff_text += f"  {score:.1f} | {img_name[:35]}...\n"
        if len(sorted_diffs) > 10:
            diff_text += f"  ... 共 {len(sorted_diffs)} 张图片\n"
        self.result_list.setText(diff_text)

    def _save_results2(self, img_path: str, results):
        """保存第二个模型的检测结果"""
        if not results or len(results) == 0:
            return
        result = results[0]
        boxes = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, str(cls_id))
            if cls_name not in self.class_colors2:
                self.class_colors2[cls_name] = self._get_next_color(len(self.class_colors2))
            boxes.append({"class": cls_name, "confidence": conf, "bbox": xyxy})
        self.detections2.append({"path": img_path, "boxes": boxes})

    def _save_results(self, img_path: str, results):
        if not results or len(results) == 0:
            return
        result = results[0]
        boxes = []
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = result.names.get(cls_id, str(cls_id))
            if cls_name not in self.class_colors:
                self.class_colors[cls_name] = self._get_next_color(len(self.class_colors))
            boxes.append({"class": cls_name, "confidence": conf, "bbox": xyxy})
        self.detections.append({"path": img_path, "boxes": boxes})

    def _get_next_color(self, index: int):
        colors = [
            QColor(255, 0, 0), QColor(0, 170, 255), QColor(0, 200, 0),
            QColor(255, 170, 0), QColor(170, 0, 255), QColor(255, 255, 0),
            QColor(255, 0, 255), QColor(0, 255, 255), QColor(255, 100, 0),
            QColor(0, 255, 100),
        ]
        return colors[index % len(colors)]

    # ── 图片浏览 ─────────────────────────────────────────
    def _prev_image(self):
        if not self.image_files:
            return
        if self.diff_sorted and self.sorted_indices:
            # 使用排序后的索引
            current_pos = self.sorted_indices.index(self.current_index) if self.current_index in self.sorted_indices else 0
            new_pos = (current_pos - 1) % len(self.sorted_indices)
            self.current_index = self.sorted_indices[new_pos]
        else:
            if self.current_index > 0:
                self.current_index -= 1
            else:
                self.current_index = len(self.image_files) - 1
        self._update_info()
        self._show_current_image()

    def _next_image(self):
        if not self.image_files:
            return
        if self.diff_sorted and self.sorted_indices:
            # 使用排序后的索引
            current_pos = self.sorted_indices.index(self.current_index) if self.current_index in self.sorted_indices else 0
            new_pos = (current_pos + 1) % len(self.sorted_indices)
            self.current_index = self.sorted_indices[new_pos]
        else:
            if self.current_index < len(self.image_files) - 1:
                self.current_index += 1
            else:
                self.current_index = 0
        self._update_info()
        self._show_current_image()

    def _update_info(self):
        if self.image_files:
            self.info_label.setText(f"图片: {self.current_index + 1} / {len(self.image_files)}")
        else:
            self.info_label.setText("图片: -- / --")

    def _show_current_image(self):
        if not self.image_files:
            return
        img_path = self.image_files[self.current_index]
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            self.image_label.setText("无法加载图片")
            self.label_image_label.setText("无法加载图片")
            return

        # ── 左侧：模型1检测结果 ──
        self._draw_model_result(self.image_label, self.detections, img_path)

        # ── 右侧：根据模式显示标签或模型2结果 ──
        if self.display_mode == "dual":
            if self.model2:
                self._draw_model_result(self.label_image_label, self.detections2, img_path)
            else:
                self.label_image_label.setText("请加载第二个模型")
        else:
            # 单模型模式：左侧模型1，右侧标签
            self._draw_labels_on_image(img_path)

        # ── 更新结果列表 ──
        # 获取当前显示的检测结果
        boxes_to_draw = []
        found = False
        for d in self.detections:
            if d["path"] == img_path and d["boxes"]:
                boxes_to_draw = d["boxes"]
                found = True
                break

        if found:
            text_parts = [f"图片: {os.path.basename(img_path)}", f"检测到 {len(boxes_to_draw)} 个目标:"]
            for b in boxes_to_draw:
                text_parts.append(f"  • {b['class']} ({b['confidence']:.2%})")
            # 如果有标签，显示差异分数
            if self.label_index:
                label_boxes = self._get_label_boxes(img_path)
                det_boxes = self._get_detection_boxes(img_path)
                score = self._calculate_diff_score(det_boxes, label_boxes)
                text_parts.append(f"差异分数: {score:.1f}")
            result_text = "\n".join(text_parts)
            if self.last_detection_info:
                self.result_list.setText(self.last_detection_info + "\n\n" + result_text)
            else:
                self.result_list.setText(result_text)
        else:
            label_boxes = self._get_label_boxes(img_path)
            if label_boxes:
                text_parts = [f"标签: {len(label_boxes)} 个目标:"]
                for lb in label_boxes:
                    cls_id, cx, cy, w, h = lb
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    text_parts.append(f"  • {cls_name}")
                self.result_list.setText("\n".join(text_parts))
            elif self.last_detection_info:
                self.result_list.setText(self.last_detection_info)
            else:
                self.result_list.setText("等待检测...")

    def _draw_labels_on_image(self, img_path: str):
        """在右侧标签窗口绘制标签框"""
        label_boxes = self._get_label_boxes(img_path)
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            self.label_image_label.setText("无法加载图片")
            return

        if not label_boxes:
            # 无标签，显示原图
            label_size = self.label_image_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_image_label.setPixmap(pixmap)
            self.label_image_label.setAlignment(Qt.AlignCenter)
            return

        # 绘制标签框
        img = pixmap.toImage()
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        pw, ph = pixmap.width(), pixmap.height()

        for lb in label_boxes:
            cls_id, cx, cy, w, h = lb
            x1 = int((cx - w / 2) * pw)
            y1 = int((cy - h / 2) * ph)
            x2 = int((cx + w / 2) * pw)
            y2 = int((cy + h / 2) * ph)
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            if cls_name not in self.class_colors:
                self.class_colors[cls_name] = self._get_next_color(len(self.class_colors))
            color = self.class_colors[cls_name]
            pen = QPen(color, 3)
            painter.setPen(pen)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.setPen(color)
            painter.setFont(QFont("Arial", 14))
            painter.drawText(x1, y1 - 5, cls_name)
        painter.end()

        self.label_result_image = QPixmap.fromImage(img)
        label_size = self.label_image_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            self.label_result_image = self.label_result_image.scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        self.label_image_label.setPixmap(self.label_result_image)
        self.label_image_label.setAlignment(Qt.AlignCenter)

    def _draw_detections(self, pixmap: QPixmap, boxes: list, use_class_colors: bool = True) -> QPixmap:
        img = pixmap.toImage()
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        colors = self.class_colors if use_class_colors else self.class_colors2
        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            color = colors.get(box["class"], QColor(255, 0, 0))
            pen = QPen(color, 5)
            painter.setPen(pen)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            label = f"{box['class']} {box['confidence']:.2f}"
            painter.setPen(color)
            painter.setFont(QFont("Arial", 32))
            painter.drawText(x1, y1 - 5, label)
        painter.end()
        return QPixmap.fromImage(img)

    def _draw_model_result(self, label: QLabel, detections: list, img_path: str):
        """绘制模型检测结果到指定标签"""
        img = QPixmap(img_path)
        if img.isNull():
            label.setText("无法加载图片")
            return

        # 查找检测结果
        boxes_to_draw = []
        for d in detections:
            if d["path"] == img_path and d["boxes"]:
                boxes_to_draw = d["boxes"]
                break

        if boxes_to_draw:
            is_first = (label == self.image_label)
            result_image = self._draw_detections(img, boxes_to_draw, is_first)
            img = result_image

        label_size = label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            img = img.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(img)
        label.setAlignment(Qt.AlignCenter)

    def _get_label_boxes(self, img_path: str) -> list:
        """根据图片路径查找对应的标签，返回 [(cls_id, cx, cy, w, h), ...]"""
        img_base = os.path.splitext(os.path.basename(img_path))[0]
        return self.label_index.get(img_base, [])

    def _get_detection_boxes(self, img_path: str) -> list:
        """获取图片的检测框列表"""
        for d in self.detections:
            if d["path"] == img_path:
                return d["boxes"]
        return []

    def _calculate_diff_score(self, det_boxes, label_boxes) -> float:
        """计算检测结果与标签的差异分数（越大表示差异越大）"""
        if not det_boxes and not label_boxes:
            return 0.0
        if not det_boxes:
            return len(label_boxes) * 2  # 无检测结果，标签存在
        if not label_boxes:
            return len(det_boxes) * 2  # 有检测结果，无标签

        # 计算IOU匹配
        matched = 0
        for lb in label_boxes:
            cls_id, cx, cy, w, h = lb
            best_iou = 0
            for db in det_boxes:
                x1, y1, x2, y2 = db["bbox"]
                # 计算IOU
                inter_x1 = max(x1, int((cx - w / 2) * 1000))  # 简化计算
                inter_y1 = max(y1, int((cy - h / 2) * 1000))
                inter_x2 = min(x2, int((cx + w / 2) * 1000))
                inter_y2 = min(y2, int((cy + h / 2) * 1000))
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    label_area = w * h * 1000 * 1000  # 简化
                    iou = inter_area / (label_area + 1e-6)
                    best_iou = max(best_iou, iou)
            if best_iou > 0.5:
                matched += 1

        # 差异分数 = 未匹配的标签数 + 未匹配的检测结果数
        unmatched_labels = len(label_boxes) - matched
        unmatched_dets = len(det_boxes) - matched
        return unmatched_labels * 2 + unmatched_dets

    # ── 差异排序 ─────────────────────────────────────────
    def _diff_sort(self):
        """计算模型检测结果与标签的差异，按差异大小排序"""
        if not self.image_files:
            QMessageBox.warning(self, "警告", "请先选择检测目录！")
            return
        if not self.label_index:
            QMessageBox.warning(self, "警告", "请先选择标签目录！")
            return
        if not self.detections:
            QMessageBox.warning(self, "警告", "请先执行检测！")
            return

        self.diff_scores = {}
        for img_path in self.image_files:
            det_boxes = self._get_detection_boxes(img_path)
            label_boxes = self._get_label_boxes(img_path)
            score = self._calculate_diff_score(det_boxes, label_boxes)
            self.diff_scores[img_path] = score

        # 按差异分数降序排序
        sorted_items = sorted(self.diff_scores.items(), key=lambda x: x[1], reverse=True)
        self.sorted_indices = [self.image_files.index(item[0]) for item in sorted_items]
        self.diff_sorted = True
        self.current_index = 0
        self._update_info()
        self._show_current_image()
        QMessageBox.information(self, "差异排序完成",
                                f"已对 {len(self.image_files)} 张图片进行差异排序\n"
                                f"差异最大的图片已显示")

    # ── 历史记录 ─────────────────────────────────────────
    def _add_history(self, detect_type: str, img_path: str, elapsed: float,
                     n_det: int, conf: float):
        record = {
            "id": str(uuid.uuid4())[:8],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": detect_type,
            "image": os.path.basename(img_path),
            "path": img_path,
            "elapsed_ms": elapsed * 1000,
            "detections": n_det,
            "confidence": conf,
        }
        self.history_records.append(record)
        self._refresh_history_list()

    def _refresh_history_list(self):
        self.history_list.clear()
        for r in self.history_records:
            item = QListWidgetItem(
                f"[{r['time']}] {r['type']} | {r['image'][:30]}... | "
                f"{r['elapsed_ms']:.0f}ms | {r['detections']}目标 | conf={r['confidence']:.2f}"
            )
            item.setData(Qt.UserRole, r)
            self.history_list.addItem(item)
        self.history_list.scrollToBottom()

    def _show_history_detail(self, item: QListWidgetItem):
        r = item.data(Qt.UserRole)
        detail = (
            f"时间: {r['time']}\n"
            f"类型: {r['type']}\n"
            f"图片: {r['image']}\n"
            f"完整路径: {r['path']}\n"
            f"耗时: {r['elapsed_ms']:.1f} ms\n"
            f"检测数量: {r['detections']}\n"
            f"置信度阈值: {r['confidence']:.2f}\n"
            f"记录ID: {r['id']}"
        )
        self.history_detail.setText(detail)

    def _clear_history(self):
        if self.history_records:
            count = len(self.history_records)
            self.history_records.clear()
            self.history_list.clear()
            self.history_detail.clear()
            QMessageBox.information(self, "提示", f"已清空 {count} 条历史记录")

    # ── 导出 ─────────────────────────────────────────────
    def _export_results_json(self):
        if not self.detections:
            QMessageBox.warning(self, "警告", "没有检测结果可导出！")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存检测结果", "", "JSON 文件 (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.detections, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "成功", f"已导出 {len(self.detections)} 条检测结果到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")

    def _export_results_csv(self):
        if not self.detections:
            QMessageBox.warning(self, "警告", "没有检测结果可导出！")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存检测结果", "", "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["图片路径", "类别", "置信度", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
                for d in self.detections:
                    for b in d["boxes"]:
                        writer.writerow([d["path"], b["class"], f"{b['confidence']:.4f}", *b["bbox"]])
            QMessageBox.information(self, "成功", f"已导出 {len(self.detections)} 条检测结果到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败:\n{e}")

    # ── GPU / 系统监控 ─────────────────────────────────
    def _load_last_settings(self):
        """加载上次保存的设置"""
        import json
        settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_settings.json")
        if not os.path.exists(settings_path):
            return
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
            # 恢复模型路径
            if "model_path" in settings and settings["model_path"]:
                self.model_path = settings["model_path"]
                self.model_input.setText(f"✅ {self.model_path}")
                self.model_input.setStyleSheet("color: green; font-weight: bold;")
                # 尝试加载模型
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
                except:
                    pass
            # 恢复检测目录
            if "image_dir" in settings and settings["image_dir"]:
                self.dir_label.setText(settings["image_dir"])
                self.dir_label.setStyleSheet("color: #333; font-weight: bold;")
                self.image_files = self._scan_images(settings["image_dir"])
                if self.image_files:
                    self.current_index = 0
                    self.prev_btn.setEnabled(True)
                    self.next_btn.setEnabled(True)
                    self._update_info()
                    self._show_current_image()
            # 恢复标签目录
            if "label_dir" in settings and settings["label_dir"]:
                self.label_dir = settings["label_dir"]
                self.label_dir_input.setText(settings["label_dir"])
                self.label_dir_input.setStyleSheet("color: #333; font-weight: bold;")
                self.label_index = self._load_labels(settings["label_dir"])
        except Exception as e:
            print(f"加载设置失败: {e}")

    def _save_settings(self):
        """保存当前设置"""
        import json
        settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_settings.json")
        settings = {
            "model_path": self.model_path,
            "image_dir": self.dir_label.text() if self.dir_label.text() != "未选择目录" else "",
            "label_dir": self.label_dir,
            "conf_threshold": self.conf_spin.value(),
            "img_size": self.img_size_spin.value(),
        }
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存设置失败: {e}")

    def _on_display_mode_changed(self, index):
        """切换显示模式"""
        # 0=单模型+标签, 1=双模型对比
        if index == 0:
            self.display_mode = "single"
            self._show_current_image()
        elif index == 1:
            if not self.model2:
                QMessageBox.warning(self, "警告", "请先选择第二个模型文件！")
                # 阻塞信号，避免递归
                self.display_mode_combo.blockSignals(True)
                self.display_mode_combo.setCurrentIndex(0)
                self.display_mode_combo.blockSignals(False)
                return
            self.display_mode = "dual"
            self._show_current_image()

    def _update_gpu_status(self):
        if TORCH_AVAILABLE:
            gpu_name = torch.cuda.get_device_name(0)
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            self.gpu_label.setText(f"GPU: {gpu_name}")
            self.gpu_mem_label.setText(f"显存: {allocated:.0f}MB / {reserved:.0f}MB (总 {total:.0f}MB)")
        else:
            self.gpu_label.setText("GPU: CPU 模式")
            self.gpu_mem_label.setText("")

        if PSUTIL_AVAILABLE:
            cpu_pct = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            self.cpu_label.setText(f"CPU: {cpu_pct}%")
            self.mem_label.setText(f"内存: {mem.percent}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")
        else:
            self.cpu_label.setText("CPU: N/A")
            self.mem_label.setText("内存: N/A")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(palette.Window, QColor(240, 240, 240))
    palette.setColor(palette.WindowText, QColor(50, 50, 50))
    app.setPalette(palette)

    window = YoloDetectorGUI()
    window.show()

    # 退出时保存设置
    def _save_on_exit():
        window._save_settings()

    app.aboutToQuit.connect(_save_on_exit)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
