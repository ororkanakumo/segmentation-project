import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import csv
from PyQt6.QtWidgets import (QGridLayout, QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QHBoxLayout, QVBoxLayout, QWidget, QProgressBar, QMessageBox, QTabWidget, QScrollArea)
from PyQt6.QtGui import QPixmap, QImage, QDesktopServices
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
MODEL_PATH = "best_model.keras"
IMG_SIZE = (192, 192)
THRESHOLD = 0.5
BUTTON_WIDTH = 180
BUTTON_HEIGHT = 25
MAX_PREVIEW_IMAGES = 30
class SegmentationModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
    def predict(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, IMG_SIZE)
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        pred = self.model.predict(input_tensor, verbose=0)[0]
        prob_map = np.squeeze(pred)
        if prob_map.ndim == 3:
            prob_map = prob_map[:,:,0]
        mask = (prob_map > THRESHOLD).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask,(image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = image.copy()
        overlay[mask_resized > 0] = [0,255,0]
        overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        person_percent = np.mean(prob_map > THRESHOLD) * 100
        background_percent = 100 - person_percent
        vals = prob_map[prob_map > THRESHOLD]
        mean_confidence = np.mean(vals) * 100 if vals.size > 0 else 0
        uncertain = np.logical_and(prob_map > 0.4, prob_map < 0.6)
        uncertain_percent = np.mean(uncertain) * 100
        metrics = {
            "person": person_percent,
            "background": background_percent,
            "confidence": mean_confidence,
            "uncertain": uncertain_percent
        }
        return mask_resized, overlay, metrics
class Worker(QThread):
    progress_changed = pyqtSignal(int)
    finished = pyqtSignal()
    result_ready = pyqtSignal(object)
    metrics_ready = pyqtSignal(object)
    preview_ready = pyqtSignal(object)
    def __init__(self, app_ref):
        super().__init__()
        self.app = app_ref
    def run(self):
        if self.app.input_dir and self.app.output_dir:
            valid_files = [
                f for f in os.listdir(self.app.input_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if len(valid_files) == 0:
                self.finished.emit()
                return
            for i, filename in enumerate(valid_files):
                path = os.path.join(self.app.input_dir, filename)
                image = self.app.imread_unicode(path)

                if image is None:
                    continue

                orig, mask, overlay, metrics = self.app.process_single_image(image, path)
                metrics["filename"] = filename
                self.metrics_ready.emit(metrics)

                save_path = os.path.join(self.app.output_dir, filename)
                self.app.imwrite_unicode(save_path, overlay)
                self.preview_ready.emit(overlay)
                self.progress_changed.emit(i + 1)

            self.finished.emit()

        elif self.app.image is not None:
            orig, mask, overlay, metrics = self.app.process_single_image(
                self.app.image,
                self.app.current_image_path
            )
            self.result_ready.emit((orig, mask, overlay, metrics))
            self.finished.emit()
class SegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_state()
        try:
            self.seg_model = SegmentationModel(MODEL_PATH)
        except Exception as e:
            QMessageBox.critical(self, "Model Error", str(e))
            sys.exit(1)
        self.init_ui()
        self.init_connections()
        self.run_mode = None
    def init_state(self):
        self.image = None
        self.input_dir = None
        self.output_dir = None
        self.all_metrics = []
    def init_ui(self):
        self.setWindowTitle("Нысанды тану бағдарламасы")
        self.setGeometry(150, 100, 1200, 750)
        self.tabs = QTabWidget()
        self.create_widgets()
        self.create_single_tab()
        self.create_batch_tab()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
    def create_widgets(self):
        self.input_label = QLabel("Input")
        self.mask_label = QLabel("Mask")
        self.output_label = QLabel("Output")
        self.btn_open_output = QPushButton("Open Output Folder")
        for lbl in [self.input_label, self.mask_label, self.output_label]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(350, 350)
            lbl.setStyleSheet("border:1px solid #333; background:#1e1e1e; color:#aaa;")
        style = """QPushButton {background-color: #7d2ae8;color: white;border-radius: 8px;font-weight: bold;}
QPushButton:hover {background-color: #5f1fc2;}"""
        self.btn_load_image = QPushButton("Load Image")
        self.btn_predict = QPushButton("Predict")
        self.btn_input_folder = QPushButton("Input Folder")
        self.btn_output_folder = QPushButton("Output Folder")
        self.btn_predict_batch = QPushButton("Run Batch")
        buttons = [self.btn_load_image,self.btn_predict,self.btn_input_folder,self.btn_output_folder,self.btn_predict_batch,self.btn_open_output
]
        for b in buttons:
            b.setFixedSize(BUTTON_WIDTH, BUTTON_HEIGHT)
            b.setStyleSheet(style)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(400)
        self.progress.setValue(0)
        self.progress.setFormat("%v/%m")
        self.metrics_container = QVBoxLayout()
        self.metrics_widget_content = QWidget()
        self.metrics_widget_content.setLayout(self.metrics_container)
        self.metrics_scroll = QScrollArea()
        self.metrics_scroll.setWidgetResizable(True)
        self.metrics_scroll.setWidget(self.metrics_widget_content)
        self.metrics_scroll.setStyleSheet("border:1px solid #444; padding:5px;")
        self.batch_metrics_container = QVBoxLayout()
        self.batch_metrics_widget_content = QWidget()
        self.batch_metrics_widget_content.setLayout(self.batch_metrics_container)
        self.batch_metrics_scroll = QScrollArea()
        self.batch_metrics_scroll.setWidgetResizable(True)
        self.batch_metrics_scroll.setWidget(self.batch_metrics_widget_content)
        self.batch_metrics_scroll.setStyleSheet("border:1px solid #444; padding:5px;")
        self.preview_count = 0

        self.batch_preview_container = QGridLayout()
        self.batch_preview_widget = QWidget()
        self.batch_preview_widget.setLayout(self.batch_preview_container)

        self.batch_preview_scroll = QScrollArea()
        self.batch_preview_scroll.setWidgetResizable(True)
        self.batch_preview_scroll.setWidget(self.batch_preview_widget)
        self.batch_preview_scroll.setFixedHeight(220)
        self.batch_preview_scroll.setStyleSheet("""
    QScrollArea {
        border: 1px solid #555;
        background-color: #2b2b2b;
    }
""")
    def create_single_tab(self):
        single_tab = QWidget()
        layout = QVBoxLayout()
        images_layout = QHBoxLayout()
        input_box = QVBoxLayout()
        mask_box = QVBoxLayout()
        output_box = QVBoxLayout()
        input_box.addWidget(QLabel("Input"))
        input_box.addWidget(self.input_label)
        mask_box.addWidget(QLabel("Mask"))
        mask_box.addWidget(self.mask_label)
        output_box.addWidget(QLabel("Output"))
        output_box.addWidget(self.output_label)
        images_layout.addLayout(input_box)
        images_layout.addLayout(mask_box)
        images_layout.addLayout(output_box)
        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_load_image)
        buttons.addWidget(self.btn_predict)
        layout.addLayout(images_layout)
        layout.addLayout(buttons)
        layout.addWidget(self.metrics_scroll, stretch=1)
        single_tab.setLayout(layout)
        self.tabs.addTab(single_tab, "Single Image")
    def create_batch_tab(self):
        batch_tab = QWidget()
        layout = QVBoxLayout()
        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_input_folder)
        buttons.addWidget(self.btn_output_folder)
        buttons.addWidget(self.btn_predict_batch)
        buttons.addWidget(self.btn_open_output)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        preview_title = QLabel(f"Overlay Preview Results — Showing first {MAX_PREVIEW_IMAGES} overlay results")
        preview_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(preview_title)
        layout.addWidget(self.batch_preview_scroll)
        layout.addWidget(self.batch_metrics_scroll, stretch=1)
        batch_tab.setLayout(layout)
        self.tabs.addTab(batch_tab, "Batch Processing")
    def init_connections(self):
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_predict.clicked.connect(self.predict)
        self.btn_input_folder.clicked.connect(self.select_input_folder)
        self.btn_output_folder.clicked.connect(self.select_output_folder)
        self.btn_predict_batch.clicked.connect(self.predict_batch)
        self.btn_open_output.clicked.connect(self.open_output_folder)
    def predict(self):
        try:
            self.run_mode = "single"
            self.all_metrics = []
            if self.input_dir:
                QMessageBox.warning(self, "Warning", "Single mode works only with one image")
                return
            if self.image is None:
                QMessageBox.warning(self, "Warning", "No input selected")
                return
            self.progress.setMaximum(1)
            self.progress.setValue(0)
            self.btn_predict.setEnabled(False)
            self.btn_predict_batch.setEnabled(False)
            self.worker = Worker(self)
            self.worker.progress_changed.connect(self.progress.setValue)
            self.worker.result_ready.connect(self.handle_result)
            self.worker.metrics_ready.connect(self.collect_metrics)
            self.worker.finished.connect(self.on_finished)
            self.worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Predict Error", str(e))
            print("ERROR:", e)
    def predict_batch(self):
        try:
            self.run_mode = "batch"
            self.all_metrics = []
            if not self.input_dir:
                QMessageBox.warning(self, "Warning", "Select input folder")
                return
            if not self.output_dir:
                QMessageBox.warning(self, "Warning", "Select output folder")
                return
            files = [f for f in os.listdir(self.input_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not files:
                QMessageBox.warning(self, "Warning", "Folder is empty")
                return
            self.progress.setMaximum(len(files))
            self.progress.setValue(0)
            self.btn_predict.setEnabled(False)
            self.btn_predict_batch.setEnabled(False)
            self.clear_batch_preview()
            self.worker = Worker(self)
            self.worker.progress_changed.connect(self.progress.setValue)
            self.worker.result_ready.connect(self.handle_result)
            self.worker.metrics_ready.connect(self.collect_metrics)
            self.worker.finished.connect(self.on_finished)
            self.worker.preview_ready.connect(self.add_batch_preview)
            self.worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Batch Error", str(e))
    def on_finished(self):
        self.btn_predict.setEnabled(True)
        self.btn_predict_batch.setEnabled(True)

        if not self.all_metrics:
            return

        if self.run_mode == "batch" and self.all_metrics:
            agg = self.aggregate_metrics()
            self.update_batch_metrics_ui(agg)
            self.save_metrics_csv()

            QMessageBox.information(
            self,
            "Completed",
            "Batch processing completed successfully."
            )
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
        self, "Open Image", "", "Images (*.png *.jpg *.jpeg)"
    )
        if not file_path:
            return
        self.image = self.imread_unicode(file_path)
        self.input_dir = None
        self.current_image_path = file_path
        if self.image is None:
            QMessageBox.critical(self, "Error", "Failed to load image")
            return
        self.show_image(self.image, self.input_label)
        self.mask_label.clear()
        self.output_label.clear()
    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_dir = folder
            self.image = None
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = folder
    def imread_unicode(self, path):
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    def show_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.AspectRatioMode.KeepAspectRatio))  
    def process_single_image(self, image, image_path=None):
        gt_mask = self.load_gt_mask(image_path)
        print("Image path:", image_path)
        print("Mask found:", gt_mask is not None)
        mask, overlay, metrics = self.seg_model.predict(image)

        if gt_mask is not None:
            if gt_mask.shape != mask.shape:
                gt_mask = cv2.resize(
                    gt_mask,
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            iou = self.compute_iou(mask, gt_mask)
            dice = self.compute_dice(mask, gt_mask)
            pixel_accuracy = self.compute_pixel_accuracy(mask, gt_mask)
            precision = self.compute_precision(mask, gt_mask)
            recall = self.compute_recall(mask, gt_mask)
            f1 = self.compute_f1(precision, recall)

            metrics["iou"] = iou * 100
            metrics["dice"] = dice * 100
            metrics["pixel_accuracy"] = pixel_accuracy * 100
            metrics["precision"] = precision * 100
            metrics["recall"] = recall * 100
            metrics["f1"] = f1 * 100

        return image, mask, overlay, metrics
    def imwrite_unicode(self, path, img):
        ext = os.path.splitext(path)[1]
        success, encoded = cv2.imencode(ext, img)
        if success:
            encoded.tofile(path)
    def handle_result(self, data):
        orig, mask, overlay, metrics = data
        self.show_image(orig, self.input_label)
        self.show_image(mask, self.mask_label)
        self.show_image(overlay, self.output_label)
        if self.run_mode == "single":
            self.update_metrics_ui(metrics)
    def collect_metrics(self, metrics):
        if self.run_mode == "batch":
            self.all_metrics.append(metrics)
    def aggregate_metrics(self):
        if not self.all_metrics:
            return None

        def mean(key):
            values = [m[key] for m in self.all_metrics if key in m]
            return np.mean(values) if values else 0

        return {
            "mean_person": mean("person"),
            "mean_background": mean("background"),
            "mean_confidence": mean("confidence"),
            "mean_uncertain": mean("uncertain"),
            "mean_iou": mean("iou"),
            "mean_dice": mean("dice"),
            "mean_pixel_accuracy": mean("pixel_accuracy"),
            "mean_precision": mean("precision"),
            "mean_recall": mean("recall"),
            "mean_f1": mean("f1")
        }
    def create_metric_row(self, name, value, inverse_color=False):
        container = QWidget()
        layout = QHBoxLayout()
        label = QLabel(name)
        label.setFixedWidth(120)
        bar = QProgressBar()
        bar.setTextVisible(False)
        bar.setFixedHeight(15)
        raw_value = float(value) if value is not None else 0.0
        bar.setValue(max(0, min(100, int(raw_value))))
        value_label = QLabel(f"{raw_value:.1f}%")
        value_label.setFixedWidth(60)
        color_value = 100 - raw_value if inverse_color else raw_value
        if color_value <= 40:
            color = "#f44336"
        elif color_value <= 80:
            color = "#ff9800"   
        else:
            color = "#4caf50"
        bar.setStyleSheet(f"""
    QProgressBar {{
        border: none;
        background-color: #3a3a3a;
    }}
    QProgressBar::chunk {{
        background-color: {color};
    }}
""")

        container.setStyleSheet("""
    QWidget {
        border: none;
        background: transparent;
    }
    QLabel {
        border: none;
        background: transparent;
    }
""")

        layout.addWidget(label)
        layout.addWidget(bar)
        layout.addWidget(value_label)
        container.setLayout(layout)
        return container
    def create_framed_block(self, layout):
        block = QWidget()
        block.setLayout(layout)
        block.setStyleSheet("""
        QWidget {
            border: 1px solid #555;
            background-color: #2b2b2b;
        }
        QLabel {
            border: none;
            background: transparent;
        }
        QProgressBar {
            border: none;
            background-color: #3a3a3a;
        }
    """)
        return block

    def create_plain_title(self, text):
        label = QLabel(text)
        label.setStyleSheet("""
        QLabel {
            border: none;
            background: transparent;
            font-weight: bold;
            font-size: 14px;
        }
    """)
        return label
    def create_class_distribution_widget(self, person_value, background_value):
        container = QWidget()
        layout = QHBoxLayout()
        def create_vertical_bar(name, value):
            box = QVBoxLayout()
            value_label = QLabel(f"{value:.1f}%")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            bar = QProgressBar()
            bar.setOrientation(Qt.Orientation.Vertical)
            bar.setRange(0, 100)
            bar.setValue(max(0, min(100, int(value))))
            bar.setTextVisible(False)
            bar.setFixedSize(45, 120)
            name_label = QLabel(name)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(value_label)
            box.addWidget(bar, alignment=Qt.AlignmentFlag.AlignCenter)
            box.addWidget(name_label)
            wrapper = QWidget()
            wrapper.setLayout(box)
            return wrapper
        layout.addWidget(create_vertical_bar("Person", person_value))
        layout.addWidget(create_vertical_bar("Background", background_value))
        container.setLayout(layout)
        container.setStyleSheet("""
    QWidget {
        border: none;
        background: transparent;
    }
    QLabel {
        border: none;
        background: transparent;
    }
    QProgressBar {
        border: none;
        background-color: #3a3a3a;
    }
    QProgressBar::chunk {
        background-color: #c77dff;
    }
""")
        return container
    def update_metrics_ui(self, metrics):
        while self.metrics_container.count():
            item = self.metrics_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        title = self.create_plain_title("Single Image Metrics")
        self.metrics_container.addWidget(title)
        grid = QGridLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        left_title = QLabel("Segmentation Metrics")
        left_title.setStyleSheet("font-weight: bold;")
        right_title = QLabel("Additional Indicators")
        right_title.setStyleSheet("font-weight: bold;")
        left_col.addWidget(left_title)
        left_col.addWidget(self.create_metric_row("IoU", metrics.get("iou", 0)))
        left_col.addWidget(self.create_metric_row("Dice", metrics.get("dice", 0)))
        left_col.addWidget(self.create_metric_row("Pixel Acc", metrics.get("pixel_accuracy", 0)))
        left_col.addWidget(self.create_metric_row("Precision", metrics.get("precision", 0)))
        left_col.addWidget(self.create_metric_row("Recall", metrics.get("recall", 0)))
        left_col.addWidget(self.create_metric_row("F1-score", metrics.get("f1", 0)))
        left_col.addStretch()
        right_col.addWidget(right_title)
        right_col.addWidget(self.create_class_distribution_widget(
        metrics.get("person", 0),
        metrics.get("background", 0)
    ))
        right_col.addWidget(self.create_metric_row("Confidence", metrics.get("confidence", 0)))
        right_col.addWidget(self.create_metric_row("Uncertain", metrics.get("uncertain", 0), inverse_color=True))
        right_col.addStretch()
        left_widget = self.create_framed_block(left_col)
        right_widget = self.create_framed_block(right_col)
        grid.addWidget(left_widget, 0, 0)
        grid.addWidget(right_widget, 0, 1)
        grid_widget = QWidget()
        grid_widget.setLayout(grid)
        self.metrics_container.addWidget(grid_widget)
        self.metrics_container.addStretch()
    def update_batch_metrics_ui(self, agg):
        while self.batch_metrics_container.count():
            item = self.batch_metrics_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        title = QLabel("Batch Metrics")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        self.batch_metrics_container.addWidget(title)
        grid = QGridLayout()
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        left_title = QLabel("Mean Segmentation Metrics")
        left_title.setStyleSheet("font-weight: bold;")
        right_title = QLabel("Mean Additional Indicators")
        right_title.setStyleSheet("font-weight: bold;")
        left_col.addWidget(left_title)
        left_col.addWidget(self.create_metric_row("IoU", agg["mean_iou"]))
        left_col.addWidget(self.create_metric_row("Dice", agg["mean_dice"]))
        left_col.addWidget(self.create_metric_row("Pixel Acc", agg["mean_pixel_accuracy"]))
        left_col.addWidget(self.create_metric_row("Precision", agg["mean_precision"]))
        left_col.addWidget(self.create_metric_row("Recall", agg["mean_recall"]))
        left_col.addWidget(self.create_metric_row("F1-score", agg["mean_f1"]))
        left_col.addStretch()
        right_col.addWidget(right_title)
        right_col.addWidget(self.create_class_distribution_widget(
        agg["mean_person"],
        agg["mean_background"]))
        right_col.addWidget(self.create_metric_row("Confidence", agg["mean_confidence"]))
        right_col.addWidget(self.create_metric_row("Uncertain", agg["mean_uncertain"], inverse_color=True))
        right_col.addStretch()
        left_widget = self.create_framed_block(left_col)
        right_widget = self.create_framed_block(right_col)
        grid.addWidget(left_widget, 0, 0)
        grid.addWidget(right_widget, 0, 1)
        grid_widget = QWidget()
        grid_widget.setLayout(grid)
        self.batch_metrics_container.addWidget(grid_widget)
        self.batch_metrics_container.addStretch()
    def compute_iou(self, pred, gt):
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)
        pred = (pred > 127)
        gt = (gt > 127)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            return 0.0
        return intersection / union
    def compute_dice(self, pred, gt):
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)
        pred = (pred > 127)
        gt = (gt > 127)
        intersection = np.logical_and(pred, gt).sum()
        denom = pred.sum() + gt.sum()
        if denom == 0:
            return 0.0
        return (2.0 * intersection) / denom
    def compute_pixel_accuracy(self, pred, gt):
        pred = np.squeeze(pred) > 127
        gt = np.squeeze(gt) > 127

        correct = np.sum(pred == gt)
        total = pred.size

        if total == 0:
            return 0.0

        return correct / total

    def compute_precision(self, pred, gt):
        pred = np.squeeze(pred) > 127
        gt = np.squeeze(gt) > 127

        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, np.logical_not(gt)).sum()

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    def compute_recall(self, pred, gt):
        pred = np.squeeze(pred) > 127
        gt = np.squeeze(gt) > 127

        tp = np.logical_and(pred, gt).sum()
        fn = np.logical_and(np.logical_not(pred), gt).sum()

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    def compute_f1(self, precision, recall):
        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)
    def load_gt_mask(self, image_path):
        if image_path is None:
            return None
        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)
        image_dir = os.path.dirname(image_path)
        dataset_dir = os.path.dirname(image_dir)
        mask_dir = os.path.join(dataset_dir, "masks")
        possible_paths = [
        os.path.join(mask_dir, name + ".png"),
        os.path.join(mask_dir, name + ".jpg"),
        os.path.join(mask_dir, name + ".jpeg"),
        os.path.join(mask_dir, filename),
        os.path.join(mask_dir, name + "_mask.png"),
        os.path.join(mask_dir, name + "_mask.jpg"),]
        print("Looking for mask:")
        for path in possible_paths:
            print(path)
            if os.path.exists(path):
                stream = np.fromfile(path, dtype=np.uint8)
                mask = cv2.imdecode(stream, cv2.IMREAD_GRAYSCALE)
                print("Mask loaded:", path)
                print("Mask unique values:", np.unique(mask))
                return mask
        print("Mask not found")
        return None
    def clear_batch_preview(self):
        self.preview_count = 0

        while self.batch_preview_container.count():
            item = self.batch_preview_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


    def add_batch_preview(self, overlay):
        if self.preview_count >= MAX_PREVIEW_IMAGES:
            return

        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFixedSize(105, 80)
        label.setStyleSheet("""
        QLabel {
            border: 1px solid #555;
            background-color: #1e1e1e;
        }
    """)

        self.show_image(overlay, label)
        cols = self.get_preview_columns()

        row = self.preview_count // cols
        col = self.preview_count % cols

        self.batch_preview_container.addWidget(label, row, col)
        self.preview_count += 1

    def get_preview_columns(self):
        width = self.batch_preview_scroll.width()

        self.batch_preview_container.setSpacing(0)
        self.batch_preview_container.setContentsMargins(0, 0, 0, 0)

        if width < 600:
            return 4
        elif width < 900:
            return 5
        elif width < 1200:
            return 6
        else:
            return 8
    def save_metrics_csv(self):
        if not self.output_dir or not self.all_metrics:
            return

        csv_path = os.path.join(self.output_dir, "batch_metrics.csv")

        headers = [ "filename","IoU","Dice","Pixel Accuracy","Precision","Recall","F1-score"]

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(headers)

            for m in self.all_metrics:
                writer.writerow([
                    m.get("filename", ""),
                    f"{m.get('iou', 0):.2f}",
                    f"{m.get('dice', 0):.2f}",
                    f"{m.get('pixel_accuracy', 0):.2f}",
                    f"{m.get('precision', 0):.2f}",
                    f"{m.get('recall', 0):.2f}",
                    f"{m.get('f1', 0):.2f}"
                ])
    def open_output_folder(self):
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Output folder is not selected")
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_dir))
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SegmentationApp()
    window.show()
    sys.exit(app.exec())