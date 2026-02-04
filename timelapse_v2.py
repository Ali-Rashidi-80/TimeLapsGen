import sys
import random
import time
import cv2
import os
import math
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QComboBox, QProgressBar,
    QFileDialog, QMessageBox, QTextEdit, QGroupBox, QFormLayout, QToolTip, QStatusBar,
    QSlider
)
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from ultralytics import YOLO
import shutil

# Enable OpenCL
try:
    import cv2.ocl as ocl
    if ocl.haveOpenCL():
        ocl.setUseOpenCL(True)
except:
    pass

def is_cuda_available():
    try:
        import cv2.cuda as cuda
        return cuda.getCudaEnabledDeviceCount() > 0
    except:
        return False

class FrameCaptureThread(QThread):
    frameCaptured = pyqtSignal(object)
    errorOccurred = pyqtSignal(str)
    progressUpdated = pyqtSignal(int)
    def __init__(self, source, interval=1, buffer_size=1000):
        super().__init__()
        self.source = source
        self.interval = interval
        self.buffer_size = buffer_size
        self.frames = []
        self.running = True
    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.errorOccurred.emit("خطا در اتصال به منبع. آدرس یا دستگاه را بررسی کنید.")
            return
        count = 0
        while self.running and len(self.frames) < self.buffer_size:
            ret, frame = cap.read()
            if ret:
                self.frames.append(frame)
                self.frameCaptured.emit(frame)
                count += 1
                self.progressUpdated.emit(count)
            else:
                break
            time.sleep(self.interval)
        cap.release()

class FrameExtractorThread(QThread):
    progressUpdated = pyqtSignal(int)
    finished = pyqtSignal(list)
    errorOccurred = pyqtSignal(str)
    def __init__(self, source, temp_dir):
        super().__init__()
        self.source = source
        self.temp_dir = temp_dir
        self.running = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.errorOccurred.emit("خطا در باز کردن ویدیو")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_paths = []
            count = 0
            while self.running:
                ret, frame = cap.read()
                if not ret: break
                path = os.path.join(self.temp_dir, f"frame_{count:06d}.jpg")
                cv2.imwrite(path, frame)
                frame_paths.append(path)
                count += 1
                self.progressUpdated.emit(int(count / total * 100))
            cap.release()
            self.finished.emit(frame_paths)
        except Exception as e:
            self.errorOccurred.emit(str(e))

class YoloClassificationThread(QThread):
    progressUpdated = pyqtSignal(int)
    finished = pyqtSignal(dict)
    errorOccurred = pyqtSignal(str)
    def __init__(self, frame_paths, model_path):
        super().__init__()
        self.frame_paths = frame_paths
        self.model_path = model_path
        self.running = True

    def run(self):
        try:
            model = YOLO(self.model_path)
            categories = {}
            total = len(self.frame_paths)
            for i, path in enumerate(self.frame_paths):
                if not self.running: break
                results = model(path)[0]
                if results.boxes:
                    cls = results.boxes.cls[0].item() if results.boxes.cls.size(0) > 0 else -1
                    cls_name = results.names.get(int(cls), "unknown")
                else:
                    cls_name = "unknown"
                if cls_name not in categories:
                    categories[cls_name] = []
                categories[cls_name].append(path)
                self.progressUpdated.emit(int((i + 1) / total * 100))
            self.finished.emit(categories)
        except Exception as e:
            self.errorOccurred.emit(str(e))

class TimelapseGeneratorThread(QThread):
    progressUpdated = pyqtSignal(int)
    finished = pyqtSignal(bool)
    errorOccurred = pyqtSignal(str)

    def __init__(self, source, selected_indices, fps, output_path, is_cuda, frames=None,
                 hold_mode="ثابت", min_hold=1, max_hold=5, frame_paths=None):
        super().__init__()
        self.source = source
        self.selected_indices = selected_indices
        self.fps = fps
        self.output_path = output_path
        self.is_cuda = is_cuda
        self.frames = frames
        self.hold_mode = hold_mode
        self.min_hold = min_hold
        self.max_hold = max_hold
        self.frame_paths = frame_paths  # For YOLO mode
        self.running = True

    def get_hold_sequence(self, n):
        if n == 0:
            return []
        if self.hold_mode == "ثابت":
            return [self.min_hold] * n
        elif self.hold_mode == "تدریجی افزایشی":
            return [self.min_hold + int((self.max_hold - self.min_hold) * i / (n - 1)) for i in range(n)]
        elif self.hold_mode == "تدریجی کاهشی":
            return [self.max_hold - int((self.max_hold - self.min_hold) * i / (n - 1)) for i in range(n)]
        elif self.hold_mode == "موج‌دار (Pulse)":
            mid = (self.min_hold + self.max_hold) / 2
            amp = (self.max_hold - self.min_hold) / 2
            return [int(mid + amp * math.sin(2 * math.pi * i / max(1, n-1))) for i in range(n)]
        elif self.hold_mode == "رندوم هوشمند":
            return [random.randint(self.min_hold, self.max_hold) for _ in range(n)]
        return [self.min_hold] * n

    def run(self):
        try:
            n = len(self.selected_indices)
            hold_sequence = self.get_hold_sequence(n)
            total = sum(hold_sequence)
            current = 0

            if self.frame_paths:  # YOLO mode, read from images
                first_img = cv2.imread(self.frame_paths[0])
                h, w = first_img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
                for i, idx in enumerate(self.selected_indices):
                    if not self.running: break
                    frame = cv2.imread(self.frame_paths[idx])
                    for _ in range(hold_sequence[i]):
                        writer.write(frame)
                        current += 1
                        self.progressUpdated.emit(int(current / total * 100))
                writer.release()
                self.finished.emit(True)
                return

            if self.source is None:  # Live source
                if not self.frames:
                    self.errorOccurred.emit("هیچ فریمی موجود نیست")
                    self.finished.emit(False)
                    return
                h, w = self.frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
                for i, idx in enumerate(self.selected_indices):
                    if not self.running: break
                    frame = self.frames[idx]
                    for _ in range(hold_sequence[i]):
                        writer.write(frame)
                        current += 1
                        self.progressUpdated.emit(int(current / total * 100))
                writer.release()
                self.finished.emit(True)
                return

            # File source with CUDA
            if self.is_cuda:
                try:
                    import cv2.cudacodec as cudacodec
                    cap = cv2.VideoCapture(self.source)
                    if not cap.isOpened(): raise Exception("خطا در باز کردن ویدیو")
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    codec = cudacodec.VideoWriterCodec.H264
                    writer = cudacodec.createVideoWriter(self.output_path, codec, self.fps, (w, h))
                    reader = cudacodec.createVideoReader(self.source)
                    for i, idx in enumerate(self.selected_indices):
                        if not self.running: break
                        reader.seekToFrame(idx)
                        gpu_frame = reader.nextFrame()
                        if gpu_frame.empty(): break
                        for _ in range(hold_sequence[i]):
                            writer.write(gpu_frame)
                            current += 1
                            self.progressUpdated.emit(int(current / total * 100))
                    writer.release()
                    reader.release()
                    self.finished.emit(True)
                    return
                except Exception as e:
                    self.errorOccurred.emit(f"خطا در CUDA: {str(e)}. استفاده از CPU.")
                    self.is_cuda = False

            # CPU Fallback
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.errorOccurred.emit("خطا در باز کردن ویدیو")
                return
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
            for i, idx in enumerate(self.selected_indices):
                if not self.running: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    for _ in range(hold_sequence[i]):
                        writer.write(frame)
                        current += 1
                        self.progressUpdated.emit(int(current / total * 100))
            writer.release()
            cap.release()
            self.finished.emit(True)
        except Exception as e:
            self.errorOccurred.emit(str(e))
            self.finished.emit(False)

class TimelapseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("تایم‌لپس ساز حرفه‌ای")
        self.setGeometry(100, 100, 1000, 750)
        self.setWindowIcon(QIcon.fromTheme("video-x-generic"))
        self.total_frames = 0
        self.frames = []
        self.capture_thread = None
        self.extractor_thread = None
        self.yolo_thread = None
        self.generator_thread = None
        self.temp_video_path = 'temp_timelapse.mp4'
        self.temp_dir = 'temp_frames'
        self.is_cuda = is_cuda_available()
        self.source_type = None
        self.source_path = None
        self.frame_paths = []
        self.categories = {}

        self.statusBar().setStyleSheet("color: #a9d6e0; font-style: italic;")
        self.update_acceleration_status()

        self.setup_ui()
        self.apply_dark_blue_theme()

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.update_slider_position)
        self.player.durationChanged.connect(self.update_slider_range)

    def update_acceleration_status(self):
        status = "شتاب‌دهنده: CUDA فعال" if self.is_cuda else "شتاب‌دهنده: CPU"
        self.statusBar().showMessage(status)

    def apply_dark_blue_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #0d1b2a; color: #e0e1dd; }
            QTabWidget::pane { border: 1px solid #1b263b; background: #0d1b2a; }
            QTabBar::tab { background: #1b263b; color: #e0e1dd; padding: 12px; border-radius: 6px; margin: 2px; }
            QTabBar::tab:selected { background: #278ea5; color: white; font-weight: bold; }
            QTabBar::tab:hover { background: #1f5f8b; }
            QLabel { color: #e0e1dd; font-size: 13px; }
            QLineEdit, QSpinBox, QComboBox { 
                background: #1b263b; color: #e0e1dd; border: 1px solid #278ea5; 
                padding: 8px; border-radius: 6px; font-size: 13px;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border: 2px solid #21a0b8; }
            QPushButton { 
                background: #278ea5; color: white; border: none; padding: 10px; 
                border-radius: 6px; font-weight: bold; font-size: 13px;
            }
            QPushButton:hover { background: #21a0b8; }
            QPushButton:pressed { background: #1f5f8b; }
            QPushButton:disabled { background: #1b263b; color: #a9d6e0; }
            QProgressBar { 
                background: #1b263b; border: 1px solid #278ea5; border-radius: 6px; 
                text-align: center; color: white; font-weight: bold;
            }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #278ea5, stop:1 #21a0b8); border-radius: 5px; }
            QTextEdit { background: #1b263b; color: #a9d6e0; border: 1px solid #278ea5; border-radius: 6px; padding: 8px; }
            QGroupBox { 
                font-weight: bold; color: #21a0b8; border: 1px solid #278ea5; 
                border-radius: 8px; margin-top: 10px; padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QToolTip { 
                background: #1f5f8b; color: white; border: 1px solid #278ea5; 
                padding: 6px; border-radius: 6px; font-size: 12px;
            }
            QStatusBar { background: #1b263b; color: #a9d6e0; }
        """)
        QToolTip.setFont(QFont("Segoe UI", 11))

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        title = QLabel("تایم‌لپس ساز حرفه‌ای")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #21a0b8; margin: 10px;")
        main_layout.addWidget(title)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.source_tab = QWidget()
        self.settings_tab = QWidget()
        self.preview_tab = QWidget()

        self.tabs.addTab(self.source_tab, "منبع")
        self.tabs.addTab(self.settings_tab, "تنظیمات")
        self.tabs.addTab(self.preview_tab, "پیش‌نمایش")

        self.setup_source_tab()
        self.setup_settings_tab()
        self.setup_preview_tab()

    def setup_source_tab(self):
        layout = QVBoxLayout(self.source_tab)
        layout.setSpacing(15)

        group = QGroupBox("انتخاب منبع ویدیو")
        group_layout = QFormLayout()
        group_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        group_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        group_layout.setSpacing(12)

        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["فایل ویدیو", "وبکم سیستم", "آدرس IP دوربین"])
        self.source_type_combo.setToolTip("نوع منبعی که می‌خواهید از آن تایم‌لپس بسازید")
        group_layout.addRow("نوع منبع:", self.source_type_combo)

        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("مثلاً: C:\\video.mp4 یا rtsp://...")
        self.source_input.setToolTip("مسیر فایل یا آدرس استریم را وارد کنید")
        group_layout.addRow("مسیر / آدرس:", self.source_input)

        browse_btn = QPushButton("انتخاب فایل")
        browse_btn.clicked.connect(self.browse_file)
        hbox = QHBoxLayout()
        hbox.addWidget(browse_btn)
        group_layout.addRow(hbox)

        group.setLayout(group_layout)
        layout.addWidget(group)

        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("بارگذاری منبع")
        self.load_btn.clicked.connect(self.load_source)
        self.load_btn.setStyleSheet("background: #21a0b8; padding: 12px; font-size: 14px;")
        btn_layout.addWidget(self.load_btn)

        self.stop_btn = QPushButton("توقف")
        self.stop_btn.clicked.connect(self.stop_loading)
        self.stop_btn.setStyleSheet("background: #c94f4f; padding: 12px; font-size: 14px;")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

        self.status_label = QLabel("آماده")
        self.status_label.setStyleSheet("color: #a9d6e0; font-style: italic;")
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("در حال بارگذاری... %p%")
        layout.addWidget(self.progress)
        layout.addStretch()

    def setup_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)
        layout.setSpacing(15)

        # --- تنظیمات خروجی ---
        group1 = QGroupBox("تنظیمات خروجی")
        form1 = QFormLayout()
        form1.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form1.setSpacing(12)

        self.frame_count = QSpinBox()
        self.frame_count.setRange(10, 10000)
        self.frame_count.setValue(120)
        self.frame_count.setToolTip("تعداد فریم‌های نهایی در ویدیو تایم‌لپس")
        form1.addRow("تعداد فریم‌ها:", self.frame_count)

        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(30)
        self.fps.setToolTip("سرعت پخش ویدیو (فریم در ثانیه)")
        form1.addRow("سرعت پخش (FPS):", self.fps)

        group1.setLayout(form1)
        layout.addWidget(group1)

        # --- تنظیمات مکث حرفه‌ای ---
        group_hold = QGroupBox("مکث حرفه‌ای روی هر فریم")
        hold_layout = QFormLayout()
        hold_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        hold_layout.setSpacing(10)

        self.hold_mode = QComboBox()
        hold_modes = [
            ("ثابت", "هر فریم به تعداد ثابت تکرار می‌شود"),
            ("تدریجی افزایشی", "مکث از کم به زیاد — شروع سریع، پایان آرام"),
            ("تدریجی کاهشی", "مکث از زیاد به کم — شروع آرام، پایان سریع"),
            ("موج‌دار (Pulse)", "مکث به صورت موجی بالا و پایین می‌رود"),
            ("رندوم هوشمند", "مکث تصادفی در بازه مشخص — طبیعی و پویا")
        ]
        for mode, tip in hold_modes:
            self.hold_mode.addItem(mode)
        self.hold_mode.currentIndexChanged.connect(
            lambda idx: self.hold_mode.setToolTip(hold_modes[idx][1])
        )
        self.hold_mode.setToolTip(hold_modes[0][1])
        hold_layout.addRow("حالت مکث:", self.hold_mode)

        min_max_layout = QHBoxLayout()
        self.min_hold = QSpinBox()
        self.min_hold.setRange(1, 50)
        self.min_hold.setValue(1)
        self.min_hold.setToolTip("حداقل تعداد تکرار هر فریم (در حالت‌های متغیر)")
        min_max_layout.addWidget(self.min_hold)
        self.max_hold = QSpinBox()
        self.max_hold.setRange(1, 50)
        self.max_hold.setValue(5)
        self.max_hold.setToolTip("حداکثر تعداد تکرار هر فریم (در حالت‌های متغیر)")
        min_max_layout.addWidget(self.max_hold)
        hold_layout.addRow("بازه مکث (فریم):", min_max_layout)

        group_hold.setLayout(hold_layout)
        layout.addWidget(group_hold)

        # --- تنظیمات YOLO ---
        group_yolo = QGroupBox("تنظیمات هوشمند YOLO")
        yolo_layout = QFormLayout()
        yolo_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        yolo_layout.setSpacing(10)

        self.yolo_model = QComboBox()
        yolo_models = [
            ("YOLO11n (سریع و سبک - برای دستگاه‌های ضعیف)", "yolo11n.pt"),
            ("YOLO11s (تعادل سرعت و دقت - توصیه شده)", "yolo11s.pt"),
            ("YOLO11m (دقت بالا - برای ویدیوهای پیچیده)", "yolo11m.pt"),
            ("YOLO11l (دقت بسیار بالا - کندتر)", "yolo11l.pt"),
            ("YOLO11x (حداکثر دقت - برای سیستم‌های قدرتمند)", "yolo11x.pt")
        ]
        for label, path in yolo_models:
            self.yolo_model.addItem(label)
        self.yolo_model.setToolTip("انتخاب مدل YOLO بر اساس سرعت و دقت")
        yolo_layout.addRow("مدل YOLO:", self.yolo_model)

        self.yolo_classes = QLineEdit()
        self.yolo_classes.setPlaceholderText("مثلاً: person,car - خالی برای همه")
        self.yolo_classes.setToolTip("کلاس‌های خاص برای فیلتر (جدا با کاما) - خالی برای همه کلاس‌ها")
        yolo_layout.addRow("کلاس‌های خاص:", self.yolo_classes)

        self.yolo_conf = QSpinBox()
        self.yolo_conf.setRange(0, 100)
        self.yolo_conf.setValue(50)
        self.yolo_conf.setToolTip("آستانه اطمینان تشخیص (0-100)")
        yolo_layout.addRow("آستانه اطمینان (%):", self.yolo_conf)

        group_yolo.setLayout(yolo_layout)
        layout.addWidget(group_yolo)

        # --- ضبط زنده ---
        group2 = QGroupBox("تنظیمات ضبط زنده (وبکم / IP)")
        form2 = QFormLayout()
        form2.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form2.setSpacing(12)

        interval_layout = QHBoxLayout()
        self.interval_min = QSpinBox(); self.interval_min.setRange(0, 60); self.interval_min.setValue(0)
        self.interval_sec = QSpinBox(); self.interval_sec.setRange(1, 59); self.interval_sec.setValue(5)
        interval_layout.addWidget(QLabel("دقیقه")); interval_layout.addWidget(self.interval_min)
        interval_layout.addWidget(QLabel("ثانیه")); interval_layout.addWidget(self.interval_sec)
        form2.addRow("فاصله ضبط:", interval_layout)

        self.buffer_size = QSpinBox(); self.buffer_size.setRange(100, 20000); self.buffer_size.setValue(2000)
        form2.addRow("حداکثر بافر:", self.buffer_size)

        group2.setLayout(form2)
        layout.addWidget(group2)

        # --- انتخاب فریم ---
        group3 = QGroupBox("حالت انتخاب فریم")
        form3 = QVBoxLayout()
        self.random_mode = QComboBox()
        modes = [
            ("رندوم کامل", "فریم‌ها کاملاً تصادفی انتخاب می‌شوند"),
            ("هر n فریم", "فریم‌ها با فاصله منظم انتخاب می‌شوند"),
            ("رندوم با فاصله", "فریم‌ها تصادفی اما مرتب انتخاب می‌شوند"),
            ("هوشمند با YOLO", "دسته‌بندی هوشمند فریم‌ها با YOLO و ساخت تایم‌لپس پویا")
        ]
        for text, tip in modes:
            self.random_mode.addItem(text)
        self.random_mode.currentIndexChanged.connect(lambda idx: self.random_mode.setToolTip(modes[idx][1]))
        self.random_mode.setToolTip(modes[0][1])
        form3.addWidget(self.random_mode)
        group3.setLayout(form3)
        layout.addWidget(group3)
        layout.addStretch()

    def setup_preview_tab(self):
        layout = QVBoxLayout(self.preview_tab)
        layout.setSpacing(15)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(300)
        self.player.setVideoOutput(self.video_widget)
        layout.addWidget(self.video_widget)

        self.preview_label = QLabel("پیش‌نمایش فریم استاتیک")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(100)
        self.preview_label.setStyleSheet("border: 2px dashed #278ea5; border-radius: 10px; background: #1b263b; color: #a9d6e0; font-style: italic;")
        layout.addWidget(self.preview_label)

        self.generate_progress = QProgressBar()
        self.generate_progress.setTextVisible(True)
        self.generate_progress.setFormat("در حال ساخت... %p%")
        layout.addWidget(self.generate_progress)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_player_position)
        layout.addWidget(self.slider)

        btn_layout = QHBoxLayout()
        play_btn = QPushButton("پخش")
        play_btn.clicked.connect(self.play_video)
        play_btn.setStyleSheet("background: #21a0b8; padding: 12px; font-size: 14px;")
        btn_layout.addWidget(play_btn)

        pause_btn = QPushButton("مکث")
        pause_btn.clicked.connect(self.pause_video)
        pause_btn.setStyleSheet("background: #1f5f8b; padding: 12px; font-size: 14px;")
        btn_layout.addWidget(pause_btn)

        stop_btn = QPushButton("توقف")
        stop_btn.clicked.connect(self.stop_video)
        stop_btn.setStyleSheet("background: #c94f4f; padding: 12px; font-size: 14px;")
        btn_layout.addWidget(stop_btn)

        save_btn = QPushButton("ذخیره ویدیو")
        save_btn.clicked.connect(self.save_output)
        save_btn.setStyleSheet("background: #1f5f8b; padding: 12px; font-size: 14px;")
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.append("سیستم آماده است.")
        layout.addWidget(self.log_text)

    def play_video(self):
        if self.player.mediaStatus() == QMediaPlayer.MediaStatus.BufferedMedia:
            self.player.play()

    def pause_video(self):
        self.player.pause()

    def stop_video(self):
        self.player.stop()

    def set_player_position(self, position):
        self.player.setPosition(position)

    def update_slider_position(self, position):
        self.slider.setValue(position)

    def update_slider_range(self, duration):
        self.slider.setRange(0, duration)

    def browse_file(self):
        file = QFileDialog.getOpenFileName(self, "انتخاب فایل ویدیو", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)")[0]
        if file:
            self.source_input.setText(file)

    def get_interval_seconds(self):
        return self.interval_min.value() * 60 + self.interval_sec.value()

    def load_source(self):
        self.source_path = self.source_input.text().strip()
        self.source_type = self.source_type_combo.currentText()
        self.frames = []
        self.total_frames = 0
        self.status_label.setText("در حال بارگذاری...")
        self.log_text.append("بارگذاری شروع شد...")
        self.progress.setValue(0)
        self.load_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        if self.source_type == "فایل ویدیو":
            if not self.source_path or not os.path.exists(self.source_path):
                self.reset_load_buttons()
                QMessageBox.warning(self, "خطا", "فایل یافت نشد!")
                return
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                self.reset_load_buttons()
                QMessageBox.warning(self, "خطا", "فایل ویدیو قابل خواندن نیست.")
                return
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress.setMaximum(100)
            self.progress.setValue(50)
            ret, frame = cap.read()
            if ret:
                self.update_preview(frame)
            cap.release()
            self.progress.setValue(100)
            self.status_label.setText(f"بارگذاری شد: {self.total_frames} فریم")
            self.log_text.append(f"موفقیت: {self.total_frames} فریم موجود.")
            self.reset_load_buttons()
        else:
            src = 0 if self.source_type == "وبکم سیستم" else self.source_path
            if self.capture_thread and self.capture_thread.isRunning():
                self.capture_thread.running = False
                self.capture_thread.wait()
            self.capture_thread = FrameCaptureThread(src, self.get_interval_seconds(), self.buffer_size.value())
            self.capture_thread.frameCaptured.connect(self.update_preview)
            self.capture_thread.errorOccurred.connect(self.show_error)
            self.capture_thread.progressUpdated.connect(lambda x: self.progress.setValue(min(x, self.buffer_size.value())))
            self.progress.setMaximum(self.buffer_size.value())
            self.capture_thread.finished.connect(self.on_capture_finished)
            self.capture_thread.start()
            self.status_label.setText("ضبط زنده شروع شد...")
            self.log_text.append("ضبط از منبع زنده فعال شد.")

    def on_capture_finished(self):
        self.frames = self.capture_thread.frames
        self.status_label.setText(f"ضبط متوقف شد: {len(self.frames)} فریم")
        self.reset_load_buttons()

    def stop_loading(self):
        if self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.running = False
            self.capture_thread.wait()
        self.reset_load_buttons()
        self.status_label.setText("بارگذاری متوقف شد")
        self.log_text.append("بارگذاری متوقف شد.")

    def reset_load_buttons(self):
        self.load_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def start_extract_frames(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)
        self.extractor_thread = FrameExtractorThread(self.source_path, self.temp_dir)
        self.extractor_thread.progressUpdated.connect(self.generate_progress.setValue)
        self.extractor_thread.finished.connect(self.on_frames_extracted)
        self.extractor_thread.errorOccurred.connect(self.show_error)
        self.extractor_thread.start()
        self.log_text.append("استخراج فریم‌ها شروع شد...")

    def on_frames_extracted(self, frame_paths):
        self.frame_paths = frame_paths
        self.log_text.append(f"استخراج شد: {len(frame_paths)} فریم")
        self.start_yolo_classification()

    def start_yolo_classification(self):
        model_index = self.yolo_model.currentIndex()
        model_paths = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        model_path = model_paths[model_index]
        self.yolo_thread = YoloClassificationThread(self.frame_paths, model_path)
        self.yolo_thread.progressUpdated.connect(self.generate_progress.setValue)
        self.yolo_thread.finished.connect(self.on_yolo_finished)
        self.yolo_thread.errorOccurred.connect(self.show_error)
        self.yolo_thread.start()
        self.log_text.append("دسته‌بندی با YOLO شروع شد...")

    def on_yolo_finished(self, categories):
        self.categories = categories
        self.log_text.append(f"دسته‌بندی شد: {len(categories)} گروه")
        self.build_yolo_timelapse()

    def build_yolo_timelapse(self):
        selected_classes = self.yolo_classes.text().strip()
        if selected_classes:
            selected_classes = [c.strip() for c in selected_classes.split(',')]
        else:
            selected_classes = list(self.categories.keys())
        all_paths = []
        for cls in selected_classes:
            if cls in self.categories:
                all_paths.extend(self.categories[cls])
                # Add transition if needed (creative: black frame)
                if len(all_paths) > 0:
                    black = cv2.imread(all_paths[-1]) * 0
                    cv2.imwrite(os.path.join(self.temp_dir, f"transition_{len(all_paths)}.jpg"), black)
                    all_paths.append(os.path.join(self.temp_dir, f"transition_{len(all_paths)}.jpg"))

        if not all_paths:
            self.show_error("هیچ فریمی در کلاس‌های انتخابی یافت نشد")
            return

        # Select indices based on mode
        mode = self.random_mode.currentText()
        n = min(self.frame_count.value(), len(all_paths))
        if mode == "رندوم کامل":
            indices = sorted(random.sample(range(len(all_paths)), n))
        elif mode == "هر n فریم":
            step = max(1, len(all_paths) // n)
            indices = list(range(0, len(all_paths), step))[:n]
        else:
            indices = sorted(random.sample(range(len(all_paths)), n))

        self.generator_thread = TimelapseGeneratorThread(
            None, indices, self.fps.value(), self.temp_video_path, self.is_cuda, None,
            self.hold_mode.currentText(), self.min_hold.value(), self.max_hold.value(), all_paths
        )
        self.generator_thread.progressUpdated.connect(self.generate_progress.setValue)
        self.generator_thread.finished.connect(self.on_generate_finished)
        self.generator_thread.errorOccurred.connect(self.show_error)
        self.generator_thread.start()
        self.log_text.append("ساخت تایم‌لپس YOLO شروع شد...")

    def start_generate_timelapse(self):
        if self.source_type == "فایل ویدیو":
            if self.total_frames == 0:
                QMessageBox.warning(self, "خطا", "ابتدا منبع فایل را بارگذاری کنید.")
                return
            total = self.total_frames
        else:
            if not self.frames:
                QMessageBox.warning(self, "خطا", "ابتدا منبع زنده را بارگذاری کنید.")
                return
            total = len(self.frames)

        mode = self.random_mode.currentText()
        if mode == "هوشمند با YOLO":
            self.start_extract_frames()
            return

        n = self.frame_count.value()
        if n > total:
            n = total
            QMessageBox.information(self, "توجه", f"از {n} فریم موجود استفاده می‌شود.")

        if mode == "رندوم کامل":
            indices = random.sample(range(total), n)
            indices.sort()
        elif mode == "هر n فریم":
            step = max(1, total // n)
            indices = list(range(0, total, step))[:n]
        else:
            indices = sorted(random.sample(range(total), n))

        self.generate_progress.setMaximum(100)
        self.generate_progress.setValue(0)
        self.generate_btn.setEnabled(False)
        self.stop_generate_btn.setEnabled(True)
        self.log_text.append("ساخت تایم‌لپس شروع شد...")

        source = self.source_path if self.source_type == "فایل ویدیو" else None
        frames = self.frames if source is None else None

        self.generator_thread = TimelapseGeneratorThread(
            source, indices, self.fps.value(), self.temp_video_path, self.is_cuda, frames,
            self.hold_mode.currentText(), self.min_hold.value(), self.max_hold.value()
        )
        self.generator_thread.progressUpdated.connect(self.generate_progress.setValue)
        self.generator_thread.finished.connect(self.on_generate_finished)
        self.generator_thread.errorOccurred.connect(self.show_error)
        self.generator_thread.start()

    def on_generate_finished(self, success):
        self.generate_btn.setEnabled(True)
        self.stop_generate_btn.setEnabled(False)
        if success:
            self.log_text.append("تایم‌لپس با موفقیت ساخته شد!")
            self.preview_video(self.temp_video_path)
            self.player.setSource(QUrl.fromLocalFile(self.temp_video_path))
        else:
            self.log_text.append("ساخت ناموفق بود.")

    def stop_generate(self):
        if self.generator_thread and self.generator_thread.isRunning():
            self.generator_thread.running = False
            self.generator_thread.wait()
        self.reset_generate_buttons()
        self.log_text.append("ساخت متوقف شد.")

    def reset_generate_buttons(self):
        self.generate_btn.setEnabled(True)
        self.stop_generate_btn.setEnabled(False)

    def update_preview(self, frame):
        if frame is None: return
        h, w, _ = frame.shape
        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg).scaled(500, 350, Qt.AspectRatioMode.KeepAspectRatio)
        self.preview_label.setPixmap(pixmap)

    def show_error(self, msg):
        self.status_label.setText("خطا")
        QMessageBox.critical(self, "خطا", msg)
        self.log_text.append(f"خطا: {msg}")
        self.reset_load_buttons()
        self.reset_generate_buttons()

    def preview_video(self, path):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.update_preview(frame)

    def save_output(self):
        if not os.path.exists(self.temp_video_path):
            QMessageBox.warning(self, "خطا", "ابتدا تایم‌لپس را بسازید.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "ذخیره تایم‌لپس", "timelapse_output.mp4", "MP4 Video (*.mp4)")
        if path:
            import shutil
            shutil.copy(self.temp_video_path, path)
            self.log_text.append(f"ذخیره شد: {path}")
            QMessageBox.information(self, "موفقیت", "ویدیو ذخیره شد.")

    def closeEvent(self, event):
        self.player.stop()
        self.stop_loading()
        self.stop_generate()
        if os.path.exists(self.temp_video_path):
            try: os.remove(self.temp_video_path)
            except: pass
        if os.path.exists(self.temp_dir):
            try: shutil.rmtree(self.temp_dir)
            except: pass
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 11))
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    window = TimelapseApp()
    window.show()
    sys.exit(app.exec())