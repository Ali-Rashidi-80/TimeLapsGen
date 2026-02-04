import sys
import random
import time
import cv2
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QComboBox, QProgressBar,
    QFileDialog, QMessageBox, QTextEdit, QGroupBox, QFormLayout, QToolTip
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPalette, QColor, QFont, QIcon

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
        while self.running and len(self.frames) < self.buffer_size:
            ret, frame = cap.read()
            if ret:
                self.frames.append(frame)
                self.frameCaptured.emit(frame)
            else:
                break
            time.sleep(self.interval)
        cap.release()

class TimelapseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("تایم‌لپس ساز حرفه‌ای")
        self.setGeometry(100, 100, 900, 650)
        self.setWindowIcon(QIcon.fromTheme("video-x-generic"))
        self.frames = []
        self.capture_thread = None
        self.temp_video_path = 'temp_timelapse.mp4'
        self.is_cuda = is_cuda_available()

        self.setup_ui()
        self.apply_dark_blue_theme()

        if self.is_cuda:
            self.log_text.append("CUDA فعال است – پردازش سریع‌تر!")

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

        self.source_type = QComboBox()
        self.source_type.addItems(["فایل ویدیو", "وبکم سیستم", "آدرس IP دوربین"])
        self.source_type.setToolTip("نوع منبعی که می‌خواهید از آن تایم‌لپس بسازید")
        group_layout.addRow("نوع منبع:", self.source_type)

        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("مثلاً: C:\\video.mp4 یا rtsp://...")
        self.source_input.setToolTip("مسیر فایل یا آدرس استریم را وارد کنید")
        group_layout.addRow("مسیر / آدرس:", self.source_input)

        browse_btn = QPushButton("انتخاب فایل")
        browse_btn.setToolTip("فایل ویدیویی را از سیستم انتخاب کنید")
        browse_btn.clicked.connect(self.browse_file)
        hbox = QHBoxLayout()
        hbox.addWidget(self.source_input)
        hbox.addWidget(browse_btn)
        group_layout.addRow(hbox)

        group.setLayout(group_layout)
        layout.addWidget(group)

        load_btn = QPushButton("بارگذاری منبع")
        load_btn.setToolTip("شروع بارگذاری یا ضبط از منبع")
        load_btn.clicked.connect(self.load_source)
        load_btn.setStyleSheet("background: #21a0b8; padding: 12px; font-size: 14px;")
        layout.addWidget(load_btn)

        self.status_label = QLabel("آماده")
        self.status_label.setStyleSheet("color: #a9d6e0; font-style: italic;")
        layout.addWidget(self.status_label)
        layout.addStretch()

    def setup_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)
        layout.setSpacing(15)

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

        group2 = QGroupBox("تنظیمات ضبط زنده (وبکم / IP)")
        form2 = QFormLayout()
        form2.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form2.setSpacing(12)

        interval_layout = QHBoxLayout()
        self.interval_min = QSpinBox()
        self.interval_min.setRange(0, 60)
        self.interval_min.setValue(0)
        self.interval_min.setToolTip("دقیقه")
        self.interval_sec = QSpinBox()
        self.interval_sec.setRange(1, 59)
        self.interval_sec.setValue(5)
        self.interval_sec.setToolTip("ثانیه")
        interval_layout.addWidget(self.interval_min)
        interval_layout.addWidget(QLabel("دقیقه"))
        interval_layout.addWidget(self.interval_sec)
        interval_layout.addWidget(QLabel("ثانیه"))
        form2.addRow("فاصله ضبط:", interval_layout)

        self.buffer_size = QSpinBox()
        self.buffer_size.setRange(100, 20000)
        self.buffer_size.setValue(2000)
        self.buffer_size.setToolTip("حداکثر فریم ذخیره شده در حافظه")
        form2.addRow("حداکثر بافر:", self.buffer_size)

        group2.setLayout(form2)
        layout.addWidget(group2)

        group3 = QGroupBox("حالت انتخاب فریم")
        form3 = QVBoxLayout()
        self.random_mode = QComboBox()
        modes = [
            ("رندوم کامل", "فریم‌ها کاملاً تصادفی انتخاب می‌شوند"),
            ("هر n فریم", "فریم‌ها با فاصله منظم انتخاب می‌شوند"),
            ("رندوم با فاصله", "فریم‌ها تصادفی اما مرتب انتخاب می‌شوند")
        ]
        for text, tip in modes:
            self.random_mode.addItem(text)
            # ToolTip on item level not supported, so use currentIndexChanged
        self.random_mode.setToolTip(modes[0][1])
        self.random_mode.currentIndexChanged.connect(
            lambda idx: self.random_mode.setToolTip(modes[idx][1])
        )
        form3.addWidget(self.random_mode)
        group3.setLayout(form3)
        layout.addWidget(group3)
        layout.addStretch()

    def setup_preview_tab(self):
        layout = QVBoxLayout(self.preview_tab)
        layout.setSpacing(15)

        self.preview_label = QLabel("پیش‌نمایش فریم")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(250)
        self.preview_label.setStyleSheet("""
            border: 2px dashed #278ea5; border-radius: 10px; 
            background: #1b263b; color: #a9d6e0; font-style: italic;
        """)
        self.preview_label.setToolTip("فریم فعلی یا اولین فریم ویدیو ساخته شده")
        layout.addWidget(self.preview_label)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("در حال پردازش... %p%")
        layout.addWidget(self.progress)

        btn_layout = QHBoxLayout()
        generate_btn = QPushButton("ساخت تایم‌لپس")
        generate_btn.setToolTip("شروع ساخت ویدیو با تنظیمات فعلی")
        generate_btn.clicked.connect(self.generate_timelapse)
        generate_btn.setStyleSheet("background: #21a0b8; padding: 12px; font-size: 14px;")
        btn_layout.addWidget(generate_btn)

        save_btn = QPushButton("ذخیره ویدیو")
        save_btn.setToolTip("ویدیو ساخته شده را ذخیره کنید")
        save_btn.clicked.connect(self.save_output)
        save_btn.setStyleSheet("background: #1f5f8b; padding: 12px; font-size: 14px;")
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.append("سیستم آماده است.")
        layout.addWidget(self.log_text)

    def browse_file(self):
        file = QFileDialog.getOpenFileName(
            self, "انتخاب فایل ویدیو", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )[0]
        if file:
            self.source_input.setText(file)

    def get_interval_seconds(self):
        return self.interval_min.value() * 60 + self.interval_sec.value()

    def load_source(self):
        source = self.source_input.text().strip()
        source_type = self.source_type.currentText()
        self.frames = []
        self.status_label.setText("در حال بارگذاری...")
        self.log_text.append("بارگذاری شروع شد...")

        if source_type == "فایل ویدیو":
            if not source or not os.path.exists(source):
                QMessageBox.warning(self, "خطا", "فایل یافت نشد!")
                return
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                QMessageBox.warning(self, "خطا", "فایل ویدیو قابل خواندن نیست.")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total > 150000:
                QMessageBox.warning(self, "هشدار", "ویدیو بسیار طولانی است!")
            self.progress.setMaximum(total)
            self.extract_frames_from_file(source)
        else:
            src = 0 if source_type == "وبکم سیستم" else source
            if self.capture_thread and self.capture_thread.isRunning():
                self.capture_thread.running = False
                self.capture_thread.wait()
            self.capture_thread = FrameCaptureThread(
                src, self.get_interval_seconds(), self.buffer_size.value()
            )
            self.capture_thread.frameCaptured.connect(self.update_preview)
            self.capture_thread.errorOccurred.connect(self.show_error)
            self.capture_thread.start()
            self.status_label.setText("ضبط زنده شروع شد...")
            self.log_text.append("ضبط از منبع زنده فعال شد.")

    def extract_frames_from_file(self, path):
        if self.is_cuda:
            try:
                import cv2.cudacodec as cudacodec
                reader = cudacodec.createVideoReader(path)
                self.log_text.append("CUDA: خواندن سریع ویدیو")
                i = 0
                while True:
                    gpu_frame = reader.nextFrame()
                    if gpu_frame.empty(): break
                    frame = gpu_frame.download()
                    self.frames.append(frame)
                    i += 1
                    if i % 50 == 0:
                        self.progress.setValue(i)
                        QApplication.processEvents()
            except Exception as e:
                self.log_text.append(f"CUDA شکست خورد: {e}. استفاده از CPU.")
                self.is_cuda = False
        if not self.is_cuda:
            cap = cv2.VideoCapture(path)
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                self.frames.append(frame)
                i += 1
                if i % 50 == 0:
                    self.progress.setValue(i)
                    QApplication.processEvents()
            cap.release()
        self.status_label.setText(f"بارگذاری شد: {len(self.frames)} فریم")
        self.log_text.append(f"موفقیت: {len(self.frames)} فریم استخراج شد.")
        if self.frames:
            self.update_preview(self.frames[0])

    def update_preview(self, frame):
        if frame is None: return
        h, w, _ = frame.shape
        qimg = QImage(frame.data, w, h, 3*w, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg).scaled(500, 350, Qt.AspectRatioMode.KeepAspectRatio)
        self.preview_label.setPixmap(pixmap)
        if hasattr(self, 'capture_thread'):
            self.frames = self.capture_thread.frames

    def show_error(self, msg):
        self.status_label.setText("خطا")
        QMessageBox.critical(self, "خطا", msg)
        self.log_text.append(f"خطا: {msg}")

    def generate_timelapse(self):
        if not self.frames:
            QMessageBox.warning(self, "خطا", "ابتدا منبع را بارگذاری کنید.")
            return
        mode = self.random_mode.currentText()
        n = self.frame_count.value()
        if n > len(self.frames):
            n = len(self.frames)
            QMessageBox.information(self, "توجه", f"از {n} فریم موجود استفاده می‌شود.")

        if mode == "رندوم کامل":
            selected = random.sample(self.frames, n)
        elif mode == "هر n فریم":
            step = max(1, len(self.frames) // n)
            selected = self.frames[::step][:n]
        else:  # رندوم با فاصله
            idx = sorted(random.sample(range(len(self.frames)), n))
            selected = [self.frames[i] for i in idx]

        self.progress.setMaximum(n)
        h, w = selected[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.temp_video_path, fourcc, self.fps.value(), (w, h))
        for i, f in enumerate(selected):
            out.write(f)
            self.progress.setValue(i + 1)
            QApplication.processEvents()
        out.release()
        self.log_text.append("تایم‌لپس با موفقیت ساخته شد!")
        self.preview_video(self.temp_video_path)

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
        path, _ = QFileDialog.getSaveFileName(
            self, "ذخیره تایم‌لپس", "timelapse_output.mp4", "MP4 Video (*.mp4)"
        )
        if path:
            import shutil
            shutil.copy(self.temp_video_path, path)
            self.log_text.append(f"ذخیره شد: {path}")
            QMessageBox.information(self, "موفقیت", "ویدیو ذخیره شد!")

    def closeEvent(self, event):
        if self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.running = False
            self.capture_thread.wait()
        if os.path.exists(self.temp_video_path):
            try: os.remove(self.temp_video_path)
            except: pass
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 11))
    window = TimelapseApp()
    window.show()
    sys.exit(app.exec())