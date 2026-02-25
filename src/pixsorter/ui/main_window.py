from pathlib import Path
import time

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QPlainTextEdit, QCheckBox, QGroupBox, QFrame,
    QScrollArea
)

from ..config.defaults import SortConfig
from ..pipeline.worker import SortWorker
from ..config.schemas import validate_config
from ..infra.resources import resource_path
from .widgets import DropFrame, help_button


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixSorter")
        self.setMinimumSize(1100, 720)

        ico = resource_path("assets/app.ico")
        if ico:
            self.setWindowIcon(QIcon(ico))

        self.worker_thread: QThread | None = None
        self.worker: SortWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # Left (scrollable)
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setSpacing(12)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left_widget)
        root.addWidget(left_scroll, 3)

        title = QLabel("PixSorter")
        title.setObjectName("Title")
        left.addWidget(title)

        subtitle = QLabel("Group duplicates by frame/pose • Pick TOP-K best shots")
        subtitle.setObjectName("Subtitle")
        left.addWidget(subtitle)

        drop = DropFrame()
        drop_layout = QVBoxLayout(drop)
        drop_layout.setContentsMargins(12, 12, 12, 12)
        drop_label = QLabel("Drag & drop a folder of photos here")
        drop_label.setObjectName("DropLabel")
        drop_label.setAlignment(Qt.AlignCenter)
        drop_layout.addWidget(drop_label)
        drop.dropped.connect(self.set_input_dir)
        left.addWidget(drop)

        # Folders
        io_box = QGroupBox("Folders")
        io_layout = QGridLayout(io_box)
        io_layout.setColumnStretch(1, 1)

        self.in_edit = QLineEdit()
        self.in_edit.setPlaceholderText("Select input folder…")
        btn_in = QPushButton("Browse…")
        btn_in.clicked.connect(self.pick_input)

        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Select output folder…")
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self.pick_output)

        self.chk_new_out = QCheckBox("Create new output folder inside selected output")
        self.new_out_name = QLineEdit("SortedOutput")
        self.new_out_name.setEnabled(False)
        self.chk_new_out.toggled.connect(self.new_out_name.setEnabled)

        io_layout.addWidget(QLabel("Input folder"), 0, 0)
        io_layout.addWidget(self.in_edit, 0, 1)
        io_layout.addWidget(btn_in, 0, 2)

        io_layout.addWidget(QLabel("Output folder"), 1, 0)
        io_layout.addWidget(self.out_edit, 1, 1)
        io_layout.addWidget(btn_out, 1, 2)

        io_layout.addWidget(self.chk_new_out, 2, 0, 1, 3)
        io_layout.addWidget(QLabel("New folder name"), 3, 0)
        io_layout.addWidget(self.new_out_name, 3, 1, 1, 2)

        left.addWidget(io_box)

        # Parameters
        params = QGroupBox("Parameters")
        grid = QGridLayout(params)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(1, 260)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        r = 0

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["copy", "move"])
        grid.addWidget(QLabel("Mode"), r, 0)
        grid.addWidget(self.mode_combo, r, 1)
        grid.addWidget(help_button(self, "Mode", "Copy keeps originals safe. Move relocates files.", rec="copy"), r, 2)
        r += 1

        self.min_group_spin = QSpinBox()
        self.min_group_spin.setRange(2, 200)
        grid.addWidget(QLabel("Min group size"), r, 0)
        grid.addWidget(self.min_group_spin, r, 1)
        grid.addWidget(help_button(self, "Min group size", "Minimum photos to create a group folder.", minv=2, maxv=200, rec=2), r, 2)
        r += 1

        self.clip_model_combo = QComboBox()
        self.clip_model_combo.addItems(["vit_l_14", "vit_b_32"])
        grid.addWidget(QLabel("CLIP model"), r, 0)
        grid.addWidget(self.clip_model_combo, r, 1)
        grid.addWidget(help_button(self, "CLIP model", "Used to find candidate neighbors. L-14 more accurate, B-32 faster.", rec="vit_l_14"), r, 2)
        r += 1

        self.knn_spin = QSpinBox()
        self.knn_spin.setRange(5, 60)
        grid.addWidget(QLabel("Neighbors (K)"), r, 0)
        grid.addWidget(self.knn_spin, r, 1)
        grid.addWidget(help_button(self, "Neighbors (K)", "How many nearest neighbors to verify per image.", minv=5, maxv=60, rec=18), r, 2)
        r += 1

        self.clip_sim = QDoubleSpinBox()
        self.clip_sim.setRange(0.70, 0.99)
        self.clip_sim.setSingleStep(0.01)
        grid.addWidget(QLabel("Candidate similarity"), r, 0)
        grid.addWidget(self.clip_sim, r, 1)
        grid.addWidget(help_button(self, "Candidate similarity", "CLIP cosine sim threshold before ORB verification.", minv=0.70, maxv=0.99, rec=0.86), r, 2)
        r += 1

        self.orb_feat_spin = QSpinBox()
        self.orb_feat_spin.setRange(800, 6000)
        self.orb_feat_spin.setSingleStep(100)
        grid.addWidget(QLabel("ORB features"), r, 0)
        grid.addWidget(self.orb_feat_spin, r, 1)
        grid.addWidget(help_button(self, "ORB features", "More features improves robustness but slows down.", minv=800, maxv=6000, rec=2500), r, 2)
        r += 1

        self.orb_ratio = QDoubleSpinBox()
        self.orb_ratio.setRange(0.55, 0.95)
        self.orb_ratio.setSingleStep(0.01)
        grid.addWidget(QLabel("Match strictness"), r, 0)
        grid.addWidget(self.orb_ratio, r, 1)
        grid.addWidget(help_button(self, "Match strictness", "Lowe ratio test. Lower=stricter.", minv=0.55, maxv=0.95, rec=0.75), r, 2)
        r += 1

        self.min_inliers_spin = QSpinBox()
        self.min_inliers_spin.setRange(10, 80)
        grid.addWidget(QLabel("Min inliers"), r, 0)
        grid.addWidget(self.min_inliers_spin, r, 1)
        grid.addWidget(help_button(self, "Min inliers", "Homography inliers required to confirm match.", minv=10, maxv=80, rec=22), r, 2)
        r += 1

        self.verify_size_spin = QSpinBox()
        self.verify_size_spin.setRange(500, 1400)
        grid.addWidget(QLabel("Verify resize"), r, 0)
        grid.addWidget(self.verify_size_spin, r, 1)
        grid.addWidget(help_button(self, "Verify resize", "Max side used during ORB verification.", minv=500, maxv=1400, rec=900), r, 2)
        r += 1

        self.best_chk = QCheckBox("Create BEST_OF picks")
        grid.addWidget(self.best_chk, r, 0, 1, 2)
        grid.addWidget(help_button(self, "Create BEST_OF picks", "Export TOP-K picks into BEST_OF folder.", rec="Enabled"), r, 2)
        r += 1

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 20)
        grid.addWidget(QLabel("TOP-K per group"), r, 0)
        grid.addWidget(self.topk_spin, r, 1)
        grid.addWidget(help_button(self, "TOP-K per group", "How many best images to export per group.", minv=1, maxv=20, rec=3), r, 2)
        r += 1

        self.diverse_chk = QCheckBox("Enforce diversity within TOP-K")
        grid.addWidget(self.diverse_chk, r, 0, 1, 2)
        grid.addWidget(help_button(self, "Diversity", "Avoid selecting near-identical burst variants.", rec="Enabled"), r, 2)
        r += 1

        self.diverse_inliers_spin = QSpinBox()
        self.diverse_inliers_spin.setRange(15, 80)
        grid.addWidget(QLabel("Diversity threshold"), r, 0)
        grid.addWidget(self.diverse_inliers_spin, r, 1)
        grid.addWidget(help_button(self, "Diversity threshold", "Inliers threshold for skipping similar candidates.", minv=15, maxv=80, rec=40), r, 2)
        r += 1

        self.eyes_chk = QCheckBox("Prefer eyes open (faces)")
        grid.addWidget(self.eyes_chk, r, 0, 1, 2)
        grid.addWidget(help_button(self, "Eyes open", "Uses MediaPipe to penalize closed eyes.", rec="Enabled"), r, 2)
        r += 1

        self.eyes_weight = QDoubleSpinBox()
        self.eyes_weight.setRange(1.0, 5.0)
        self.eyes_weight.setSingleStep(0.1)
        grid.addWidget(QLabel("Eyes penalty"), r, 0)
        grid.addWidget(self.eyes_weight, r, 1)
        grid.addWidget(help_button(self, "Eyes penalty", "Higher penalizes closed eyes more strongly.", minv=1.0, maxv=5.0, rec=2.2), r, 2)
        r += 1

        left.addWidget(params)

        run_row = QHBoxLayout()
        self.reco_btn = QPushButton("Apply recommended")
        self.reco_btn.clicked.connect(self.apply_recommended)

        self.run_btn = QPushButton("Start")
        self.run_btn.clicked.connect(self.start)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)

        run_row.addWidget(self.reco_btn)
        run_row.addStretch(1)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.cancel_btn)
        left.addLayout(run_row)

        left.addStretch(1)

        # Right panel
        right = QVBoxLayout()
        right.setSpacing(10)
        root.addLayout(right, 2)

        self.stage_label = QLabel("Idle")
        self.stage_label.setObjectName("Stage")
        right.addWidget(self.stage_label)
        
        self.eta_label = QLabel("ETA: --")
        self.eta_label.setObjectName("Eta")
        right.addWidget(self.eta_label)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        right.addWidget(self.pbar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("Log")
        right.addWidget(self.log_view, 1)

        self._setup_menu()
        self.apply_recommended()

    def on_progress(self, p: int):
        p = max(0, min(int(p), 100))
        self.pbar.setValue(p)
    
        now = time.time()
        if p <= 0:
            self._eta_last_t = now
            self._eta_last_p = p
            self._eta_rate_ema = None
            self.eta_label.setText("ETA: --")
            return
    
        if self._eta_last_t is None or self._eta_last_p is None:
            self._eta_last_t = now
            self._eta_last_p = p
            return
    
        dt = now - self._eta_last_t
        dp = p - self._eta_last_p
    
        # ignore tiny/noisy updates
        if dt < 0.35 or dp <= 0:
            return
    
        rate = dp / dt  # % per second
        alpha = 0.25    # EMA smoothing
        self._eta_rate_ema = rate if self._eta_rate_ema is None else (alpha * rate + (1 - alpha) * self._eta_rate_ema)
    
        self._eta_last_t = now
        self._eta_last_p = p
    
        if p >= 100:
            self.eta_label.setText("ETA: 0s")
            return
    
        if not self._eta_rate_ema or self._eta_rate_ema <= 1e-6:
            self.eta_label.setText("ETA: --")
            return
    
        remaining_s = (100 - p) / self._eta_rate_ema
        self.eta_label.setText(f"ETA: {self._fmt_seconds(remaining_s)}")
    
    @staticmethod
    def _fmt_seconds(secs: float) -> str:
        secs = max(0, int(secs + 0.5))
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def _setup_menu(self):
        menubar = self.menuBar()
        filem = menubar.addMenu("File")
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        filem.addAction(act_quit)

    def append_log(self, text: str):
        self.log_view.appendPlainText(text)

    def set_input_dir(self, path: str):
        self.in_edit.setText(path)

    def pick_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if d:
            self.in_edit.setText(d)

    def pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self.out_edit.setText(d)

    def apply_recommended(self):
        d = SortConfig()
        self.mode_combo.setCurrentText(d.mode)
        self.min_group_spin.setValue(d.min_group)

        self.clip_model_combo.setCurrentText(d.clip_model)
        self.knn_spin.setValue(d.knn_k)
        self.clip_sim.setValue(d.clip_sim_thresh)

        self.orb_feat_spin.setValue(d.orb_nfeatures)
        self.orb_ratio.setValue(d.orb_match_thresh)
        self.min_inliers_spin.setValue(d.min_inliers)
        self.verify_size_spin.setValue(d.max_side_verify)

        self.best_chk.setChecked(d.best_of)
        self.topk_spin.setValue(d.top_k)
        self.diverse_chk.setChecked(d.diverse)
        self.diverse_inliers_spin.setValue(d.diverse_inliers)
        self.eyes_chk.setChecked(d.use_eyes)
        self.eyes_weight.setValue(d.eyes_weight)

        self.append_log("[UI] Applied recommended settings.")

    def collect_cfg(self) -> SortConfig:
        c = SortConfig()
        c.input_dir = self.in_edit.text().strip()
        c.output_dir = self.out_edit.text().strip()
        c.output_create_new = self.chk_new_out.isChecked()
        c.output_new_name = self.new_out_name.text().strip() or "SortedOutput"

        c.mode = self.mode_combo.currentText()
        c.min_group = int(self.min_group_spin.value())

        c.clip_model = self.clip_model_combo.currentText()
        c.knn_k = int(self.knn_spin.value())
        c.clip_sim_thresh = float(self.clip_sim.value())

        c.orb_nfeatures = int(self.orb_feat_spin.value())
        c.orb_match_thresh = float(self.orb_ratio.value())
        c.min_inliers = int(self.min_inliers_spin.value())
        c.max_side_verify = int(self.verify_size_spin.value())

        c.best_of = self.best_chk.isChecked()
        c.top_k = int(self.topk_spin.value())
        c.diverse = self.diverse_chk.isChecked()
        c.diverse_inliers = int(self.diverse_inliers_spin.value())

        c.use_eyes = self.eyes_chk.isChecked()
        c.eyes_weight = float(self.eyes_weight.value())
        return c

    def validate_cfg(self, cfg: SortConfig) -> bool:
        errs = validate_config(cfg)
        if errs:
            QMessageBox.warning(self, "Invalid configuration", "• " + "\n• ".join(errs))
            return False
        return True

    def start(self):
        cfg = self.collect_cfg()
        if not self.validate_cfg(cfg):
            return

        self.log_view.clear()
        self.pbar.setValue(0)
        self.stage_label.setText("Starting…")
        
        self._eta_last_t = None
        self._eta_last_p = None
        self._eta_rate_ema = None   # % per second
        self.eta_label.setText("ETA: --")

        self.run_btn.setEnabled(False)
        self.reco_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.worker_thread = QThread()
        self.worker = SortWorker(cfg)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.stage.connect(self.stage_label.setText)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.worker.deleteLater)

        self.worker_thread.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.append_log("Cancelling…")
            self.cancel_btn.setEnabled(False)

    def on_finished(self, ok: bool, msg: str):
        self.run_btn.setEnabled(True)
        self.reco_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if ok:
            QMessageBox.information(self, "Done", msg)
        else:
            QMessageBox.critical(self, "Stopped", msg)

        self.stage_label.setText("Idle")
        self.worker = None
        self.worker_thread = None