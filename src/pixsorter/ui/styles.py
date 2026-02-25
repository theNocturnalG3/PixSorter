DARK_QSS = """
QMainWindow { background: #0f1115; }
QMenuBar { background: #0f1115; color: #d7dae0; }
QMenuBar::item { background: transparent; padding: 6px 10px; }
QMenuBar::item:selected { background: #1d2330; }
QMenu { background: #141823; color: #d7dae0; border: 1px solid #242b3a; }
QMenu::item:selected { background: #1d2330; }

QGroupBox {
  color: #d7dae0;
  border: 1px solid #242b3a;
  border-radius: 10px;
  margin-top: 10px;
  background: #111521;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 12px;
  padding: 0 6px;
  color: #d7dae0;
}

QLabel { color: #d7dae0; }
QLabel#Title { font-size: 26px; font-weight: 700; color: #f0f2f7; }
QLabel#Subtitle { color: #9aa3b2; margin-bottom: 6px; }
QLabel#Stage { font-size: 14px; color: #f0f2f7; padding: 6px 8px; background: #111521; border: 1px solid #242b3a; border-radius: 10px; }

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
  background: #0f1115;
  border: 1px solid #242b3a;
  border-radius: 8px;
  padding: 6px 8px;
  color: #d7dae0;
}
QComboBox::drop-down { border-left: 1px solid #242b3a; width: 24px; }
QComboBox QAbstractItemView {
  background: #111521; color: #d7dae0; selection-background-color: #1d2330;
  border: 1px solid #242b3a;
}

QPushButton {
  background: #1d2330;
  border: 1px solid #2b3550;
  border-radius: 10px;
  padding: 10px 14px;
  color: #f0f2f7;
  font-weight: 600;
}
QPushButton:hover { background: #232c3f; }
QPushButton:pressed { background: #1a2130; }
QPushButton:disabled { background: #151a24; border: 1px solid #20263a; color: #6b7280; }

QProgressBar {
  border: 1px solid #242b3a;
  border-radius: 10px;
  background: #111521;
  color: #d7dae0;
  text-align: center;
  height: 18px;
}
QProgressBar::chunk { background-color: #3b82f6; border-radius: 10px; }

QPlainTextEdit#Log {
  background: #0f1115;
  border: 1px solid #242b3a;
  border-radius: 10px;
  color: #d7dae0;
  padding: 8px;
  font-family: Consolas, Menlo, monospace;
  font-size: 12px;
}

QCheckBox { spacing: 8px; }
QCheckBox::indicator {
  width: 18px; height: 18px;
  border: 1px solid #2b3550;
  border-radius: 5px;
  background: #0f1115;
}
QCheckBox::indicator:checked { background: #3b82f6; border: 1px solid #3b82f6; }

QFrame#DropFrame {
  border: 2px dashed #2b3550;
  border-radius: 14px;
  background: #0f1115;
  min-height: 84px;
}
QLabel#DropLabel { color: #9aa3b2; font-size: 13px; }

QPushButton#HelpBadge {
  background: #1d2330;
  border: 1px solid #2b3550;
  border-radius: 9px;
  color: #cfd6e6;
  font-weight: 800;
  padding: 0px;
}
QPushButton#HelpBadge:hover { background: #232c3f; }
QPushButton#HelpBadge:pressed { background: #1a2130; }
"""