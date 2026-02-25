import sys
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from .infra.resources import init_heic, set_windows_app_id, APP_ID, resource_path
from .ui.styles import DARK_QSS
from .ui.main_window import MainWindow


def main():
    init_heic()
    set_windows_app_id(APP_ID)

    app = QApplication(sys.argv)

    # Global icon helps taskbar + dialogs
    ico = resource_path("assets/app.ico")
    if ico:
        app.setWindowIcon(QIcon(ico))

    app.setStyleSheet(DARK_QSS)

    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())