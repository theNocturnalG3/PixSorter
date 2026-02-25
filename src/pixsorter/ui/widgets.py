from pathlib import Path
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import QPushButton, QMessageBox, QFrame


class HelpButton(QPushButton):
    def __init__(self, parent, title: str, desc: str, minv=None, maxv=None, rec=None):
        super().__init__("?", parent)
        self._title = title
        self._desc = desc
        self._minv = minv
        self._maxv = maxv
        self._rec = rec

        self.setObjectName("HelpBadge")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(QSize(18, 18))
        self.setFocusPolicy(Qt.NoFocus)
        self.clicked.connect(self._show)

    def _show(self):
        parts = [f"<b>{self._title}</b><br><br>{self._desc}"]
        if self._minv is not None or self._maxv is not None:
            parts.append(f"<br><br><b>Range:</b> {self._minv} → {self._maxv}")
        if self._rec is not None:
            parts.append(f"<br><b>Recommended:</b> {self._rec}")
        QMessageBox.information(self.window(), self._title, "".join(parts))


def help_button(parent, title: str, desc: str, minv=None, maxv=None, rec=None) -> HelpButton:
    return HelpButton(parent, title, desc, minv=minv, maxv=maxv, rec=rec)


class DropFrame(QFrame):
    dropped = Signal(str)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setObjectName("DropFrame")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.exists() and p.is_dir():
                self.dropped.emit(str(p))
                break