import logging
from typing import Callable, Optional


def setup_logging(ui_sink: Optional[Callable[[str], None]] = None) -> logging.Logger:
    """
    ui_sink: callable that receives log lines (e.g. MainWindow.append_log)
    """
    logger = logging.getLogger("pixsorter")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if ui_sink is not None:
        class UiHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    ui_sink(self.format(record))
                except Exception:
                    pass

        uh = UiHandler()
        uh.setLevel(logging.INFO)
        uh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        logger.addHandler(uh)

    return logger