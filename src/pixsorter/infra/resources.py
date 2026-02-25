import sys
from pathlib import Path
import ctypes

APP_ID = "com.pixsorter.app"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif"}


def init_heic():
    try:
        import pillow_heif  # type: ignore
        pillow_heif.register_heif_opener()
    except Exception:
        pass


def set_windows_app_id(app_id: str):
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def resource_root() -> Path:
    """
    Returns the package root (the folder that contains assets/).
    - Dev/installed: .../pixsorter/
    - PyInstaller:  _MEIPASS/pixsorter/   (because we bundle datas to pixsorter/assets)
    """
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS) / "pixsorter"  # type: ignore[attr-defined]
        if base.exists():
            return base
        return Path(sys._MEIPASS)  # fallback
    return Path(__file__).resolve().parents[1]


def resource_path(relative_path: str) -> str:
    p = resource_root() / relative_path
    return str(p) if p.exists() else ""