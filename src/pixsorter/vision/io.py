from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.array(img)


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def resize_max_side(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = max(h, w) / max_side if max(h, w) > max_side else 1.0
    if scale > 1.0:
        arr = cv2.resize(arr, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
    return arr