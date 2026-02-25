import math
import cv2
import numpy as np

from .saliency import saliency_map


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def roi_sharpness_score(rgb: np.ndarray, face_bboxes: list[tuple[int, int, int, int]] | None) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if face_bboxes:
        x, y, w, h = max(face_bboxes, key=lambda b: b[2] * b[3])
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
        roi = gray[y0:y1, x0:x1]
        val = laplacian_var(roi) if roi.size else laplacian_var(gray)
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        sm = saliency_map(bgr)
        m = (sm > 0.65).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        ys, xs = np.where(m > 0)
        if len(xs) > 50:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            roi = gray[y0:y1 + 1, x0:x1 + 1]
            val = laplacian_var(roi) if roi.size else laplacian_var(gray)
        else:
            val = laplacian_var(gray)

    return float(np.clip((math.log1p(val) - 3.5) / 2.0, 0.0, 1.0))