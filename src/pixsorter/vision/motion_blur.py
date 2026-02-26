import math
import cv2
import numpy as np
from .io import resize_max_side, rgb_to_gray

BLUR_PRESENT_MIN = 0.18       
COHERENCE_THRESH = 0.18       
FAULTY_MAX = 0.55             

def _blur_amount(gray: np.ndarray) -> float:
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    lv = float(cv2.Laplacian(g, cv2.CV_64F).var())
    sharp = float(np.clip((math.log1p(lv) - 3.5) / 2.0, 0.0, 1.0))
    return float(np.clip(1.0 - sharp, 0.0, 1.0))

def _coherence(gray: np.ndarray) -> float:
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)

    a = float(np.mean(jxx)); b = float(np.mean(jxy)); c = float(np.mean(jyy))
    tr = a + c
    tmp = math.sqrt(max((a - c) * (a - c) + 4.0 * b * b, 0.0))
    l1 = 0.5 * (tr + tmp)
    l2 = 0.5 * (tr - tmp)
    return float(np.clip((l1 - l2) / (l1 + l2 + 1e-9), 0.0, 1.0))

def assess_motion_blur(rgb: np.ndarray) -> dict:
    rgb_small = resize_max_side(rgb, 900)
    gray = rgb_to_gray(rgb_small)

    blur = _blur_amount(gray)
    coh = _coherence(gray)

    is_motion = (blur >= BLUR_PRESENT_MIN) and (coh >= COHERENCE_THRESH)

    label = ""
    if is_motion:
        label = "MB_FAULTY" if blur <= FAULTY_MAX else "MB_INTENTIONAL"

    return {"blur_amount": blur, "coherence": coh, "is_motion_blur": is_motion, "label": label}