import math
import cv2
import numpy as np
from .io import resize_max_side, rgb_to_gray

# Tune these if needed:
COHERENCE_THRESH = 0.35      # higher => stricter "motion-like" requirement
BLUR_PRESENT_MIN = 0.35      # ignore tiny blur
FAULTY_MAX = 0.65            # <= moderate => faulty, > => intentional

def _blur_amount_from_laplacian(gray: np.ndarray) -> float:
    # Similar scaling to sharpness.py mapping (log1p + clamp), but inverted
    lv = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp = float(np.clip((math.log1p(lv) - 3.5) / 2.0, 0.0, 1.0))
    return float(np.clip(1.0 - sharp, 0.0, 1.0))

def _structure_tensor_coherence(gray: np.ndarray) -> tuple[float, float]:
    # returns (coherence in [0,1], dominant angle radians)
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)

    a = float(np.mean(jxx))
    b = float(np.mean(jxy))
    c = float(np.mean(jyy))

    tr = a + c
    tmp = math.sqrt(max((a - c) * (a - c) + 4.0 * b * b, 0.0))
    l1 = 0.5 * (tr + tmp)
    l2 = 0.5 * (tr - tmp)

    coherence = (l1 - l2) / (l1 + l2 + 1e-9)
    angle = 0.5 * math.atan2(2.0 * b, (a - c) + 1e-9)
    return float(np.clip(coherence, 0.0, 1.0)), float(angle)

def assess_motion_blur(rgb: np.ndarray) -> dict:
    """
    Returns:
      {
        "blur_amount": 0..1,
        "coherence": 0..1,
        "is_motion_blur": bool,
        "label": "", "MB_FAULTY", or "MB_INTENTIONAL"
      }
    """
    rgb_small = resize_max_side(rgb, 900)
    gray = rgb_to_gray(rgb_small)

    blur_amount = _blur_amount_from_laplacian(gray)
    coherence, _ = _structure_tensor_coherence(gray)

    is_motion = (blur_amount >= BLUR_PRESENT_MIN) and (coherence >= COHERENCE_THRESH)

    label = ""
    if is_motion:
        # your rule:
        if blur_amount <= FAULTY_MAX:
            label = "MB_FAULTY"
        else:
            label = "MB_INTENTIONAL"

    return {
        "blur_amount": blur_amount,
        "coherence": coherence,
        "is_motion_blur": is_motion,
        "label": label,
    }