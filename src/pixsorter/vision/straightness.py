import numpy as np
import cv2


def tilt_estimate_degrees(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 60, 140)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None:
        return 0.0
    angles: list[float] = []
    for rho, theta in lines[:, 0]:
        a = (theta - np.pi / 2.0)
        deg = abs(a) * 180.0 / np.pi
        if deg < 25:
            angles.append(float(deg))
    if not angles:
        return 0.0
    return float(np.median(angles))


def straightness_score(gray: np.ndarray) -> float:
    tilt = tilt_estimate_degrees(gray)
    if tilt <= 3.0:
        return 1.0
    return float(np.clip(1.0 - (tilt - 3.0) / 9.0, 0.0, 1.0))