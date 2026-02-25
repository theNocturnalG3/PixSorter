import numpy as np
import cv2

from .saliency import saliency_map, saliency_centroid


def _points_score(cx: float, cy: float, w: int, h: int, xs: list[float], ys: list[float]) -> float:
    pts = [(x, y) for x in xs for y in ys]
    d = min(np.hypot(cx - x, cy - y) for x, y in pts)
    norm = np.hypot(w, h)
    return float(np.clip(1.0 - d / (0.22 * norm), 0.0, 1.0))


def _leading_lines_score(gray: np.ndarray, target_xy: tuple[float, float]) -> float:
    h, w = gray.shape
    cx, cy = target_xy
    edges = cv2.Canny(gray, 60, 140)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=70,
        minLineLength=int(0.18 * min(w, h)),
        maxLineGap=12
    )
    if lines is None:
        return 0.0

    scores: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        vx, vy = (x2 - x1), (y2 - y1)
        vnorm = np.hypot(vx, vy) + 1e-9
        vx, vy = vx / vnorm, vy / vnorm

        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        tx, ty = (cx - mx), (cy - my)
        tnorm = np.hypot(tx, ty) + 1e-9
        tx, ty = tx / tnorm, ty / tnorm

        align = vx * tx + vy * ty
        length = vnorm / (min(w, h) + 1e-9)
        if align > 0:
            scores.append(float(align * np.clip(length, 0, 1)))

    if not scores:
        return 0.0
    return float(np.clip(np.mean(sorted(scores, reverse=True)[:8]), 0.0, 1.0))


def composition_score(bgr: np.ndarray) -> float:
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sm = saliency_map(bgr)
    cx, cy, conc = saliency_centroid(sm)

    thirds = _points_score(cx, cy, w, h, [w/3, 2*w/3], [h/3, 2*h/3])
    phi = _points_score(cx, cy, w, h, [0.382*w, 0.618*w], [0.382*h, 0.618*h])
    lines = _leading_lines_score(gray, (cx, cy))
    isolation = float(np.clip((conc - 0.15) / 0.35, 0.0, 1.0))

    return float(0.30*thirds + 0.25*phi + 0.30*lines + 0.15*isolation)