import cv2
import numpy as np

from .composition import composition_score
from .sharpness import roi_sharpness_score
from .straightness import straightness_score
from .io import resize_max_side


def best_of_score(
    rgb: np.ndarray,
    face_bboxes: list[tuple[int, int, int, int]],
    eyes_open_frac: float,
    eyes_weight: float,
) -> float:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = resize_max_side(bgr, 1200)
    rgb2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    sharp = roi_sharpness_score(rgb2, face_bboxes)
    comp = composition_score(bgr)
    straight = straightness_score(gray)
    eyes = float(np.clip(eyes_open_frac, 0.0, 1.0))

    base = 0.55 * sharp + 0.25 * comp + 0.20 * straight
    score = base * (0.55 + 0.45 * (eyes ** eyes_weight))
    return float(np.clip(score, 0.0, 1.0))