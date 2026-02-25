import numpy as np
import cv2
from pixsorter.vision.orb import orb_verify_same_frame

def _dot_pattern(seed: int, size=420, n=250):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.uint8)
    for _ in range(n):
        x = int(rng.integers(10, size-10))
        y = int(rng.integers(10, size-10))
        r = int(rng.integers(2, 6))
        cv2.circle(img, (x, y), r, 255, -1)
    return img

def test_orb_same_image_true():
    g1 = _dot_pattern(123)
    g2 = g1.copy()
    same, inliers, ratio = orb_verify_same_frame(
        g1, g2, orb_nfeatures=2500, match_ratio=0.75, min_inliers=20
    )
    assert same is True
    assert inliers >= 20

def test_orb_blank_image_false():
    g1 = _dot_pattern(123)
    g2 = np.zeros_like(g1)
    same, inliers, ratio = orb_verify_same_frame(
        g1, g2, orb_nfeatures=2500, match_ratio=0.75, min_inliers=20
    )
    assert same is False