import cv2
import numpy as np


def orb_verify_same_frame(
    gray1: np.ndarray,
    gray2: np.ndarray,
    orb_nfeatures: int,
    match_ratio: float,
    min_inliers: int,
) -> tuple[bool, int, float]:
    orb = cv2.ORB_create(nfeatures=orb_nfeatures)
    k1, d1 = orb.detectAndCompute(gray1, None)
    k2, d2 = orb.detectAndCompute(gray2, None)

    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return False, 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in pairs:
        if m.distance < match_ratio * n.distance:
            good.append(m)

    if len(good) < 12:
        return False, 0, len(good) / max(1, len(pairs))

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])

    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if mask is None:
        return False, 0, len(good) / max(1, len(pairs))

    inliers = int(mask.sum())
    ratio = len(good) / max(1, len(pairs))
    return (inliers >= min_inliers), inliers, ratio