import cv2
import numpy as np


def saliency_map(bgr: np.ndarray) -> np.ndarray:
    # Prefer contrib saliency if present, else fallback proxy.
    try:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, sm = sal.computeSaliency(bgr)
        if ok:
            sm = cv2.GaussianBlur(sm.astype(np.float32), (0, 0), 3)
            sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-6)
            return sm
    except Exception:
        pass

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    edges = cv2.Canny((gray * 255).astype("uint8"), 80, 160).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    proxy = abs(lap) + 0.7 * edges
    proxy = cv2.GaussianBlur(proxy, (0, 0), 3)
    proxy = (proxy - proxy.min()) / (proxy.max() - proxy.min() + 1e-6)
    return proxy


def saliency_centroid(sm: np.ndarray) -> tuple[float, float, float]:
    h, w = sm.shape
    s = sm.astype("float64")
    total = s.sum() + 1e-9
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((xs * s).sum() / total)
    cy = float((ys * s).sum() / total)

    flat = s.flatten()
    k = max(1, int(0.05 * flat.size))
    top = np.partition(flat, -k)[-k:].sum()
    conc = float(top / total)
    return cx, cy, conc
