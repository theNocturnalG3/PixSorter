import numpy as np


class FaceEyeAnalyzer:
    """
    Lazy imports mediapipe.
    Install extras:
      pip install "pixsorter[eyes]"
    """
    def __init__(self):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.LEFT = dict(p1=33, p2=160, p3=158, p4=133, p5=153, p6=144)
        self.RIGHT = dict(p1=362, p2=385, p3=387, p4=263, p5=373, p6=380)

    def _ear(self, lm, idxs, w, h):
        def pt(i):
            return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)
        p1 = pt(idxs["p1"]); p4 = pt(idxs["p4"])
        p2 = pt(idxs["p2"]); p6 = pt(idxs["p6"])
        p3 = pt(idxs["p3"]); p5 = pt(idxs["p5"])
        num = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        den = 2.0 * (np.linalg.norm(p1 - p4) + 1e-6)
        return float(num / den)

    def analyze(self, rgb: np.ndarray) -> tuple[list[tuple[int, int, int, int]], float]:
        h, w = rgb.shape[:2]
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return [], 1.0

        face_bboxes = []
        open_count = 0
        total = 0

        for face in res.multi_face_landmarks:
            lm = face.landmark
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x0 = int(max(0, min(xs) * w))
            x1 = int(min(w, max(xs) * w))
            y0 = int(max(0, min(ys) * h))
            y1 = int(min(h, max(ys) * h))
            face_bboxes.append((x0, y0, x1 - x0, y1 - y0))

            left = self._ear(lm, self.LEFT, w, h)
            right = self._ear(lm, self.RIGHT, w, h)
            ear = 0.5 * (left + right)

            total += 1
            if ear > 0.22:
                open_count += 1

        return face_bboxes, float(open_count / max(1, total))