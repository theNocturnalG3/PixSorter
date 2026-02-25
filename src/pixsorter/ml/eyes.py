import numpy as np
import mediapipe as mp

from ..infra.resources import resource_path


class FaceEyeAnalyzer:
    """
    Uses MediaPipe *Tasks* FaceLandmarker (not legacy mp.solutions).

    Requires model at:
      src/pixsorter/assets/models/face_landmarker.task
    """

    # FaceMesh landmark indices for EAR calculation (works with FaceLandmarker output)
    LEFT = dict(p1=33, p2=160, p3=158, p4=133, p5=153, p6=144)
    RIGHT = dict(p1=362, p2=385, p3=387, p4=263, p5=373, p6=380)

    def __init__(self, ear_open_thresh: float = 0.22, max_faces: int = 5):
        self.ear_open_thresh = float(ear_open_thresh)

        model_path = resource_path("assets/models/face_landmarker.task")
        if not model_path:
            raise FileNotFoundError(
                "Missing face_landmarker.task. Run: python scripts/download_models.py "
                "to download into src/pixsorter/assets/models/face_landmarker.task"
            )

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=max_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def close(self):
        # Safe cleanup (not strictly required, but good practice)
        try:
            self.landmarker.close()
        except Exception:
            pass

    @staticmethod
    def _ear(lm_list, idxs, w, h) -> float:
        def pt(i: int) -> np.ndarray:
            lm = lm_list[i]
            return np.array([lm.x * w, lm.y * h], dtype=np.float32)

        p1 = pt(idxs["p1"]); p4 = pt(idxs["p4"])
        p2 = pt(idxs["p2"]); p6 = pt(idxs["p6"])
        p3 = pt(idxs["p3"]); p5 = pt(idxs["p5"])
        num = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        den = 2.0 * (np.linalg.norm(p1 - p4) + 1e-6)
        return float(num / den)

    def analyze(self, rgb: np.ndarray) -> tuple[list[tuple[int, int, int, int]], float]:
        """
        Returns:
          face_bboxes: [(x,y,w,h), ...] in pixels
          eyes_open_frac: fraction of faces with EAR > threshold
        """
        h, w = rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)
        faces = result.face_landmarks or []
        if not faces:
            return [], 1.0

        face_bboxes: list[tuple[int, int, int, int]] = []
        open_count = 0
        total = 0

        for lm_list in faces:
            xs = [lm.x for lm in lm_list]
            ys = [lm.y for lm in lm_list]
            x0 = int(max(0, min(xs) * w))
            x1 = int(min(w, max(xs) * w))
            y0 = int(max(0, min(ys) * h))
            y1 = int(min(h, max(ys) * h))
            face_bboxes.append((x0, y0, x1 - x0, y1 - y0))

            left = self._ear(lm_list, self.LEFT, w, h)
            right = self._ear(lm_list, self.RIGHT, w, h)
            ear = 0.5 * (left + right)

            total += 1
            if ear > self.ear_open_thresh:
                open_count += 1

        return face_bboxes, float(open_count / max(1, total))