import numpy as np
from pixsorter.vision.scoring import best_of_score

def test_best_of_score_range_and_eyes_effect():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(720, 1080, 3), dtype=np.uint8)

    s_open = best_of_score(rgb, face_bboxes=[], eyes_open_frac=1.0, eyes_weight=3.0)
    s_closed = best_of_score(rgb, face_bboxes=[], eyes_open_frac=0.0, eyes_weight=3.0)

    assert 0.0 <= s_open <= 1.0
    assert 0.0 <= s_closed <= 1.0
    assert s_open >= s_closed