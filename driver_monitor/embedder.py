"""Face embeddings for recognising drivers: OpenCV's SFace network.

The landmark-shape signature in ``profiles.py`` describes the geometry of a face mesh. Meshes of
different people are often closer to each other than one person's mesh is to itself on another
day, so it cannot reliably tell two drivers apart. SFace is a small neural network trained to
do exactly that: it turns an aligned face crop into 128 numbers, and two photos of the same
person give vectors with a high cosine similarity.

No second detector is needed: the five alignment points (eye centres, nose tip, mouth corners)
come from the MediaPipe landmarks that the app already computes. Only the embedding is stored,
never the picture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import metrics

EMBED_DIM = 128
EMBED_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_recognition_sface_2021dec.onnx"
EMBED_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

_NOSE_TIP = 1
_MOUTH_CORNERS = (61, 291)


def alignment_row(points: np.ndarray) -> np.ndarray | None:
    """The 15-number 'face row' that OpenCV's aligner expects, built from the 478 landmarks.

    Layout: box x, y, w, h, then (x, y) of the right eye, left eye, nose tip, right mouth corner and
    left mouth corner, then a score. 'Right' and 'left' are the subject's, which is the image's
    left and right for a normal (not mirrored) view; they are ordered by position in the image so
    a mirrored frame works too.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[0] < 468 or p.shape[1] < 2:
        return None
    eye_a = p[list(metrics.RIGHT_EYE_CONTOUR), :2].mean(axis=0)
    eye_b = p[list(metrics.LEFT_EYE_CONTOUR), :2].mean(axis=0)
    if abs(eye_a[0] - eye_b[0]) < 1.0:
        return None
    mouth_a, mouth_b = p[_MOUTH_CORNERS[0], :2], p[_MOUTH_CORNERS[1], :2]
    if eye_a[0] > eye_b[0]:                 # keep "first" = the one on the image's left
        eye_a, eye_b = eye_b, eye_a
    if mouth_a[0] > mouth_b[0]:
        mouth_a, mouth_b = mouth_b, mouth_a
    x0, y0, x1, y1 = metrics.face_box(p)
    return np.array([x0, y0, x1 - x0, y1 - y0, *eye_a, *eye_b, *p[_NOSE_TIP, :2], *mouth_a, *mouth_b, 1.0],
                    dtype=np.float32).reshape(1, 15)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity. 0 = identical direction; unrelated faces are usually well above 0.6."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - float(a @ b) / (na * nb)


class FaceEmbedder:
    """Wraps ``cv2.FaceRecognizerSF``. Raises FileNotFoundError if the model has not been downloaded."""

    def __init__(self, model: Path | str = EMBED_MODEL) -> None:
        model = Path(model)
        if not model.exists():
            raise FileNotFoundError(f"{model} not found. Run: python -m driver_monitor.download_model")
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError("This OpenCV build has no FaceRecognizerSF (needs opencv-python 4.5.4 or newer)")
        self._rec = cv2.FaceRecognizerSF.create(str(model), "")

    def embed(self, frame_bgr: np.ndarray, points: np.ndarray) -> np.ndarray | None:
        """128-number description of the face in this frame, or None if it cannot be aligned."""
        row = alignment_row(points)
        if row is None:
            return None
        try:
            aligned = self._rec.alignCrop(frame_bgr, row)
            vec = self._rec.feature(aligned)
        except cv2.error:
            return None
        vec = np.asarray(vec, dtype=float).ravel()
        n = float(np.linalg.norm(vec))
        return vec / n if n > 1e-9 and vec.size == EMBED_DIM else None
