"""Yüz ifadesi ön işleme.

Videodan/kareden yüz bölgesini bulup hizalar ve yüz işaretçilerini
(landmark) çıkarır. Ağır bağımlılıklar (opencv, mediapipe) yalnızca
fonksiyon çağrıldığında yüklenir; böylece paketin geri kalanı bu
bağımlılıklar kurulmadan da import edilebilir (`[vision]` ekstrası).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# MediaPipe FaceMesh 468 işaretçi üretir; (x, y, z) => 3 kanal.
NUM_FACE_LANDMARKS = 468


@dataclass
class FaceFeatures:
    """Bir kareye ait çıkarılmış yüz özellikleri."""

    landmarks: np.ndarray  # şekil: (NUM_FACE_LANDMARKS, 3)
    action_units: dict[str, float] = field(default_factory=dict)  # FACS AU'ları
    detected: bool = True


def extract_face_landmarks(image: np.ndarray) -> FaceFeatures:
    """Tek bir kareden yüz işaretçilerini çıkarır.

    Args:
        image: RGB görüntü, şekil (H, W, 3).

    Returns:
        FaceFeatures. Yüz bulunamazsa ``detected=False`` ve sıfır dizisi.
    """
    try:
        import mediapipe as mp  # noqa: PLC0415  (isteğe bağlı bağımlılık)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Görüntü modalitesi için 'mediapipe' gerekli. "
            "Kurulum: pip install 'psikiyatri-ai[vision]'"
        ) from exc

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as mesh:
        result = mesh.process(image)

    if not result.multi_face_landmarks:
        return FaceFeatures(
            landmarks=np.zeros((NUM_FACE_LANDMARKS, 3), dtype=np.float32),
            detected=False,
        )

    lm = result.multi_face_landmarks[0].landmark
    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    return FaceFeatures(landmarks=coords, detected=True)


def sample_frames(video_path: Path, *, fps: float = 2.0) -> list[np.ndarray]:
    """Videodan belirli aralıklarla kare örnekler.

    Yüz ifadesi analizinde her kareyi işlemek gereksizdir; saniyede
    birkaç kare mikro-ifade dinamiğini yakalamaya yeterlidir.
    """
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Görüntü modalitesi için 'opencv-python' gerekli. "
            "Kurulum: pip install 'psikiyatri-ai[vision]'"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(native_fps / fps), 1)

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


__all__ = ["FaceFeatures", "extract_face_landmarks", "sample_frames", "NUM_FACE_LANDMARKS"]
