"""Yüz ifadesi ön işleme.

Videodan/kareden yüz bölgesini bulup hizalar, yüz işaretçilerini
(landmark) çıkarır ve bu işaretçilerden **yorumlanabilir geometrik
özellikler** (göz açıklığı/kırpma, ağız köşesi eğimi, kaş hareketi)
hesaplar. Bu ölçütler FACS Eylem Birimlerinin (AU) hafif bir vekilidir
ve hem uzmana sunulabilir hem de modele ek öznitelik olarak verilebilir.

Ağır bağımlılıklar (opencv, mediapipe) yalnızca fonksiyon çağrıldığında
yüklenir; böylece paketin geri kalanı bu bağımlılıklar kurulmadan da
import edilebilir (``[vision]`` ekstrası). Geometrik özellik hesabı
tamamen NumPy'dir; bağımlılık gerektirmez ve bağımsız test edilebilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# MediaPipe FaceMesh 468 işaretçi üretir; (x, y, z) => 3 kanal.
NUM_FACE_LANDMARKS = 468

# --- MediaPipe FaceMesh kanonik işaretçi indeksleri (0..467) --------------- #
# Göz açıklığı oranı (EAR) için 6'şar nokta (dış, üst×2, iç, alt×2 sırasıyla).
_LEFT_EYE = (362, 385, 387, 263, 373, 380)
_RIGHT_EYE = (33, 160, 158, 133, 153, 144)
# Ağız: köşeler + üst/alt dudak orta noktaları.
_MOUTH_L, _MOUTH_R = 61, 291
_LIP_TOP, _LIP_BOTTOM = 13, 14
# Kaş üstü ve göz üstü (kaş-göz mesafesi = kaş kalkması/çatılması).
_BROW_L, _EYE_TOP_L = 105, 159
_BROW_R, _EYE_TOP_R = 334, 386
# Yüz genişliği (yanak-yanak) — ölçek normalizasyonu için.
_CHEEK_L, _CHEEK_R = 234, 454

# ``geometric_feature_vector`` çıktısının sabit sırası (modele beslemek için).
GEOMETRIC_FEATURE_NAMES = (
    "goz_acikligi",       # ortalama EAR (düşük = kapalı/kırpma)
    "goz_asimetrisi",     # |sol EAR - sağ EAR|
    "agiz_acikligi",      # ağız yüksekliği / genişliği (MAR)
    "agiz_genisligi",     # ağız genişliği / yüz genişliği
    "agiz_kosesi_egimi",  # gülümseme(+) / düşük ağız köşesi(-)
    "kas_hareketi",       # kaş-göz mesafesi (kalkma/çatılma)
)


@dataclass
class FaceFeatures:
    """Bir kareye ait çıkarılmış yüz özellikleri."""

    landmarks: np.ndarray  # şekil: (NUM_FACE_LANDMARKS, 3)
    # Geometrik AU-vekili ölçütler (bkz. compute_geometric_features).
    action_units: dict[str, float] = field(default_factory=dict)
    detected: bool = True


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ear(pts: np.ndarray, idx: tuple[int, ...]) -> float:
    """Göz açıklığı oranı (Eye Aspect Ratio) — 2B (x, y) üzerinden."""
    p1, p2, p3, p4, p5, p6 = (pts[i, :2] for i in idx)
    horizontal = _dist(p1, p4)
    if horizontal == 0.0:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def compute_geometric_features(landmarks: np.ndarray) -> dict[str, float]:
    """Yüz işaretçilerinden yorumlanabilir geometrik özellikler çıkarır.

    Tüm ölçütler yüz genişliğine göre normalize edilir (kameraya
    uzaklıktan bağımsız). Dejenere/sıfır işaretçi durumunda tüm değerler
    0.0 döner.

    Args:
        landmarks: (NUM_FACE_LANDMARKS, 3) NumPy dizisi (normalize x, y, z).

    Returns:
        ``GEOMETRIC_FEATURE_NAMES`` anahtarlarını içeren sözlük.
    """
    zero = dict.fromkeys(GEOMETRIC_FEATURE_NAMES, 0.0)
    if landmarks is None or landmarks.shape[0] < NUM_FACE_LANDMARKS:
        return zero

    face_w = _dist(landmarks[_CHEEK_L, :2], landmarks[_CHEEK_R, :2])
    if face_w == 0.0:
        return zero

    ear_l = _ear(landmarks, _LEFT_EYE)
    ear_r = _ear(landmarks, _RIGHT_EYE)

    mouth_w = _dist(landmarks[_MOUTH_L, :2], landmarks[_MOUTH_R, :2])
    mouth_h = _dist(landmarks[_LIP_TOP, :2], landmarks[_LIP_BOTTOM, :2])

    # Ağız köşesi eğimi: dudak ortasının y'si - köşelerin ortalama y'si.
    # Görüntüde y aşağı doğru artar → pozitif = köşeler yukarıda (gülümseme).
    lip_center_y = float(landmarks[_LIP_TOP, 1])
    corner_y = float((landmarks[_MOUTH_L, 1] + landmarks[_MOUTH_R, 1]) / 2.0)
    smile = (lip_center_y - corner_y) / face_w

    brow_l = _dist(landmarks[_BROW_L, :2], landmarks[_EYE_TOP_L, :2])
    brow_r = _dist(landmarks[_BROW_R, :2], landmarks[_EYE_TOP_R, :2])

    return {
        "goz_acikligi": (ear_l + ear_r) / 2.0,
        "goz_asimetrisi": abs(ear_l - ear_r),
        "agiz_acikligi": _safe_ratio(mouth_h, mouth_w),
        "agiz_genisligi": mouth_w / face_w,
        "agiz_kosesi_egimi": smile,
        "kas_hareketi": (brow_l + brow_r) / (2.0 * face_w),
    }


def _safe_ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def geometric_feature_vector(landmarks: np.ndarray) -> np.ndarray:
    """Geometrik özellikleri sabit sıralı bir NumPy vektörüne çevirir."""
    feats = compute_geometric_features(landmarks)
    return np.array([feats[name] for name in GEOMETRIC_FEATURE_NAMES], dtype=np.float32)


def extract_face_landmarks(image: np.ndarray) -> FaceFeatures:
    """Tek bir kareden yüz işaretçilerini ve geometrik özellikleri çıkarır.

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
    return FaceFeatures(
        landmarks=coords,
        action_units=compute_geometric_features(coords),
        detected=True,
    )


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


__all__ = [
    "FaceFeatures",
    "GEOMETRIC_FEATURE_NAMES",
    "NUM_FACE_LANDMARKS",
    "compute_geometric_features",
    "extract_face_landmarks",
    "geometric_feature_vector",
    "sample_frames",
]
