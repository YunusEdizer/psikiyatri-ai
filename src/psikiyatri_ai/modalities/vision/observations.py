"""Betimsel (descriptive) yüz gözlemleri.

Kaydedilen videodan **doğrudan yorumlanabilir** yüz ifadesi ölçütleri
çıkarır: yüz saptanma (kamera etkileşimi) oranı, göz kırpma, ifade
değişkenliği (düz duygulanım göstergesi olabilir), gülümseme ve kaş
hareketi. Bu ölçütler klinisyene sunulan gözlemsel sinyallerdir;
eğitilmiş bir sınıflandırıcı GEREKTİRMEZ ve tanı üretmez.

Ayrım önemlidir (bkz. ``audio/observations.py``): proje modelinin görüntü
*kodlayıcısı* henüz etiketli video ile eğitilmediğinden görüntü, tanı
füzyonuna katılmaz (katılırsa gürültü ekler). Aşağıdaki ölçütler ham
karelerden MediaPipe FaceLandmarker (Tasks API) blendshape'leriyle
deterministik hesaplanır ve gözlemsel olarak güvenilirdir
(proje tanımı: "yüz kaslarının hareketleri, göz hareketleri").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

# Değerlendirilen blendshape'ler (MediaPipe kategori adları -> ölçüt).
_BLINK = ("eyeBlinkLeft", "eyeBlinkRight")
_SMILE = ("mouthSmileLeft", "mouthSmileRight")
_BROW_DOWN = ("browDownLeft", "browDownRight")
# İfade değişkenliği için izlenen dışavurum kanalları (düz duygulanım tespiti).
_EXPRESSIVE = (
    "mouthSmileLeft", "mouthSmileRight", "browInnerUp",
    "browDownLeft", "browDownRight", "jawOpen",
    "eyeWideLeft", "eyeWideRight", "mouthFrownLeft", "mouthFrownRight",
)

#: Varsayılan model paketi (models/ altına indirilir).
DEFAULT_MODEL = "face_landmarker.task"


@dataclass
class FaceObservation:
    """Betimsel yüz gözlemi (tanı değil)."""

    frames_analyzed: int
    face_detected_ratio: float  # yüz saptanan kare oranı (kamera etkileşimi)
    blink: float | None  # ortalama göz kırpma etkinliği (0-1)
    expressivity: float | None  # dışavurum değişkenliği (düşük = düz/monoton)
    smile: float | None  # ortalama gülümseme (0-1)
    brow_lower: float | None  # ortalama kaş çatma (0-1)
    qualitative: dict[str, str]


@lru_cache(maxsize=1)
def _landmarker(model_path: str):  # noqa: ANN202
    """FaceLandmarker'ı (Tasks API) bir kez kurar; blendshape üretir."""
    from mediapipe.tasks import python  # noqa: PLC0415
    from mediapipe.tasks.python import vision  # noqa: PLC0415

    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=True,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def _resolve_model() -> Path | None:
    """Model paketini config'teki model dizininde arar."""
    from ...config import get_settings  # noqa: PLC0415

    p = get_settings().model_dir / DEFAULT_MODEL
    return p if p.exists() else None


def describe_video(
    video_path: Path, *, fps: float = 2.0, max_frames: int = 40
) -> FaceObservation | None:
    """Videodan betimsel yüz gözlemlerini çıkarır.

    Model paketi yoksa, okuma/çözme başarısızsa veya hiç kare yoksa ``None``
    döner (rapor akışını kesmez). Yüz hiç saptanmazsa gözlem yine döner
    (``face_detected_ratio == 0``), nitel etikette "yüz saptanamadı".
    """
    try:
        import cv2  # noqa: PLC0415
        import mediapipe as mp  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return None

    model = _resolve_model()
    if model is None:
        return None

    from ..audio.features import ensure_ffmpeg_on_path  # noqa: PLC0415

    ensure_ffmpeg_on_path()  # WebM çözümü için (OpenCV ffmpeg arka ucu)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(native / fps), 1)

    try:
        lm = _landmarker(str(model))
    except Exception:  # noqa: BLE001 - model yüklenemezse gözlem atlanır
        cap.release()
        return None

    def _bs_mean(bs: dict, names: tuple[str, ...]) -> float:
        vals = [bs[n] for n in names if n in bs]
        return float(np.mean(vals)) if vals else 0.0

    analyzed = 0
    faces = 0
    blink_v: list[float] = []
    smile_v: list[float] = []
    brow_v: list[float] = []
    expr_frames: list[list[float]] = []  # kare başına dışavurum vektörü

    idx = 0
    while analyzed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        idx += 1
        analyzed += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            res = lm.detect(img)
        except Exception:  # noqa: BLE001
            continue
        if not res.face_blendshapes:
            continue
        faces += 1
        bs = {b.category_name: b.score for b in res.face_blendshapes[0]}
        blink_v.append(_bs_mean(bs, _BLINK))
        smile_v.append(_bs_mean(bs, _SMILE))
        brow_v.append(_bs_mean(bs, _BROW_DOWN))
        expr_frames.append([bs.get(n, 0.0) for n in _EXPRESSIVE])
    cap.release()

    if analyzed == 0:
        return None

    ratio = faces / analyzed
    if faces == 0:
        return FaceObservation(
            frames_analyzed=analyzed, face_detected_ratio=0.0,
            blink=None, expressivity=None, smile=None, brow_lower=None,
            qualitative={"yuz": "saptanamadı"},
        )

    blink = round(float(np.mean(blink_v)), 3)
    smile = round(float(np.mean(smile_v)), 3)
    brow = round(float(np.mean(brow_v)), 3)
    # İfade değişkenliği: her kanalın kareler arası std'sinin ortalaması.
    expr_arr = np.asarray(expr_frames)
    expressivity = round(float(np.mean(np.std(expr_arr, axis=0))), 4) if len(expr_arr) > 1 else 0.0

    return FaceObservation(
        frames_analyzed=analyzed,
        face_detected_ratio=round(ratio, 3),
        blink=blink, expressivity=expressivity, smile=smile, brow_lower=brow,
        qualitative=_qualitative(ratio, expressivity, smile, brow),
    )


def _qualitative(ratio: float, expr: float, smile: float, brow: float) -> dict[str, str]:
    """Ham ölçütleri nötr, betimsel Türkçe etiketlere çevirir (tanı değil)."""
    q = {
        "kamera_etkilesimi": "düşük" if ratio < 0.4 else "yüksek" if ratio > 0.8 else "orta",
        # Düşük değişkenlik = az dışavurum (düz/donuk duygulanım göstergesi olabilir).
        "ifade_degiskenligi": "düşük" if expr < 0.02 else "yüksek" if expr > 0.08 else "orta",
        "gulumseme": "az" if smile < 0.05 else "belirgin" if smile > 0.25 else "orta",
        "kas_catma": "az" if brow < 0.05 else "belirgin" if brow > 0.25 else "orta",
    }
    return q


def observation_to_dict(obs: FaceObservation | None) -> dict | None:
    return asdict(obs) if obs is not None else None


__all__ = ["FaceObservation", "describe_video", "observation_to_dict"]
