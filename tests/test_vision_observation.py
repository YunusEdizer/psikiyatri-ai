"""Betimsel yüz gözlemi testleri (MediaPipe FaceLandmarker).

Model paketi veya OpenCV yoksa testler atlanır. Gerçek yüz içeren bir
video üretmek zor olduğundan, davranış (zarif None/boş dönüş, yüz
saptanamayınca gözlem yapısı) doğrulanır.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from psikiyatri_ai.modalities.vision import observations as vo


def _make_webm(dest: Path) -> bool:
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False
    cmd = [ff, "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=8",
           "-c:v", "libvpx", str(dest)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        return dest.exists()
    except Exception:
        return False


def test_describe_video_missing_file_returns_none(tmp_path):
    assert vo.describe_video(tmp_path / "yok.webm") is None


def test_describe_video_no_face_graceful(tmp_path):
    if vo._resolve_model() is None:
        pytest.skip("face_landmarker.task model paketi yok")
    webm = tmp_path / "t.webm"
    if not _make_webm(webm):
        pytest.skip("ffmpeg ile test videosu üretilemedi")
    obs = vo.describe_video(webm)
    # Kareler analiz edildi ama test deseninde yüz yok
    assert obs is not None
    assert obs.frames_analyzed > 0
    assert obs.face_detected_ratio == 0.0
    assert obs.qualitative.get("yuz") == "saptanamadı"


def test_observation_to_dict_none():
    assert vo.observation_to_dict(None) is None


def test_qualitative_labels_bounds():
    q = vo._qualitative(0.9, 0.10, 0.30, 0.30)
    assert q["kamera_etkilesimi"] == "yüksek"
    assert q["ifade_degiskenligi"] == "yüksek"
    assert q["gulumseme"] == "belirgin"
    q2 = vo._qualitative(0.1, 0.001, 0.0, 0.0)
    assert q2["kamera_etkilesimi"] == "düşük"
    assert q2["ifade_degiskenligi"] == "düşük"
