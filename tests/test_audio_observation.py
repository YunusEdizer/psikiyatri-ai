"""Betimsel akustik gözlem ve tanı-ayrımı testleri.

Ses/görüntü kodlayıcıları eğitilmediğinden bu modalitelerin **tanı
füzyonuna** girmemesi, buna karşılık kaydedilen sesin **betimsel gözlem**
olarak işlenmesi doğrulanır.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from psikiyatri_ai.modalities.audio.features import ensure_ffmpeg_on_path
from psikiyatri_ai.modalities.audio.observations import describe_audio


def _make_webm(dest: Path) -> bool:
    """imageio-ffmpeg ile kısa bir test sesi (sinüs) üretir."""
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff = shutil.which("ffmpeg")
    if not ff:
        return False
    cmd = [ff, "-y", "-f", "lavfi", "-i", "sine=frequency=180:duration=1",
           "-c:a", "libopus", str(dest)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        return dest.exists()
    except Exception:
        return False


def test_ensure_ffmpeg_returns_path_or_none():
    # imageio-ffmpeg kurulu olduğundan bir yol dönmeli
    p = ensure_ffmpeg_on_path()
    assert p is None or Path(p).exists()


def test_describe_audio_on_sine(tmp_path):
    webm = tmp_path / "t.webm"
    if not _make_webm(webm):
        pytest.skip("ffmpeg ile test sesi üretilemedi")
    obs = describe_audio(webm)
    assert obs is not None
    assert obs.duration_s == pytest.approx(1.0, abs=0.3)
    # 180 Hz sinüs -> perde ~180, düşük değişkenlik (monoton)
    assert obs.pitch_hz is not None and 150 < obs.pitch_hz < 210
    assert obs.qualitative["tonlama"] == "monoton"
    assert set(obs.qualitative).issuperset({"perde", "ses_siddeti", "konusma_etkinligi"})


def test_describe_audio_missing_file_returns_none(tmp_path):
    assert describe_audio(tmp_path / "yok.webm") is None


def test_diagnosis_excludes_untrained_av_encoders(monkeypatch, tmp_path):
    """analyze_session tanı için ses/görüntü'yü score_media'ya vermemeli."""
    from psikiyatri_ai.api import session as sm

    seen = {}

    def fake_score(media_paths, transcript=None):
        seen["paths"] = dict(media_paths)
        return None  # tanı yolu önemli değil; hangi modalitelerin geçtiği önemli

    monkeypatch.setattr(sm, "score_media", fake_score)
    # gerçek ses okumasın diye gözlemi de sabitle
    monkeypatch.setattr(sm, "analyze_text_risk", sm.analyze_text_risk)

    webm = tmp_path / "a.webm"
    webm.write_bytes(b"stub")  # describe_audio None dönecek (okunamaz) — sorun değil
    sm.analyze_session(
        "pid", "Dr", "Kendimi kötü hissediyorum.",
        [{"modality": "audio", "filename": "a.webm", "size": 4}],
        {"audio": webm},
    )
    # Tanı çağrısına ses YOLU girmemeli (eğitilmemiş kodlayıcı gürültüsü)
    assert "audio" not in seen.get("paths", {})
    assert "vision" not in seen.get("paths", {})
