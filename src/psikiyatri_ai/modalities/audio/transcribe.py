"""Sunucu-taraflı konuşma tanıma (faster-whisper).

Tarayıcının bulut STT hizmeti yerine, kaydedilen sesi **yerelde**
(sunucuda) Türkçe metne çevirir; böylece ham ses üçüncü taraf bir
hizmete gönderilmez (KVKK açısından daha güvenli). Model tembel yüklenir
ve önbelleğe alınır; CUDA varsa float16, yoksa CPU int8 kullanılır.

faster-whisper kurulu değilse ``transcribe_audio`` ``None`` döner ve
çağıran zarifçe tarayıcı STT'sine geri düşer.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .features import ensure_ffmpeg_on_path

#: Varsayılan model boyutu. "small" Türkçe için hız/doğruluk dengesi iyidir;
#: ``PAI_WHISPER_MODEL`` ortam değişkeniyle geçersiz kılınabilir.
DEFAULT_MODEL = os.environ.get("PAI_WHISPER_MODEL", "small")


@lru_cache(maxsize=1)
def _model(size: str):  # noqa: ANN202
    """WhisperModel'i bir kez kurar.

    Varsayılan CPU int8'dir: kısa yanıt klipleri için yeterince hızlıdır
    (~1 sn model yükleme + saniye altı çıkarım) ve Windows'ta CUDA için
    gereken cuDNN/cuBLAS DLL bağımlılıklarını atlar. CUDA ortamı hazırsa
    ``PAI_WHISPER_DEVICE=cuda`` ile açılabilir.
    """
    from faster_whisper import WhisperModel  # noqa: PLC0415

    device = os.environ.get("PAI_WHISPER_DEVICE", "cpu")
    compute = "float16" if device == "cuda" else "int8"
    try:
        return WhisperModel(size, device=device, compute_type=compute)
    except Exception:  # noqa: BLE001 - CUDA DLL/uyumsuzluk → CPU'ya düş
        return WhisperModel(size, device="cpu", compute_type="int8")


def is_available() -> bool:
    """faster-whisper içe aktarılabiliyor (ve içe aktarımı bozuk değil) mi?"""
    try:
        import faster_whisper  # noqa: F401, PLC0415

        return True
    except Exception:  # noqa: BLE001 - ImportError + bozuk native DLL/OSError vb.
        return False


def transcribe_audio(
    audio_path: Path, *, language: str = "tr", model_size: str | None = None
) -> str | None:
    """Ses dosyasını (WebM/WAV/…) yerelde Türkçe metne çevirir.

    faster-whisper yoksa ya da çözme başarısızsa ``None`` döner (akış
    kesilmez). Boş/çok kısa seste boş dize dönebilir.
    """
    if not is_available():
        return None
    ensure_ffmpeg_on_path()  # WebM/Opus çözümü için
    try:
        model = _model(model_size or DEFAULT_MODEL)
        segments, _info = model.transcribe(
            str(audio_path), language=language, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception:  # noqa: BLE001 - okunamayan medya/model hatası
        return None


__all__ = ["DEFAULT_MODEL", "is_available", "transcribe_audio"]
