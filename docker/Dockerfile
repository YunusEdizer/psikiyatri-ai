# Psikiyatri AI — üretim API imajı (tüm modaliteler, CPU)
#
# Build (repo kökünden):
#   docker build -f docker/Dockerfile -t psikiyatri-ai .
# Genellikle doğrudan docker-compose ile kullanılır (bkz. DEPLOY.md).
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # HF (BERTurk/Whisper) indirmeleri kalıcı volume'a gitsin
    HF_HOME=/cache/hf

# Sistem bağımlılıkları:
# - ffmpeg    : tarayıcı WebM/Opus ses çözümü (librosa/audioread)
# - mediainfo : opensmile/eGeMAPS ses meta okuması
# - libgl1 + libglib2.0-0 : opencv/mediapipe çalışma zamanı
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg mediainfo libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Önce CPU torch (CUDA indirmesini ve ~4GB'lık imaj şişmesini önler)
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# Bağımlılık manifestosu + kaynak (katman önbelleği için sıralı)
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

# Tüm modaliteler: metin (BERTurk) + görüntü (mediapipe) + ses (librosa/opensmile)
# + yerel konuşma tanıma (faster-whisper)
RUN pip install -e ".[text,vision,audio,stt]"

EXPOSE 8020

# Sağlık kontrolü (curl yok; python yeterli)
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8020/health', timeout=4)" || exit 1

# Başlangıç: yüz modeli eksikse indir (models volume'una), sonra servis.
# psikiyatri_model.pt volume ile sağlanır (git'te değildir — bkz. DEPLOY.md);
# yoksa servis uyarı loglayıp yine açılır.
CMD ["sh", "-c", "python scripts/download_models.py || true; exec uvicorn psikiyatri_ai.api.main:app --host 0.0.0.0 --port 8020"]
