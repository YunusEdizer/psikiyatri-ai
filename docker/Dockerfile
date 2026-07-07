# Psikiyatri AI — API servis imajı
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Önce bağımlılık manifestosu (katman önbelleği için)
COPY pyproject.toml README.md ./
COPY src ./src

# Servis için çekirdek + metin modalitesi bağımlılıkları
RUN pip install --upgrade pip && pip install -e ".[text]"

EXPOSE 8000

# Üretimde --reload kullanmayın
CMD ["uvicorn", "psikiyatri_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
