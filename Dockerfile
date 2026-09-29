FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ffmpeg нужен для mp3/m4a/webm (запись с микрофона приходит в webm),
# libsndfile — для wav/flac через soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-ml.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Тяжёлые модели ставятся отдельно: docker build --build-arg WITH_ML=1
# По умолчанию берём CPU-сборку torch: обычный пакет с PyPI тянет 3.2 ГБ
# библиотек CUDA, которые на VPS без видеокарты лежат мёртвым грузом.
# Для GPU-машины: --build-arg TORCH_INDEX=https://pypi.org/simple
ARG WITH_ML=0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN if [ "$WITH_ML" = "1" ]; then \
        pip install --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
            -r requirements-ml.txt; \
    fi

COPY . /app
RUN mkdir -p /app/media /app/staticfiles

CMD ["gunicorn", "vibetrack_site.wsgi:application", "--bind", "0.0.0.0:8000", \
     "--workers", "3", "--timeout", "120"]
