# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg build-essential libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

# Собираем статику при сборке если нужно
# RUN python manage.py collectstatic --noinput

CMD ["gunicorn", "vibetrack_site.wsgi:application", "--bind", "0.0.0.0:8000"]
