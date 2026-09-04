# vibetrack_site/celery.py
import logging
import os

from celery import Celery
from celery.signals import worker_ready

logger = logging.getLogger(__name__)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vibetrack_site.settings')

app = Celery('vibetrack_site')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


@worker_ready.connect
def warm_models(**_kwargs):  # pragma: no cover - выполняется только в воркере
    """Прогрев моделей при старте воркера.

    Первая загрузка Demucs и Whisper занимает десятки секунд. Без прогрева
    их ждёт первый же пользователь — поэтому грузим заранее, если включено.
    """
    if os.getenv('VIBETRACK_PRELOAD_MODELS', '0').strip().lower() not in ('1', 'true', 'yes', 'on'):
        return
    from engine import models

    for name, result in models.preload().items():
        logger.info('Модель %s: %s', name, result)


@app.task(bind=True)
def debug_task(self):  # pragma: no cover
    print(f'Request: {self.request!r}')
