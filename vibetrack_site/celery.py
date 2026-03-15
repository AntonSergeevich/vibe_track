# vibetrack_site/celery.py
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'vibetrack_site.settings')

app = Celery('vibetrack_site')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
