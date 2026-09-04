"""Проверка и прогрев нейросетевых моделей.

    python manage.py models             # что установлено и доступно
    python manage.py models --preload   # скачать веса и прогреть (делать до приёма трафика)
"""
from django.conf import settings
from django.core.management.base import BaseCommand

from engine import models as engine_models


class Command(BaseCommand):
    help = "Показывает состояние моделей (Demucs, Whisper, LLM) и прогревает их."

    def add_arguments(self, parser):
        parser.add_argument("--preload", action="store_true",
                            help="Загрузить веса в память (и скачать при первом запуске)")

    def handle(self, *args, **options):
        self.stdout.write(f"Кэш весов: {engine_models.cache_dir()}")
        self.stdout.write(f"Потоков torch: {engine_models.configure_torch_threads()}")
        self.stdout.write(f"Бэкенд разделения: {settings.VIBETRACK['SEPARATION_BACKEND']}")

        if options["preload"]:
            self.stdout.write("\nПрогрев моделей…")
            for name, result in engine_models.preload().items():
                self.stdout.write(f"  {name}: {result}")

        self.stdout.write("")
        for name, info in engine_models.status().items():
            mark = self.style.SUCCESS("есть") if info.available else self.style.WARNING("нет")
            loaded = " (в памяти)" if info.loaded else ""
            seconds = f", загрузка {info.load_seconds} с" if info.load_seconds else ""
            self.stdout.write(f"  {name:8s} [{mark}] {info.name} — {info.detail}{loaded}{seconds}")

        missing = [n for n, i in engine_models.status().items() if not i.available]
        if missing:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING(
                f"Недоступно: {', '.join(missing)}. "
                "Модели ставятся из requirements-ml.txt, ключ LLM — в ANTHROPIC_API_KEY. "
                "Сервис работает и без них: разделение уйдёт в DSP-режим, "
                "текст можно вписать вручную, описание разберётся по ключевым словам."))
