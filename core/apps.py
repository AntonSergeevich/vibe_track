from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'
    verbose_name = 'VibeTrack'

    def ready(self):
        from django.db.backends.signals import connection_created
        from django.dispatch import receiver

        @receiver(connection_created)
        def enable_sqlite_wal(sender, connection, **kwargs):
            """WAL нужен, когда рендер идёт в фоновом потоке.

            Иначе поток пишет прогресс, а HTTP-запрос в это время читает — и
            SQLite отвечает «database is locked».
            """
            if connection.vendor == 'sqlite':
                with connection.cursor() as cursor:
                    cursor.execute('PRAGMA journal_mode=WAL;')
                    cursor.execute('PRAGMA busy_timeout=5000;')

        self._recover_orphaned_jobs()

    def _recover_orphaned_jobs(self):
        """Закрывает рендеры, оборванные падением или перезапуском процесса.

        При работе в фоновом потоке задача живёт внутри сервера: упал
        процесс — поток умер вместе с ним, а задача навсегда осталась «в
        работе». Браузер продолжает опрашивать статус, который никогда не
        изменится, и человек смотрит на 25% до скончания века.

        Задачи Celery живут в отдельном воркере и переживают перезапуск
        сервера, поэтому их не трогаем.
        """
        import os
        import sys

        from django.conf import settings

        if not getattr(settings, 'VIBETRACK_INLINE_WORKER', False):
            return
        # автоперезагрузчик запускает ready() дважды; чинить нужно один раз,
        # в рабочем процессе, и никогда — во время миграций или тестов
        if os.environ.get('RUN_MAIN') == 'false' or 'test' in sys.argv:
            return

        try:
            from . import billing
            from .models import RenderJob

            stuck = list(RenderJob.objects.filter(status=RenderJob.STATUS_RUNNING))
            for job in stuck:
                job.status = RenderJob.STATUS_ERROR
                job.stage = 'error'
                job.error = ('Обработка прервалась: сервер был перезапущен или '
                             'ему не хватило памяти. Списание возвращено, '
                             'можно запустить трек заново.')
                job.save(update_fields=['status', 'stage', 'error', 'updated_at'])
                owner = job.track.project.owner if job.track.project_id else None
                if owner is not None:
                    billing.refund(owner, job, 'обработка прервана')
                    billing.refund_cover(owner, job, 'обработка прервана')
            if stuck:
                import logging

                logging.getLogger(__name__).warning(
                    'Оборванных рендеров закрыто: %s', len(stuck))
        except Exception:  # noqa: BLE001 — база может быть ещё не мигрирована
            pass
