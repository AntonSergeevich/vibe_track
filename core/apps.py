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
