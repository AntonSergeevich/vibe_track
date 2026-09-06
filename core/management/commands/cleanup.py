"""Что занимает место на диске — и как это освободить.

Веса моделей и результаты рендеров растут молча: Demucs и Whisper кладут
гигабайты в домашнюю папку (на Windows это диск C), а каждый рендер
оставляет мастер, кавер и десяток дорожек. Когда на системном диске
кончается место, Windows перестаёт наращивать файл подкачки — и процесс
умирает от нехватки памяти там, где памяти хватало бы.
"""
import os
import shutil
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone


def folder_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def human(size: int) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024 or unit == "ГБ":
            return f"{size:.0f} {unit}" if unit == "Б" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} ГБ"


class Command(BaseCommand):
    help = "Показывает, что занимает диск, и чистит старые рендеры и веса моделей."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30,
                            help="удалять рендеры старше скольких дней")
        parser.add_argument("--models", action="store_true",
                            help="удалить и скачанные веса моделей (скачаются заново)")
        parser.add_argument("--yes", action="store_true",
                            help="действительно удалять; без него — только показ")

    def handle(self, *args, **options):
        from engine.models import cache_dir

        media = Path(str(settings.MEDIA_ROOT))
        places = [
            ("Рендеры", media / "renders"),
            ("Загруженные треки", media / "audio"),
            ("Записи голоса", media / "takes"),
            ("Веса моделей VibeTrack", Path(cache_dir())),
            ("Кэш Hugging Face", Path.home() / ".cache" / "huggingface"),
            ("Кэш torch", Path.home() / ".cache" / "torch"),
            ("Кэш pip", Path.home() / "AppData" / "Local" / "pip" / "cache"
             if os.name == "nt" else Path.home() / ".cache" / "pip"),
        ]

        self.stdout.write("Что занимает место:\n")
        for title, path in places:
            size = folder_size(path)
            mark = "" if path.exists() else "  (нет)"
            self.stdout.write(f"  {title:26} {human(size):>10}  {path}{mark}")

        for label, path in (("Диск с проектом", Path(str(settings.BASE_DIR))),
                            ("Домашняя папка", Path.home())):
            try:
                usage = shutil.disk_usage(path)
                self.stdout.write(
                    f"\n{label} ({path.anchor or path}): свободно "
                    f"{human(usage.free)} из {human(usage.total)}")
            except OSError:
                pass

        self._clean_renders(media, options)
        if options["models"]:
            self._clean_models(Path(cache_dir()), options)

        if not options["yes"]:
            self.stdout.write(self.style.WARNING(
                "\nЭто был только показ. Чтобы удалить — добавьте --yes"))

    def _clean_renders(self, media: Path, options):
        """Удаляет папки рендеров, которых больше нет в базе или которые стары."""
        from core.models import RenderJob

        renders = media / "renders"
        if not renders.exists():
            return
        cutoff = timezone.now() - timedelta(days=options["days"])
        alive = set(RenderJob.objects.filter(created_at__gte=cutoff)
                    .values_list("pk", flat=True))

        freed, removed = 0, 0
        for folder in sorted(renders.iterdir()):
            if not folder.is_dir() or not folder.name.startswith("job_"):
                continue
            try:
                job_id = int(folder.name.removeprefix("job_"))
            except ValueError:
                continue
            if job_id in alive:
                continue
            size = folder_size(folder)
            freed += size
            removed += 1
            self.stdout.write(f"  удалить {folder.name}: {human(size)}")
            if options["yes"]:
                shutil.rmtree(folder, ignore_errors=True)
                RenderJob.objects.filter(pk=job_id).delete()

        verb = "Освобождено" if options["yes"] else "Освободится"
        self.stdout.write(
            f"\nСтарые рендеры (старше {options['days']} дн.): {removed} шт., "
            f"{verb} {human(freed)}")

    def _clean_models(self, cache: Path, options):
        size = folder_size(cache)
        self.stdout.write(
            f"\nВеса моделей: {human(size)} в {cache}. "
            "Скачаются заново при следующем рендере (минуты, не часы).")
        if options["yes"]:
            shutil.rmtree(cache, ignore_errors=True)
            self.stdout.write(self.style.SUCCESS(f"Удалено {human(size)}"))
