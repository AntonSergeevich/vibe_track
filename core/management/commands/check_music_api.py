"""Проверка подключения внешней нейросети одним реальным вызовом.

    python manage.py check_music_api we_angel.mp3 --seconds 20

Команда специально делает короткий запрос: он стоит копейки, а показывает
всё — принят ли ключ, верны ли имена полей, что именно возвращает провайдер.
Документация может устареть, живой ответ — нет.
"""
import os
import time

from django.core.management.base import BaseCommand, CommandError

from engine import generation
from engine.audio_io import load, save


class Command(BaseCommand):
    help = "Делает один запрос к API генерации и печатает всё, что вернулось."

    def add_arguments(self, parser):
        parser.add_argument("track", nargs="?", default="we_angel.mp3",
                            help="исходный трек для пробы")
        parser.add_argument("--seconds", type=float, default=0.0,
                            help="сколько секунд отправить; 0 — весь трек "
                                 "(вызов стоит одинаково при любой длине)")
        parser.add_argument("--prompt", default=generation.style_prompt_for("nu_metal"),
                            help="описание стиля; по умолчанию — тот же промпт, "
                                 "что уходит из студии")
        parser.add_argument("--strength", default="")
        parser.add_argument("--out", default="cover_test.mp3")

    def handle(self, *args, **options):
        provider = generation.provider_name()
        key = os.getenv("VIBETRACK_MUSIC_API_KEY", "")

        self.stdout.write(f"Провайдер: {provider}")
        self.stdout.write(f"Ключ: {'…' + key[-4:] if key else self.style.ERROR('не задан')}")
        if provider == "stability":
            self.stdout.write(f"Эндпоинт: {os.getenv('VIBETRACK_STABILITY_URL', generation.STABILITY_URL)}")
        if not generation.is_configured():
            raise CommandError(
                "Не хватает настроек. Задайте VIBETRACK_MUSIC_PROVIDER и VIBETRACK_MUSIC_API_KEY.")
        if not os.path.exists(options["track"]):
            raise CommandError(f"Файл {options['track']} не найден")

        if options["strength"]:
            os.environ["VIBETRACK_STABILITY_STRENGTH"] = options["strength"]
        source = load(options["track"])
        # цена не зависит от длины, поэтому по умолчанию слушаем весь трек:
        # на двадцати секундах не видно ни куплета, ни припева
        limit = float(os.getenv("VIBETRACK_STABILITY_MAX_SECONDS", "180"))
        seconds = min(options["seconds"] or source.duration, limit)
        os.environ["VIBETRACK_STABILITY_MAX_SECONDS"] = str(seconds)

        self.stdout.write(f"Исходник: {source.duration:.0f} с, отправляем "
                          f"{seconds:.0f} с")
        self.stdout.write(f"Сила переделки: "
                          f"{os.getenv('VIBETRACK_STABILITY_STRENGTH', '0.6')}")
        self.stdout.write(f"Промпт: {options['prompt']}")
        self.stdout.write("\nЗапрос пошёл…")

        started = time.time()
        result = generation.generate_cover(generation.CoverRequest(
            source_path=options["track"], style_prompt=options["prompt"],
            duration=seconds))
        elapsed = time.time() - started

        if not result.ok:
            self.stdout.write(self.style.ERROR(f"\nНе получилось за {elapsed:.0f} с:"))
            self.stdout.write(result.error)
            if "No module named" in result.error:
                # до провайдера дело даже не дошло — это окружение, а не API
                raise CommandError(
                    "Не хватает библиотеки в venv. Поставьте зависимости:\n"
                    "    pip install -r requirements.txt")
            self.stdout.write(
                "\nЧто обычно значат ответы:\n"
                "  401 / 403 — ключ неверный или не активирован\n"
                "  402       — закончились кредиты, пополните баланс\n"
                "  400       — не то имя поля или значение вне допустимого;\n"
                "              текст ошибки называет поле — пришлите его мне\n"
                "  429       — слишком часто, подождите минуту")
            raise CommandError("Проверка не прошла")

        path = save(options["out"], result.audio)
        self.stdout.write(self.style.SUCCESS(f"\nГотово за {elapsed:.0f} с"))
        self.stdout.write(f"Файл: {path} ({result.audio.duration:.0f} с)")
        self.stdout.write(f"Ориентировочная стоимость вызова: ${result.cost_usd:.2f}")
        self.stdout.write("\nПослушайте файл. Если звучит как нужный жанр — "
                          "включаем генерацию в студии галочкой «Заказать кавер».")
