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
        parser.add_argument("--seconds", type=float, default=20.0,
                            help="сколько секунд отправить (короче — дешевле)")
        parser.add_argument("--prompt", default="aggressive nu metal, downtuned guitars, heavy drums")
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
        os.environ["VIBETRACK_STABILITY_MAX_SECONDS"] = str(options["seconds"])

        source = load(options["track"])
        self.stdout.write(f"Исходник: {source.duration:.0f} с, отправляем первые "
                          f"{options['seconds']:.0f} с")
        self.stdout.write(f"Промпт: {options['prompt']}")
        self.stdout.write("\nЗапрос пошёл…")

        started = time.time()
        result = generation.generate_cover(generation.CoverRequest(
            source_path=options["track"], style_prompt=options["prompt"],
            duration=options["seconds"]))
        elapsed = time.time() - started

        if not result.ok:
            self.stdout.write(self.style.ERROR(f"\nНе получилось за {elapsed:.0f} с:"))
            self.stdout.write(result.error)
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
