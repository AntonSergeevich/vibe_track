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
        parser.add_argument("--sweep", default="",
                            help="сравнить несколько значений силы за прогон, "
                                 "например 0.35,0.5,0.65 — каждое стоит $0.20")
        parser.add_argument("--instrumental", action="store_true",
                            help="отправить минусовку без вокала: фильтр "
                                 "авторских прав срабатывает реже")
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
        limit = generation.stability_limit()
        seconds = min(options["seconds"] or source.duration, limit)
        os.environ["VIBETRACK_STABILITY_MAX_SECONDS"] = str(seconds)

        self.stdout.write(f"Исходник: {source.duration:.0f} с, отправляем "
                          f"{seconds:.0f} с")
        self.stdout.write(f"Сила переделки: "
                          f"{os.getenv('VIBETRACK_STABILITY_STRENGTH', '0.6')}")
        self.stdout.write(f"Промпт: {options['prompt']}")
        self.stdout.write("\nЗапрос пошёл…")

        track_path = options["track"]
        if options["instrumental"]:
            track_path = self._instrumental(track_path)
            self.stdout.write(f"Отправляем минусовку: {track_path}")

        values = [v.strip() for v in options["sweep"].split(",") if v.strip()]
        if values:
            self.stdout.write(self.style.WARNING(
                f"\nСравнение {len(values)} вариантов — это ${0.2 * len(values):.2f} "
                f"({20 * len(values)} кредитов)."))
            return self._sweep(track_path, options, values, seconds)

        started = time.time()
        result = generation.generate_cover(generation.CoverRequest(
            source_path=track_path, style_prompt=options["prompt"],
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

    def _sweep(self, track_path, options, values, seconds):
        """Гоняет один и тот же трек с разной силой переделки.

        Слушать варианты подряд — единственный способ найти своё значение:
        на слух разница между 0.35 и 0.5 больше, чем кажется по числам.
        """
        import time

        for value in values:
            os.environ["VIBETRACK_STABILITY_STRENGTH"] = value
            self.stdout.write(f"\nСила переделки {value} …")
            started = time.time()
            result = generation.generate_cover(generation.CoverRequest(
                source_path=track_path, style_prompt=options["prompt"],
                duration=seconds))
            if not result.ok:
                self.stdout.write(self.style.ERROR(f"  не вышло: {result.error}"))
                continue
            name = f"cover_{value.replace('.', '_')}.mp3"
            save(name, result.audio)
            self.stdout.write(self.style.SUCCESS(
                f"  {name} — {result.audio.duration:.0f} с за {time.time() - started:.0f} с"))

        self.stdout.write("\nПослушайте файлы подряд и скажите, какой ближе. "
                          "Победившее значение впишем в .env как основное.")

    def _instrumental(self, track_path: str) -> str:
        """Минусовка исходника: у модели не будет чужого голоса."""
        import numpy as np

        from engine.audio_io import Audio
        from engine.separation import separate

        self.stdout.write("Убираю вокал (это займёт минуту)…")
        stems = separate(load(track_path)).stems
        parts = [a.data for name, a in stems.items() if name != "vocals"]
        width = max(p.shape[-1] for p in parts)
        mix = np.zeros((2, width), dtype=np.float32)
        for part in parts:
            mix[:, : part.shape[-1]] += part
        return save("cover_input.wav", Audio(mix, load(track_path).sr))
