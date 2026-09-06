"""Проба генеративной модели на своём треке — до того, как встраивать её в сервис.

Скрипт нарочно отдельный: сначала слушаем результат, потом решаем, стоит ли
он GPU и потери дорожек с табами. Генеративные модели отдают готовый микс —
разложить его обратно на инструменты нельзя, а значит нельзя и построить
табулатуру.

ВАЖНО про окружение. audiocraft тянет spacy старой версии, который не
собирается на Python 3.12 и 3.13 — нужен отдельный интерпретатор 3.10 или
3.11. И нужна видеокарта NVIDIA: без неё 30 секунд считаются десятки минут.
То есть запускать это на обычном рабочем ноутбуке смысла нет — только на
арендованной машине с GPU.

Подготовка на арендованном сервере (Linux, Python 3.11):

    python3.11 -m venv ~/mg && source ~/mg/bin/activate
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchaudio
    pip install audiocraft
    python scripts/try_musicgen.py we_angel.mp3 \
        --style "aggressive nu metal, downtuned 7-string guitars, heavy drums" --seconds 30
"""
from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Проба MusicGen на своём треке")
    parser.add_argument("track", help="исходный аудиофайл")
    parser.add_argument("--style", default="aggressive nu metal, downtuned guitars, heavy drums",
                        help="текстовое описание желаемого звучания (модель понимает английский)")
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="сколько секунд генерировать (модель умеет примерно до 30)")
    parser.add_argument("--model", default="facebook/musicgen-melody",
                        help="musicgen-melody следует мелодии исходника; musicgen-small — нет")
    parser.add_argument("--out", default="musicgen_preview.wav")
    args = parser.parse_args()

    if sys.version_info >= (3, 12):
        print(f"Python {sys.version_info.major}.{sys.version_info.minor}: audiocraft на нём не "
              "собирается — его зависимость spacy требует Python 3.10 или 3.11.\n"
              "Сделайте отдельное окружение на 3.11, см. шапку файла.")
        return 1

    try:
        import torch
        import torchaudio
        from audiocraft.models import MusicGen
    except ImportError as exc:
        print(f"Не хватает пакета: {exc}\nПоставьте torch, torchaudio и audiocraft — "
              "см. шапку файла.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ВНИМАНИЕ: видеокарта не найдена. На процессоре 30 секунд "
              "генерации займут десятки минут — это проба ради пробы.")

    print(f"Гружу {args.model} на {device}…")
    started = time.time()
    model = MusicGen.get_pretrained(args.model, device=device)
    model.set_generation_params(duration=args.seconds)
    print(f"Модель загружена за {time.time() - started:.0f} с")

    melody, sr = torchaudio.load(args.track)
    melody = melody[:, : int(sr * args.seconds)]

    print(f"Генерирую {args.seconds:.0f} с в стиле «{args.style}»…")
    started = time.time()
    if "melody" in args.model:
        # мелодия исходника задаёт гармонию, текст — характер звучания
        output = model.generate_with_chroma([args.style], melody[None], sr)
    else:
        output = model.generate([args.style])
    elapsed = time.time() - started
    print(f"Готово за {elapsed:.0f} с — это {elapsed / args.seconds:.1f}× от длительности")

    torchaudio.save(args.out, output[0].cpu(), model.sample_rate)
    print(f"Результат: {args.out}")
    print("\nСлушать так: похоже ли это на песню в нужном жанре и стоит ли "
          "потери дорожек и табов, которые генеративная модель не отдаёт.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
