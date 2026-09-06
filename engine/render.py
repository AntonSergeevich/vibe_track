"""Оркестратор: исходный трек + описание → ню-метал версия.

Это верхнеуровневый вход движка, которым пользуется Celery-задача.
Каждый шаг возвращает данные, а не пишет в БД, — движок остаётся
независимым от Django и легко тестируется.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import SR
from .analysis import TrackAnalysis, analyze
from .arrangement import ArrangementSpec, parse_prompt
from .llm import refine as llm_refine
from .audio_io import Audio, load, resample, save
from .chords import ChordEvent, recognize, to_power_chords, transpose as transpose_chords
from .generation import (CoverRequest, generate_cover, is_configured as generation_configured,
                         stability_limit, style_prompt_for)
from . import models
from .mixing import MixSettings, build_gains, mixdown, stem_report
from .dsp import normalize_loudness
from .rendering import render_arrangement
from .sampler import SampleLibrary, load_library, library_dir
from .separation import separate
from .sequencer import Arrangement, sequence
from .tabs import chord_chart, render_tab, tab_for_all
from .transcription import Lyrics, lyric_sheet, lyrics_from_text, transcribe
from .dsp import formant_shift, pitch_shift
from .vocals import VocalTakeResult, harmony, process_take

logger = logging.getLogger(__name__)

STAGES = ("load", "analyze", "separate", "transcribe", "arrange", "render",
          "vocals", "mix", "score", "export")


@dataclass
class RenderOptions:
    separate_source: bool = True
    keep_original_vocals: bool = True     # использовать вокал исходника в новой версии
    transcribe_lyrics: bool = True
    manual_lyrics: str = ""
    blend_source_db: float | None = None  # подмешать исходник (None = не подмешивать)
    export_format: str = "wav"
    master_loudness_db: float = -10.0
    demucs_model: str = "htdemucs"
    separation_backend: str = "auto"      # auto | demucs | dsp
    max_duration: float = 480.0           # защита от гигантских файлов
    use_llm: bool = True                  # уточнять описание через LLM, если она настроена
    samples_dir: str = ""                 # папка с живыми сэмплами (пусто = из настроек)
    generate_cover: bool = False          # заказать кавер у внешней нейросети
    lyrics_language: str = ""             # пусто = определить автоматически


@dataclass
class RenderResult:
    out_dir: str
    master_path: str = ""
    cover_path: str = ""                  # кавер от внешней модели, если заказан
    cover_cost_usd: float = 0.0
    stem_paths: dict[str, str] = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    spec: dict = field(default_factory=dict)
    arrangement: dict = field(default_factory=dict)
    chords: list = field(default_factory=list)
    lyrics: dict = field(default_factory=dict)
    lyric_sheet: str = ""
    chord_chart: str = ""
    tabs: dict[str, str] = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "out_dir": self.out_dir, "master_path": self.master_path,
            "cover_path": self.cover_path, "cover_cost_usd": self.cover_cost_usd,
            "stem_paths": self.stem_paths, "analysis": self.analysis, "spec": self.spec,
            "arrangement": self.arrangement, "chords": self.chords, "lyrics": self.lyrics,
            "lyric_sheet": self.lyric_sheet, "chord_chart": self.chord_chart,
            "tabs": self.tabs, "report": self.report, "timings": self.timings,
            "warnings": self.warnings,
        }


class _Timer:
    def __init__(self):
        self.marks: dict[str, float] = {}
        self._t = time.time()

    def mark(self, name: str):
        now = time.time()
        self.marks[name] = round(now - self._t, 2)
        self._t = now


def transform(source_path: str, prompt: str = "", overrides: dict | None = None,
              out_dir: str = "render", options: RenderOptions | None = None,
              progress=None) -> RenderResult:
    """Полный конвейер превращения трека в ню-метал."""
    options = options or RenderOptions()
    os.makedirs(out_dir, exist_ok=True)
    timer = _Timer()
    warnings: list[str] = []

    def _progress(stage: str, pct: int):
        if progress:
            try:
                progress(stage, pct)
            except Exception:  # прогресс не должен ронять рендер
                logger.debug("progress callback failed", exc_info=True)

    ext = options.export_format
    stems_dir = os.path.join(out_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)
    stem_paths: dict[str, str] = {}

    _progress("load", 2)
    source = load(source_path, sr=SR)
    if source.duration > options.max_duration:
        source = Audio(source.data[:, : int(options.max_duration * SR)], source.sr)
        warnings.append(f"Трек обрезан до {options.max_duration / 60:.0f} мин.")
    timer.mark("load")

    _progress("analyze", 10)
    analysis = analyze(source)
    chords = recognize(analysis)
    timer.mark("analyze")

    _progress("separate", 25)
    source_stems: dict[str, Audio] = {}
    cover_input_path = ""
    if options.separate_source:
        backend = options.separation_backend
        result = separate(source, model=options.demucs_model,
                          prefer_backend=None if backend == "auto" else backend)
        source_stems = result.stems
        if result.backend != "demucs" and options.separation_backend != "dsp":
            reason = result.detail or "причина неизвестна"
            warnings.append(
                f"Дорожки разделены DSP-методом вместо Demucs ({reason}). "
                "Качество ниже: поставьте requirements-ml.txt и прогрейте веса "
                "командой `python manage.py models --preload`.")
        else:
            logger.info("Demucs %s: %s дорожек за %.1f с", result.model,
                        len(result.stems), result.seconds)
        del result
        if not models.keep_loaded():
            # Demucs своё отработал: дальше идут Whisper и синтез, и держать
            # рядом с ними ещё и torch-модель типичной машине не по силам
            models.release("demucs:")

        # Разобранные дорожки сразу уходят на диск: в памяти дальше нужен
        # только вокал (текст и его переработка), а остальные три — это ещё
        # четверть гигабайта на трёхминутном треке, лежащая мёртвым грузом
        # до самого экспорта.
        # Кавер лучше делать с минусовки, а не с полного микса: у модели нет
        # чужого вокала, который она всё равно перепоёт по-своему, а фильтр
        # авторских прав реже узнаёт песню без голоса.
        if options.generate_cover:
            cover_input_path = _build_cover_input(source_stems, stems_dir)

        for name in list(source_stems):
            stem_paths[f"source_{name}"] = save(
                os.path.join(stems_dir, f"source_{name}.{ext}"), source_stems[name])
            if name != "vocals":
                source_stems.pop(name)
    timer.mark("separate")

    _progress("transcribe", 40)
    lyrics = Lyrics(backend="none")
    if options.manual_lyrics.strip():
        lyrics = lyrics_from_text(options.manual_lyrics, analysis)
    elif options.transcribe_lyrics and "vocals" in source_stems:
        lyrics = transcribe(source_stems["vocals"], language=options.lyrics_language or None,
                            analysis=analysis)
        if lyrics.backend == "unavailable":
            warnings.append("Whisper не установлен — текст не расшифрован. "
                            "Можно вставить текст вручную, аккорды к нему подставятся.")
        if not models.keep_loaded():
            models.release("whisper:")
    timer.mark("transcribe")

    _progress("arrange", 50)
    spec = parse_prompt(prompt, analysis, overrides)
    if options.use_llm:
        spec = llm_refine(spec, prompt, analysis, user_overrides=overrides)
    riff_chords = to_power_chords(chords)
    if spec.transpose:
        riff_chords = transpose_chords(riff_chords, spec.transpose)
    arrangement = sequence(analysis, riff_chords, spec)
    timer.mark("arrange")

    _progress("render", 60)
    library = load_library(options.samples_dir or library_dir(), SR)
    if library.notes:
        for note in library.notes:
            logger.info("Сэмплы: %s", note)
    if not (library.has_drums or library.has_guitar):
        warnings.append("Живых сэмплов не найдено — инструменты синтезируются. "
                        "Положите барабанный луп и гитару в папку samples/.")
    stems = render_arrangement(arrangement, SR, library=library)
    timer.mark("render")

    _progress("vocals", 78)
    if options.keep_original_vocals and "vocals" in source_stems:
        vocal_stems = _rework_source_vocals(source_stems["vocals"], spec, analysis)
        stems.update(vocal_stems)
    if options.blend_source_db is not None:
        stems["source"] = source
    timer.mark("vocals")

    _progress("mix", 86)
    settings = MixSettings(gains_db=build_gains(spec),
                           master_loudness_db=options.master_loudness_db)
    if options.blend_source_db is not None:
        settings.gains_db["source"] = options.blend_source_db
    # consume=True: дорожки освобождаются по мере сведения, дальше они не нужны
    master, processed = mixdown(stems, settings, sr=SR, consume=True)
    # отчёт снимаем сразу: на экспорте дорожки уходят на диск и освобождаются
    report = stem_report(processed)
    timer.mark("mix")

    _progress("score", 92)
    tabs = tab_for_all(arrangement)
    sheet = lyric_sheet(lyrics, chords) if lyrics.lines else ""
    timer.mark("score")

    _progress("export", 95)
    # исходные дорожки уже на диске — сохраняем только то, что насчитали сами
    for name in list(processed):
        stem_paths[name] = save(os.path.join(stems_dir, f"{name}.{ext}"),
                                processed.pop(name))
    master_path = save(os.path.join(out_dir, f"master.{ext}"), master)

    cover_path, cover_cost = "", 0.0
    if options.generate_cover:
        if not generation_configured():
            warnings.append("Кавер не заказан: внешняя нейросеть не настроена "
                            "(VIBETRACK_MUSIC_PROVIDER и ключ).")
        else:
            _progress("cover", 97)
            limit = stability_limit()
            if analysis.duration > limit:
                # модель обрежет сама и молча — пусть человек узнает от нас
                warnings.append(
                    f"Кавер сделан на первые {int(limit) // 60}:{int(limit) % 60:02d} — "
                    "модель не принимает треки длиннее. Разбор при этом по всему треку.")
            cover = generate_cover(CoverRequest(
                source_path=cover_input_path or source_path,
                style_prompt=style_prompt_for(spec.genre, " ".join(spec.notes)),
                genre=spec.genre, duration=analysis.duration))
            if cover.ok:
                cover_path = save(os.path.join(out_dir, f"cover.{ext}"), cover.audio)
                cover_cost = cover.cost_usd
                logger.info("Кавер от %s за %.0f с", cover.provider, cover.seconds)
                # Свой голос поверх нового аккомпанемента — то, ради чего
                # человек и приносил трек: мелодия и слова остаются его.
                if options.keep_original_vocals and "vocals" in source_stems:
                    voiced = _cover_with_vocals(cover.audio, source_stems["vocals"],
                                                options.master_loudness_db)
                    if voiced is not None:
                        stem_paths["cover_vocals"] = save(
                            os.path.join(out_dir, f"cover_with_vocals.{ext}"), voiced)
                        warnings.append(
                            "Кавер сведён и с вашим голосом — файл «Кавер с вашим "
                            "вокалом». Если голос разъезжается с музыкой, "
                            "уменьшите VIBETRACK_STABILITY_STRENGTH.")
            else:
                warnings.append(f"Кавер не получился: {cover.error}")
            if cover_input_path and os.path.exists(cover_input_path):
                os.remove(cover_input_path)

    result = RenderResult(
        out_dir=out_dir, master_path=master_path, cover_path=cover_path,
        cover_cost_usd=cover_cost, stem_paths=stem_paths,
        analysis=analysis.to_dict(), spec=spec.to_dict(), arrangement=arrangement.to_dict(),
        chords=[c.to_dict() for c in chords], lyrics=lyrics.to_dict(), lyric_sheet=sheet,
        chord_chart=chord_chart(chords), tabs=tabs,
        report=report, warnings=warnings,
    )
    timer.mark("export")
    result.timings = timer.marks
    _write_score_files(out_dir, result)
    _progress("done", 100)
    return result


def _build_cover_input(source_stems: dict, stems_dir: str) -> str:
    """Минусовка для внешней модели: всё, кроме вокала.

    Пустая строка — если разделения нет и брать нечего; тогда уйдёт полный
    микс, как раньше.
    """
    parts = [audio.data for name, audio in source_stems.items() if name != "vocals"]
    if not parts:
        return ""
    width = max(part.shape[-1] for part in parts)
    mix = np.zeros((2, width), dtype=np.float32)
    for part in parts:
        stereo = part if part.shape[0] == 2 else np.repeat(part[:1], 2, axis=0)
        mix[:, : stereo.shape[-1]] += stereo
    mix = normalize_loudness(mix, -16.0, SR)
    return save(os.path.join(stems_dir, "_cover_input.wav"), Audio(mix, SR))


def _cover_with_vocals(cover: Audio, vocals: Audio, loudness_db: float) -> "Audio | None":
    """Сводит сгенерированный аккомпанемент с исходным голосом.

    Кавер обычно короче трека (модель принимает ограниченный кусок), поэтому
    равняемся по нему: лучше короткий кусок с узнаваемым голосом, чем
    длинный без него.
    """
    voice = vocals.stereo()
    if voice.sr != cover.sr:
        voice = resample(voice, cover.sr)
    n = min(cover.n_samples, voice.n_samples)
    if n < cover.sr:                       # меньше секунды — сводить нечего
        return None
    mixed = cover.data[:, :n] * 0.85 + voice.data[:, :n] * 1.1
    return Audio(normalize_loudness(mixed, loudness_db, cover.sr), cover.sr)


def _rework_source_vocals(vocals: Audio, spec: ArrangementSpec,
                          analysis: TrackAnalysis) -> dict[str, Audio]:
    """Вокал исходника вписываем в новую аранжировку согласно спецификации."""
    out: dict[str, Audio] = {}
    v = spec.vocals
    if spec.transpose:
        # вокал едет за тональностью, иначе он разойдётся с гитарами;
        # форманты подтягиваем обратно, чтобы голос не стал мультяшным
        shifted = pitch_shift(vocals.data, spec.transpose, vocals.sr)
        shifted = formant_shift(shifted, -spec.transpose * 0.35, vocals.sr)
        vocals = Audio(shifted, vocals.sr)
    if v.male:
        res = process_take(vocals, style=v.male_style, analysis=analysis,
                           autotune_strength=1.0 if v.autotune else None)
        out["vocals"] = res.audio
    if v.female:
        # женскую партию делаем перекраской исходного голоса
        res = process_take(vocals, style=v.female_style, analysis=analysis,
                           gender="male", target_gender="female",
                           autotune_strength=1.0 if v.autotune else None)
        out["vocals_female"] = res.audio
    if v.harmony and out:
        base = out.get("vocals") or out.get("vocals_female")
        out["vocals_harmony"] = harmony(base, semitones=7.0, level_db=-11.0)
    return out


def add_vocal_take(instrumental_path: str, take_path: str, out_dir: str,
                   style: str = "rap", gender: str = "male",
                   target_gender: str | None = None, autotune_strength: float | None = None,
                   take_gain_db: float = -3.0, master_loudness_db: float = -10.0,
                   export_format: str = "wav") -> dict:
    """Вписывает записанный голос в готовую минусовку с автообработкой."""
    os.makedirs(out_dir, exist_ok=True)
    instrumental = load(instrumental_path, sr=SR)
    take = load(take_path, sr=SR)
    analysis = analyze(instrumental)

    processed: VocalTakeResult = process_take(
        take, style=style, analysis=analysis, reference=instrumental,
        gender=gender, target_gender=target_gender, autotune_strength=autotune_strength)

    settings = MixSettings(master_loudness_db=master_loudness_db)
    settings.gains_db["instrumental"] = 0.0
    settings.gains_db["vocals"] = take_gain_db
    master, _ = mixdown({"instrumental": instrumental, "vocals": processed.audio},
                        settings, sr=SR)

    vocal_path = save(os.path.join(out_dir, f"vocal_take.{export_format}"), processed.audio)
    master_path = save(os.path.join(out_dir, f"master_with_vocal.{export_format}"), master)
    return {
        "vocal_path": vocal_path,
        "master_path": master_path,
        "latency_ms": processed.latency_ms,
        "peak_db": processed.peak_db,
        "notes": processed.notes,
        "analysis": analysis.to_dict(),
    }


def _write_score_files(out_dir: str, result: RenderResult) -> None:
    """Кладём рядом с аудио человекочитаемую «партитуру»."""
    score_dir = os.path.join(out_dir, "score")
    os.makedirs(score_dir, exist_ok=True)
    if result.lyric_sheet:
        _write(os.path.join(score_dir, "lyrics_chords.txt"), result.lyric_sheet)
    if result.chord_chart:
        _write(os.path.join(score_dir, "chords.txt"), result.chord_chart)
    for name, tab in result.tabs.items():
        _write(os.path.join(score_dir, f"tab_{name}.txt"), tab)
    _write(os.path.join(score_dir, "session.json"),
           json.dumps({"analysis": result.analysis, "spec": result.spec,
                       "chords": result.chords, "report": result.report,
                       "timings": result.timings, "warnings": result.warnings},
                      ensure_ascii=False, indent=2))


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
