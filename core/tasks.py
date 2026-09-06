# core/tasks.py
"""Celery-задачи: мост между Django-моделями и аудиодвижком.

Движок ничего не знает про Django — он получает пути к файлам и
возвращает результат, а задачи раскладывают его по моделям.
"""
from __future__ import annotations

import logging
import os
import threading
import traceback

from celery import shared_task
from django.conf import settings
from django.db import connections

from engine.render import RenderOptions, add_vocal_take, transform

from . import billing
from .models import AudioFile, RenderJob, Score, Stem, VocalTake

logger = logging.getLogger(__name__)

def enqueue(task, *args):
    """Ставит задачу в очередь — или запускает в фоновом потоке.

    В eager-режиме Celery выполняет задачу прямо внутри HTTP-запроса: ответ
    не возвращается, пока рендер не закончится, поэтому браузеру нечего
    опрашивать и полоса прогресса стоит на месте. Поток решает это без
    Redis — ровно для локальной разработки.
    """
    if getattr(settings, "VIBETRACK_INLINE_WORKER", False):
        threading.Thread(target=_run_inline, args=(task, *args), daemon=True).start()
        return None
    return task.delay(*args)


def _run_inline(task, *args) -> None:
    try:
        task(*args)
    except Exception:  # noqa: BLE001 — поток не должен уронить процесс
        logger.exception("Фоновая задача %s упала", getattr(task, "name", task))
    finally:
        connections.close_all()      # иначе соединение потока останется висеть


STEM_LABELS = {
    "guitar_rhythm": "Ритм-гитара (лево)",
    "guitar_rhythm_r": "Ритм-гитара (право)",
    "guitar_lead": "Соло-гитара",
    "bass": "Бас",
    "drums": "Барабаны",
    "percussion": "Перкуссия",
    "turntables": "Скретчи",
    "synth_pad": "Пэд",
    "synth_stab": "Синт-стэбы",
    "vocals": "Вокал (муж.)",
    "vocals_female": "Вокал (жен.)",
    "vocals_harmony": "Гармония",
    "source": "Исходник (подложка)",
}


def media_path(*parts: str) -> str:
    return os.path.join(str(settings.MEDIA_ROOT), *parts)


def relative_to_media(path: str) -> str:
    return os.path.relpath(path, str(settings.MEDIA_ROOT)).replace(os.sep, "/")


def _engine_options(job: RenderJob) -> RenderOptions:
    cfg = settings.VIBETRACK
    options = job.options or {}
    return RenderOptions(
        separate_source=options.get("separate_source", True),
        keep_original_vocals=options.get("keep_original_vocals", True),
        transcribe_lyrics=options.get("transcribe_lyrics", True),
        manual_lyrics=options.get("manual_lyrics", "") or "",
        blend_source_db=options.get("blend_source_db"),
        export_format=options.get("export_format", cfg["EXPORT_FORMAT"]),
        master_loudness_db=options.get("master_loudness_db", cfg["MASTER_LOUDNESS_DB"]),
        demucs_model=cfg["DEMUCS_MODEL"],
        separation_backend=options.get("separation_backend", cfg["SEPARATION_BACKEND"]),
        max_duration=cfg["MAX_DURATION"],
        use_llm=options.get("use_llm", True),
        samples_dir=str(getattr(settings, "VIBETRACK_SAMPLES_DIR", "")),
        generate_cover=options.get("generate_cover", False),
        lyrics_language=options.get("lyrics_language", "") or "",
    )


@shared_task(bind=True)
def render_track(self, job_id: int) -> dict:
    """Главная задача: исходный трек + описание → ню-метал версия."""
    job = RenderJob.objects.select_related("track").get(pk=job_id)
    job.status = RenderJob.STATUS_RUNNING
    job.stage = "load"
    job.progress = 1
    job.celery_task_id = getattr(self.request, "id", "") or ""
    job.error = ""
    job.save(update_fields=["status", "stage", "progress", "celery_task_id", "error", "updated_at"])

    source = job.track.source_file
    if source is None or not source.file:
        job.status = RenderJob.STATUS_ERROR
        job.error = "У трека нет исходного файла."
        job.save(update_fields=["status", "error", "updated_at"])
        billing.refund(job.track.project.owner, job, "нет исходного файла")
        return {"status": "error", "detail": job.error}

    out_dir = media_path("renders", f"job_{job.pk}")

    def progress(stage: str, pct: int) -> None:
        RenderJob.objects.filter(pk=job.pk).update(stage=stage, progress=pct)

    try:
        result = transform(
            source_path=source.file.path,
            prompt=job.prompt or "",
            overrides=job.overrides or {},
            out_dir=out_dir,
            options=_engine_options(job),
            progress=progress,
        )
    except Exception as exc:  # noqa: BLE001 — пользователю нужен текст ошибки
        logger.exception("Рендер #%s упал", job.pk)
        job.status = RenderJob.STATUS_ERROR
        job.stage = "error"
        job.error = f"{exc}\n{traceback.format_exc(limit=4)}"
        job.save(update_fields=["status", "stage", "error", "updated_at"])
        # трек не получился — списание возвращаем, это наша ошибка, а не его
        billing.refund(job.track.project.owner, job, f"рендер упал: {exc}"[:200])
        return {"status": "error", "detail": str(exc)}

    _store_result(job, result)
    return {"status": "done", "job": job.pk, "stems": len(result.stem_paths)}


def _store_result(job: RenderJob, result) -> None:
    job.master.name = relative_to_media(result.master_path)
    job.spec = result.spec
    job.warnings = result.warnings
    job.result = {
        "analysis": result.analysis,
        "report": result.report,
        "timings": result.timings,
        "cover_cost_usd": result.cover_cost_usd,
        "arrangement_bars": len(result.arrangement.get("bars", [])),
    }
    job.status = RenderJob.STATUS_DONE
    job.stage = "done"
    job.progress = 100
    job.save()

    job.track.analysis = result.analysis
    job.track.save(update_fields=["analysis"])

    job.stems.all().delete()
    if result.cover_path:
        Stem.objects.create(job=job, name="cover", label="Кавер от нейросети",
                            file=relative_to_media(result.cover_path), source="generated")
    for name, path in result.stem_paths.items():
        info = result.report.get(name, {})
        Stem.objects.create(
            job=job,
            name=name,
            label=STEM_LABELS.get(name, name.replace("source_", "Исходник: ")),
            file=relative_to_media(path),
            source="source" if name.startswith("source_") else "generated",
            peak_db=info.get("peak_db"),
            rms_db=info.get("rms_db"),
            duration=info.get("duration"),
        )

    Score.objects.update_or_create(
        job=job,
        defaults={
            "lyrics": result.lyrics,
            "lyric_sheet": result.lyric_sheet,
            "chords": result.chords,
            "chord_chart": result.chord_chart,
            "tabs": result.tabs,
            "key": result.analysis.get("key_name", ""),
            "tempo": result.analysis.get("tempo"),
            "tuning": (result.spec or {}).get("tuning", ""),
        },
    )

    AudioFile.objects.update_or_create(
        track=job.track,
        kind=AudioFile.KIND_MASTER,
        file=job.master.name,
        defaults={"status": "done", "duration": result.analysis.get("duration")},
    )


@shared_task(bind=True)
def process_vocal_take(self, take_id: int) -> dict:
    """Записанный голос: автообработка и вписывание в минусовку."""
    take = VocalTake.objects.select_related("track", "job").get(pk=take_id)
    take.status = RenderJob.STATUS_RUNNING
    take.error = ""
    take.save(update_fields=["status", "error"])

    instrumental = _instrumental_for(take)
    if instrumental is None:
        take.status = RenderJob.STATUS_ERROR
        take.error = "Не найдена минусовка: сначала сделайте рендер трека."
        take.save(update_fields=["status", "error"])
        return {"status": "error", "detail": take.error}

    out_dir = media_path("takes", f"take_{take.pk}")
    try:
        result = add_vocal_take(
            instrumental_path=instrumental,
            take_path=take.raw_file.path,
            out_dir=out_dir,
            style=take.style,
            gender=take.gender,
            target_gender=take.target_gender or None,
            autotune_strength=take.autotune,
            take_gain_db=take.gain_db,
            master_loudness_db=settings.VIBETRACK["MASTER_LOUDNESS_DB"],
            export_format=settings.VIBETRACK["EXPORT_FORMAT"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Обработка дубля #%s упала", take.pk)
        take.status = RenderJob.STATUS_ERROR
        take.error = str(exc)
        take.save(update_fields=["status", "error"])
        return {"status": "error", "detail": str(exc)}

    take.processed_file.name = relative_to_media(result["vocal_path"])
    take.mixed_file.name = relative_to_media(result["master_path"])
    take.latency_ms = result["latency_ms"]
    take.notes = result["notes"]
    take.status = RenderJob.STATUS_DONE
    take.save()
    return {"status": "done", "take": take.pk, "latency_ms": take.latency_ms}


def _instrumental_for(take: VocalTake) -> str | None:
    """Минусовка = мастер последнего рендера трека."""
    job = take.job or take.track.render_jobs.filter(status=RenderJob.STATUS_DONE).first()
    if job and job.master:
        return job.master.path
    source = take.track.source_file
    return source.file.path if source else None
