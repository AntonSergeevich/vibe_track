"""Ленивая загрузка и кэширование тяжёлых моделей.

Ключевой момент для продакшена: воркер Celery живёт долго, а модель
Demucs или Whisper грузится десятки секунд. Поэтому модели создаются один
раз на процесс и переиспользуются между задачами. Загрузка защищена
блокировкой — при параллельных задачах модель не будет собрана дважды.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE: dict[str, object] = {}
_LOAD_TIMES: dict[str, float] = {}


class ModelUnavailable(RuntimeError):
    """Модель не установлена или не смогла загрузиться."""


@dataclass
class ModelInfo:
    name: str
    available: bool
    loaded: bool
    detail: str = ""
    load_seconds: float | None = None


def cache_dir() -> str:
    """Куда складывать веса. По умолчанию — рядом с медиа, чтобы не качать заново."""
    path = os.getenv("VIBETRACK_MODEL_CACHE", os.path.expanduser("~/.cache/vibetrack-models"))
    os.makedirs(path, exist_ok=True)
    return path


def configure_torch_threads() -> int:
    """Ограничивает число потоков torch.

    Без этого torch на CPU занимает все ядра, и параллельные задачи Celery
    начинают драться за них, замедляя друг друга.
    """
    threads = int(os.getenv("VIBETRACK_TORCH_THREADS", "0"))
    if threads <= 0:
        threads = max(1, (os.cpu_count() or 2))
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:  # torch может быть не установлен — это норма
        pass
    return threads


def _cached(key: str, builder):
    if key in _CACHE:
        return _CACHE[key]
    with _LOCK:
        if key in _CACHE:  # мог загрузить другой поток, пока мы ждали
            return _CACHE[key]
        started = time.time()
        _CACHE[key] = builder()
        _LOAD_TIMES[key] = round(time.time() - started, 2)
        logger.info("Модель %s загружена за %.1f с", key, _LOAD_TIMES[key])
    return _CACHE[key]


# ------------------------------------------------------------------ Demucs
def demucs_available() -> bool:
    try:
        import demucs.api  # noqa: F401

        return True
    except Exception:
        return False


def get_demucs(model_name: str = "htdemucs", device: str | None = None,
               segment: float | None = None):
    """Возвращает готовый demucs.api.Separator (или бросает ModelUnavailable)."""
    device = device or os.getenv("VIBETRACK_DEMUCS_DEVICE", "cpu")
    segment = segment if segment is not None else _float_env("VIBETRACK_DEMUCS_SEGMENT")
    shifts = int(os.getenv("VIBETRACK_DEMUCS_SHIFTS", "0"))
    key = f"demucs:{model_name}:{device}:{segment}:{shifts}"

    def _build():
        try:
            from demucs.api import Separator
        except Exception as exc:
            raise ModelUnavailable(
                "Demucs не установлен. Поставьте: pip install -r requirements-ml.txt"
            ) from exc
        configure_torch_threads()
        os.environ.setdefault("TORCH_HOME", cache_dir())
        kwargs = {"model": model_name, "device": device, "progress": False}
        if shifts:
            kwargs["shifts"] = shifts          # усреднение по сдвигам: точнее, но кратно дольше
        if segment:
            kwargs["segment"] = segment        # короче куски — меньше пик памяти на CPU
        try:
            return Separator(**kwargs)
        except Exception as exc:
            raise ModelUnavailable(f"Не удалось загрузить Demucs «{model_name}»: {exc}") from exc

    return _cached(key, _build)


# ----------------------------------------------------------------- Whisper
def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        return False


def get_whisper(model_size: str | None = None, device: str | None = None,
                compute_type: str | None = None):
    """Возвращает готовую faster_whisper.WhisperModel."""
    model_size = model_size or os.getenv("VIBETRACK_WHISPER_MODEL", "small")
    device = device or os.getenv("VIBETRACK_WHISPER_DEVICE", "cpu")
    compute_type = compute_type or os.getenv(
        "VIBETRACK_WHISPER_COMPUTE", "int8" if device == "cpu" else "float16")
    key = f"whisper:{model_size}:{device}:{compute_type}"

    def _build():
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise ModelUnavailable(
                "faster-whisper не установлен. Поставьте: pip install -r requirements-ml.txt"
            ) from exc
        try:
            return WhisperModel(
                model_size, device=device, compute_type=compute_type,
                download_root=cache_dir(),
                cpu_threads=configure_torch_threads() if device == "cpu" else 0,
            )
        except Exception as exc:
            raise ModelUnavailable(f"Не удалось загрузить Whisper «{model_size}»: {exc}") from exc

    return _cached(key, _build)


# -------------------------------------------------------------------- LLM
def llm_available() -> bool:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401

        return True
    except Exception:
        return False


def get_anthropic_client():
    def _build():
        try:
            import anthropic
        except Exception as exc:
            raise ModelUnavailable("Пакет anthropic не установлен.") from exc
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ModelUnavailable("Не задан ANTHROPIC_API_KEY.")
        return anthropic.Anthropic()

    return _cached("anthropic:client", _build)


# ------------------------------------------------------------------ статус
def status() -> dict[str, ModelInfo]:
    """Что доступно прямо сейчас — для /api/capabilities и диагностики."""
    return {
        "demucs": ModelInfo(
            name=os.getenv("VIBETRACK_DEMUCS_MODEL", "htdemucs"),
            available=demucs_available(),
            loaded=any(k.startswith("demucs:") for k in _CACHE),
            detail="разделение на дорожки",
            load_seconds=next((v for k, v in _LOAD_TIMES.items() if k.startswith("demucs:")), None),
        ),
        "whisper": ModelInfo(
            name=os.getenv("VIBETRACK_WHISPER_MODEL", "small"),
            available=whisper_available(),
            loaded=any(k.startswith("whisper:") for k in _CACHE),
            detail="расшифровка текста",
            load_seconds=next((v for k, v in _LOAD_TIMES.items() if k.startswith("whisper:")), None),
        ),
        "llm": ModelInfo(
            name=os.getenv("VIBETRACK_LLM_MODEL", "claude-opus-5"),
            available=llm_available(),
            loaded="anthropic:client" in _CACHE,
            detail="разбор описания аранжировки",
        ),
    }


def preload() -> dict[str, str]:
    """Прогрев моделей при старте воркера — чтобы первый пользователь не ждал."""
    report = {}
    for name, getter in (("demucs", get_demucs), ("whisper", get_whisper)):
        try:
            getter()
            report[name] = f"загружена за {_LOAD_TIMES.get(next(k for k in _LOAD_TIMES if k.startswith(name)), 0)} с"
        except ModelUnavailable as exc:
            report[name] = f"недоступна: {exc}"
        except Exception as exc:  # noqa: BLE001
            report[name] = f"ошибка: {exc}"
    return report


def clear() -> None:
    """Выгрузить модели (тесты, освобождение памяти)."""
    with _LOCK:
        _CACHE.clear()
        _LOAD_TIMES.clear()


def _float_env(name: str) -> float | None:
    raw = os.getenv(name)
    try:
        return float(raw) if raw else None
    except ValueError:
        return None
