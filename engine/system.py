"""Сведения о машине, на которой считается рендер.

Нужны ровно для одного решения: хватит ли памяти на Demucs. Когда её не
хватает, процесс не бросает исключение — операционная система убивает его
целиком, и пользователь видит зависший прогресс, а не ошибку. Поэтому
проверяем заранее.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def available_memory_mb() -> float | None:
    """Свободная оперативная память, МБ. None — если определить не удалось."""
    try:
        if os.name == "nt":
            return _windows_available_mb()
        return _linux_available_mb()
    except Exception:  # noqa: BLE001 — не знать объём памяти не страшно
        logger.debug("Не удалось определить свободную память", exc_info=True)
        return None


def _windows_available_mb() -> float:
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 1048576


def _linux_available_mb() -> float | None:
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024
    return None


def demucs_needs_mb(duration_s: float) -> float:
    """Оценка потребности Demucs на CPU.

    Число подобрано с запасом и намеренно грубое: точную величину знает
    только сама машина, а ошибка в меньшую сторону стоит убитого процесса.
    Настраивается через VIBETRACK_DEMUCS_MB_BASE и VIBETRACK_DEMUCS_MB_PER_MIN.
    """
    base = float(os.getenv("VIBETRACK_DEMUCS_MB_BASE", "1200"))
    per_minute = float(os.getenv("VIBETRACK_DEMUCS_MB_PER_MIN", "450"))
    return base + per_minute * (duration_s / 60.0)


def enough_memory_for_demucs(duration_s: float) -> tuple[bool, str]:
    """(хватит ли, объяснение). Неизвестный объём памяти считаем достаточным."""
    if os.getenv("VIBETRACK_SKIP_MEMORY_CHECK", "0").strip().lower() in ("1", "true", "yes", "on"):
        return True, ""
    available = available_memory_mb()
    if available is None:
        return True, ""
    needed = demucs_needs_mb(duration_s)
    if available >= needed:
        return True, ""
    return False, (f"свободно {available / 1024:.1f} ГБ, "
                   f"а Demucs на трек {duration_s / 60:.1f} мин нужно около "
                   f"{needed / 1024:.1f} ГБ")
