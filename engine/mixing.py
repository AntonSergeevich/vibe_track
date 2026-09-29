"""Сведение и мастеринг: из дорожек — в готовый трек."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import SR
from .audio_io import Audio, pad_to
from .dsp import (compressor, db_to_lin, highpass, limiter, lowpass, normalize_loudness,
                  normalize_peak, peaking_eq, rms_db, sidechain_duck, stereo_width)

# Стартовые уровни шины (дБ). Классический ню-метал: гитары широко,
# бас в центре и плотно, барабаны бьют, вокал впереди.
DEFAULT_GAINS = {
    "guitar_rhythm": 0.0,
    "guitar_rhythm_r": 0.0,
    "guitar_lead": -7.0,
    "bass": -3.0,
    "drums": -4.0,
    "percussion": -16.0,
    "turntables": -17.0,
    "synth_pad": -20.0,
    "synth_stab": -16.0,
    "vocals": -1.0,
    "vocals_female": -3.0,
    "vocals_harmony": -12.0,
    "instrumental": 0.0,
    "source": -20.0,
}


def build_gains(spec=None) -> dict[str, float]:
    """Баланс микса + пользовательские трим-регуляторы из спецификации."""
    from .arrangement import INSTRUMENT_CATALOG

    gains = dict(DEFAULT_GAINS)
    if spec is None:
        return gains
    for item in getattr(spec, "instruments", []):
        catalog_default = INSTRUMENT_CATALOG.get(item.id, {}).get("gain_db", item.gain_db)
        trim = item.gain_db - catalog_default          # 0, пока пользователь не крутил
        gains[item.id] = gains.get(item.id, -8.0) + trim
    return gains


@dataclass
class MixSettings:
    gains_db: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GAINS))
    duck_instruments_db: float = -3.5      # приглушение инструментала под вокал
    bass_guitar_duck_db: float = -2.0      # бас пропускает бочку вперёд
    master_loudness_db: float = -10.0      # метал сводят громко
    ceiling_db: float = -0.8
    glue_ratio: float = 2.2
    width: float = 1.2


def mixdown(stems: dict[str, Audio], settings: MixSettings | None = None,
            vocal_keys: tuple[str, ...] = ("vocals", "vocals_female"),
            sr: int = SR, consume: bool = False) -> tuple[Audio, dict[str, Audio]]:
    """Возвращает (мастер, обработанные дорожки).

    consume=True освобождает исходные дорожки по мере обработки: иначе в
    памяти одновременно лежат две копии всего микса. На трёхминутном треке
    это разница в сотни мегабайт, то есть разница в тарифе сервера.
    """
    settings = settings or MixSettings()
    if not stems:
        return Audio(np.zeros((2, 1), dtype=np.float32), sr), {}

    n = max(a.n_samples for a in stems.values())
    processed: dict[str, Audio] = {}
    for name in list(stems):
        audio = stems.pop(name) if consume else stems[name]
        data = pad_to(audio.stereo(), n).data * db_to_lin(settings.gains_db.get(name, -8.0))
        del audio
        data = _bus_eq(name, data, sr)
        if float(np.max(np.abs(data))) > db_to_lin(-1.0):
            data = limiter(data, ceiling_db=-1.0, sr=sr)   # дорожки отдаём без клиппинга
        processed[name] = Audio(data, sr)

    vocal_bus = np.zeros((2, n), dtype=np.float32)
    for key in vocal_keys:
        if key in processed:
            vocal_bus += processed[key].data

    # бас уступает бочке, инструментал — вокалу
    if "bass" in processed and "drums" in processed:
        processed["bass"] = Audio(
            sidechain_duck(processed["bass"].data, processed["drums"].data, sr,
                           amount_db=settings.bass_guitar_duck_db, attack_ms=4, release_ms=90), sr)
    if np.any(vocal_bus):
        for name in list(processed):
            if name in vocal_keys:
                continue
            processed[name] = Audio(
                sidechain_duck(processed[name].data, vocal_bus, sr,
                               amount_db=settings.duck_instruments_db,
                               attack_ms=12, release_ms=220), sr)

    master = np.zeros((2, n), dtype=np.float32)
    for audio in processed.values():
        master += audio.data

    master = _master_chain(master, settings, sr)
    return Audio(master, sr), processed


def _bus_eq(name: str, data: np.ndarray, sr: int) -> np.ndarray:
    """Разводим инструменты по частотам, чтобы микс не превращался в кашу."""
    if name.startswith("guitar"):
        data = highpass(data, 95, sr)
        data = peaking_eq(data, 250, -2.0, q=1.0, sr=sr)     # место для баса
        data = peaking_eq(data, 3000, 2.0, q=1.0, sr=sr)
    elif name == "bass":
        data = highpass(data, 35, sr)
        data = lowpass(data, 7000, sr)
        data = peaking_eq(data, 80, 2.5, q=0.8, sr=sr)
        data = peaking_eq(data, 800, 1.5, q=1.2, sr=sr)      # чтобы бас было слышно
    elif name == "drums":
        data = peaking_eq(data, 60, 3.0, q=0.9, sr=sr)
        data = peaking_eq(data, 400, -2.5, q=1.0, sr=sr)
        data = peaking_eq(data, 8000, 2.5, q=0.8, sr=sr)
    elif name.startswith("vocals"):
        data = highpass(data, 90, sr)
        data = peaking_eq(data, 3200, 2.0, q=1.0, sr=sr)
    elif name == "source":
        data = lowpass(highpass(data, 200, sr), 6000, sr)    # исходник только как «подложка»
    return data


def _master_chain(master: np.ndarray, settings: MixSettings, sr: int) -> np.ndarray:
    master = stereo_width(master, settings.width)
    master = compressor(master, threshold_db=-14, ratio=settings.glue_ratio,
                        attack_ms=18, release_ms=180, sr=sr)
    master = peaking_eq(master, 90, 1.5, q=0.8, sr=sr)
    master = peaking_eq(master, 450, -1.5, q=1.0, sr=sr)
    master = peaking_eq(master, 6500, 2.0, q=0.7, sr=sr)
    master = normalize_loudness(master, settings.master_loudness_db, sr)
    master = limiter(master, ceiling_db=settings.ceiling_db, sr=sr)
    return normalize_peak(master, settings.ceiling_db)


def stem_report(stems: dict[str, Audio]) -> dict[str, dict]:
    return {name: {"peak_db": round(20 * np.log10(max(a.peak(), 1e-9)), 2),
                   "rms_db": round(rms_db(a.data), 2),
                   "duration": round(a.duration, 2)}
            for name, a in stems.items()}
