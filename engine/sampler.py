"""Живые сэмплы вместо синтеза.

Пользователь редко приносит аккуратный набор «бочка.wav, рабочий.wav».
Обычно это луп барабанов и кусок гитары одним файлом — как раз такие два
файла и лежат в `samples/`. Поэтому здесь не «загрузчик сэмплов», а
разборщик: луп режется по атакам на отдельные удары, удары раскладываются
по ролям (бочка, рабочий, хэт, тарелка) по спектру, а гитарный кусок
превращается в ноту, которую можно переигрывать любой высотой.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from . import SR
from .analysis import onset_envelope
from .audio_io import Audio, load
from .dsp import normalize_peak

logger = logging.getLogger(__name__)

DRUM_ROLES = ("kick", "snare", "hat", "openhat", "crash", "tom_hi", "tom_lo")


@dataclass
class DrumKit:
    """Удары, разложенные по ролям. На роль может быть несколько вариантов."""

    hits: dict[str, list[np.ndarray]] = field(default_factory=dict)
    source: str = ""

    def get(self, role: str, index: int = 0) -> np.ndarray | None:
        variants = self.hits.get(role)
        if not variants:
            return None
        return variants[index % len(variants)]      # круговая смена — живее, чем один удар

    @property
    def roles(self) -> list[str]:
        return sorted(self.hits)


@dataclass
class GuitarSample:
    """Одна нота живой гитары, которую можно переиграть любой высотой."""

    audio: np.ndarray
    base_freq: float
    pre_amped: bool = True      # файл уже с перегрузом — второй раз не перегружаем
    source: str = ""

    def play(self, freq: float, dur: float, sr: int = SR, velocity: float = 1.0,
             palm_mute: bool = False) -> np.ndarray:
        """Переигрывает сэмпл нужной высотой и длительностью."""
        ratio = freq / max(self.base_freq, 1e-6)
        n_src = len(self.audio)
        n_out = max(int(dur * sr), 64)
        # изменение скорости воспроизведения: и высота, и характер атаки едут
        # вместе, как у настоящей струны, взятой на другом ладу
        idx = np.arange(n_out, dtype=np.float32) * ratio
        idx = np.clip(idx, 0, n_src - 1)
        out = np.interp(idx, np.arange(n_src, dtype=np.float32), self.audio).astype(np.float32)

        env = np.ones(n_out, dtype=np.float32)
        attack = min(int(0.002 * sr), n_out)
        env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
        if palm_mute:
            decay = min(int(0.09 * sr), n_out)
            env[:decay] *= np.linspace(1.0, 0.25, decay, dtype=np.float32)
            env[decay:] = 0.0
        else:
            release = min(int(0.05 * sr), n_out)
            env[-release:] *= np.linspace(1, 0, release, dtype=np.float32)
        return out * env * velocity


@dataclass
class SampleLibrary:
    kit: DrumKit | None = None
    guitar: GuitarSample | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_drums(self) -> bool:
        return self.kit is not None and bool(self.kit.hits)

    @property
    def has_guitar(self) -> bool:
        return self.guitar is not None


def library_dir() -> str:
    return os.getenv("VIBETRACK_SAMPLES_DIR", "samples")


def load_library(directory: str | None = None, sr: int = SR) -> SampleLibrary:
    """Собирает библиотеку из папки с сэмплами. Ошибки не фатальны."""
    directory = directory or library_dir()
    library = SampleLibrary()
    if not os.path.isdir(directory):
        return library

    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path) or os.path.splitext(name)[1].lower() not in (
                ".wav", ".flac", ".ogg", ".mp3", ".aiff"):
            continue
        lower = name.lower()
        try:
            audio = load(path, sr=sr, mono=True)
        except Exception as exc:  # noqa: BLE001 — битый файл не должен ронять рендер
            logger.warning("Сэмпл %s не читается: %s", name, exc)
            library.notes.append(f"{name}: не читается ({exc})")
            continue

        if any(word in lower for word in ("drum", "kit", "барабан", "beat", "perc")):
            kit = slice_drum_loop(audio, source=name)
            if kit.hits:
                library.kit = kit
                library.notes.append(
                    f"{name}: разобран на удары — {', '.join(f'{r}×{len(v)}' for r, v in kit.hits.items())}")
        elif any(word in lower for word in ("guitar", "гитар", "riff", "chug")):
            sample = extract_guitar_note(audio, source=name)
            if sample is not None:
                library.guitar = sample
                library.notes.append(
                    f"{name}: взята нота {sample.base_freq:.0f} Гц как основа гитары")
    return library


def slice_drum_loop(audio: Audio, source: str = "", sr: int | None = None,
                    max_hit: float = 0.9) -> DrumKit:
    """Режет барабанный луп по атакам и раскладывает удары по ролям."""
    sr = sr or audio.sr
    mono = audio.mono()
    hop = 256
    env = onset_envelope(mono, sr, hop=hop)
    if env.size < 4:
        return DrumKit(source=source)

    threshold = float(np.percentile(env, 88))
    peaks = []
    for i in range(1, len(env) - 1):
        if env[i] >= threshold and env[i] >= env[i - 1] and env[i] > env[i + 1]:
            position = i * hop
            if not peaks or position - peaks[-1] > int(0.04 * sr):   # не ближе 40 мс
                peaks.append(position)
    if not peaks:
        return DrumKit(source=source)

    hits: dict[str, list[tuple[float, np.ndarray]]] = {}
    for i, start in enumerate(peaks):
        end = peaks[i + 1] if i + 1 < len(peaks) else len(mono)
        end = min(end, start + int(max_hit * sr))
        piece = mono[start:end]
        if piece.size < int(0.01 * sr):
            continue
        role, score = classify_hit(piece, sr)
        hits.setdefault(role, []).append((score, normalize_peak(piece, -1.0)))

    kit = DrumKit(source=source)
    for role, variants in hits.items():
        # берём три самых характерных примера — этого хватает, чтобы удары
        # не звучали одинаково, и не раздувает память
        best = [piece for _, piece in sorted(variants, key=lambda v: -v[0])[:3]]
        kit.hits[role] = best
    _fill_missing_roles(kit)
    return kit


def classify_hit(piece: np.ndarray, sr: int) -> tuple[str, float]:
    """Определяет роль удара по распределению энергии и длительности.

    Пороги подобраны по измеренным профилям настоящих ударов: у бочки почти
    вся энергия ниже 150 Гц и нет верха, у хэта наоборот — один верх без
    середины, а рабочий узнаётся по сочетанию тела и «щелчка».
    """
    spectrum = np.abs(np.fft.rfft(piece * np.hanning(len(piece))))
    freqs = np.fft.rfftfreq(len(piece), 1.0 / sr)
    total = float(spectrum.sum()) + 1e-9

    low = float(spectrum[freqs < 150].sum()) / total
    mid = float(spectrum[(freqs >= 150) & (freqs < 2000)].sum()) / total
    high = float(spectrum[freqs >= 4000].sum()) / total
    duration = len(piece) / sr

    if low > 0.5 and mid < 0.2:
        return "kick", low                          # низ и почти ничего больше
    if low > 0.3:
        return "tom_lo", low + mid                  # низ есть, но с телом
    if high > 0.75 and mid < 0.1:
        return ("crash", high) if duration > 0.5 else ("hat", high)
    if high > 0.2 and mid > 0.1:
        return "snare", mid + high                  # тело плюс щелчок
    if high > 0.5:
        return "hat", high
    return "tom_hi", mid
def _fill_missing_roles(kit: DrumKit) -> None:
    """Недостающие роли берём у ближайших по смыслу, а не молчим."""
    fallbacks = {"openhat": "hat", "tom_hi": "snare", "tom_lo": "kick", "crash": "hat"}
    for role, source_role in fallbacks.items():
        if role not in kit.hits and source_role in kit.hits:
            kit.hits[role] = kit.hits[source_role]


def extract_guitar_note(audio: Audio, source: str = "", sr: int | None = None) -> GuitarSample | None:
    """Берёт из гитарного файла первую ноту и определяет её высоту."""
    sr = sr or audio.sr
    mono = audio.mono()
    if mono.size < int(0.05 * sr):
        return None

    start = int(np.argmax(np.abs(mono) > np.max(np.abs(mono)) * 0.2))
    piece = mono[start:start + int(1.2 * sr)]
    if piece.size < int(0.05 * sr):
        piece = mono

    freq = detect_pitch(piece, sr)
    if freq <= 0:
        # риффовый луп без явной высоты — играть им ноты нельзя
        logger.info("В сэмпле %s не нашлась высота ноты", source)
        return None
    return GuitarSample(audio=normalize_peak(piece, -1.0), base_freq=freq, source=source)


def detect_pitch(piece: np.ndarray, sr: int, fmin: float = 30.0, fmax: float = 500.0) -> float:
    """Автокорреляционный детектор высоты — тот же, что и для вокала."""
    x = piece[: int(0.4 * sr)].astype(np.float32)
    x = x - x.mean()
    if np.max(np.abs(x)) < 1e-4:
        return 0.0
    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    lag_min, lag_max = int(sr / fmax), min(int(sr / fmin), len(corr) - 1)
    if lag_max <= lag_min:
        return 0.0
    segment = corr[lag_min:lag_max]
    lag = int(np.argmax(segment)) + lag_min
    if corr[lag] < corr[0] * 0.25:
        return 0.0
    return sr / lag
