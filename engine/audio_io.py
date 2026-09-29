"""Чтение/запись аудио и конвертация форматов.

soundfile — основной путь (wav/flac/ogg), ffmpeg — для всего остального
(mp3, m4a, webm с диктофона браузера). Если ffmpeg недоступен, мы честно
сообщаем об этом, а не падаем с невнятной ошибкой.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from . import SR


class AudioIOError(RuntimeError):
    pass


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


@dataclass
class Audio:
    """Аудио всегда храним как float32 (channels, samples)."""

    data: np.ndarray
    sr: int = SR

    def __post_init__(self):
        arr = np.asarray(self.data, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        self.data = arr

    @property
    def channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / float(self.sr)

    def mono(self) -> np.ndarray:
        return self.data.mean(axis=0)

    def stereo(self) -> "Audio":
        if self.channels == 2:
            return self
        if self.channels == 1:
            return Audio(np.repeat(self.data, 2, axis=0), self.sr)
        return Audio(np.stack([self.data[0], self.data[1]]), self.sr)

    def peak(self) -> float:
        return float(np.max(np.abs(self.data))) if self.n_samples else 0.0

    def copy(self) -> "Audio":
        return Audio(self.data.copy(), self.sr)


def _decode_with_ffmpeg(path: str, sr: int) -> Audio:
    if not has_ffmpeg():
        raise AudioIOError(
            f"Не могу прочитать {os.path.basename(path)}: нужен ffmpeg "
            "(установите ffmpeg или загрузите wav/flac/ogg)."
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path, "-ar", str(sr),
             "-ac", "2", "-c:a", "pcm_f32le", tmp_path],
            check=True, capture_output=True,
        )
        data, file_sr = sf.read(tmp_path, dtype="float32", always_2d=True)
        return Audio(data.T, file_sr)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - зависит от ffmpeg
        raise AudioIOError(f"ffmpeg не смог декодировать файл: {exc.stderr[-400:]!r}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load(path: str, sr: int = SR, mono: bool = False) -> Audio:
    """Загружает файл, приводя к целевой частоте дискретизации."""
    try:
        data, file_sr = sf.read(path, dtype="float32", always_2d=True)
        audio = Audio(data.T, file_sr)
    except Exception:
        audio = _decode_with_ffmpeg(path, sr)
    if audio.sr != sr:
        audio = resample(audio, sr)
    if mono:
        audio = Audio(audio.mono()[None, :], audio.sr)
    return audio


def save(path: str, audio: Audio, subtype: str = "PCM_16") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".flac", ".ogg"):
        sf.write(path, audio.data.T, audio.sr, subtype=subtype if ext == ".wav" else None)
        return path
    # mp3 и прочее — через ffmpeg, с падением обратно в wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio.data.T, audio.sr, subtype="PCM_16")
        if not has_ffmpeg():
            fallback = os.path.splitext(path)[0] + ".wav"
            shutil.copyfile(tmp_path, fallback)
            return fallback
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", tmp_path, "-b:a", "256k", path],
            check=True, capture_output=True,
        )
        return path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def resample(audio: Audio, target_sr: int) -> Audio:
    """Полифазная передискретизация (scipy) с линейной интерполяцией как запасной вариант."""
    if audio.sr == target_sr:
        return audio
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(audio.sr), int(target_sr))
        up, down = target_sr // g, audio.sr // g
        out = resample_poly(audio.data, up, down, axis=1)
    except Exception:
        ratio = target_sr / float(audio.sr)
        n_out = int(round(audio.n_samples * ratio))
        src_idx = np.linspace(0, audio.n_samples - 1, n_out)
        out = np.stack([np.interp(src_idx, np.arange(audio.n_samples), ch) for ch in audio.data])
    return Audio(np.asarray(out, dtype=np.float32), target_sr)


def silence(seconds: float, sr: int = SR, channels: int = 2) -> Audio:
    return Audio(np.zeros((channels, int(round(seconds * sr))), dtype=np.float32), sr)


def pad_to(audio: Audio, n_samples: int) -> Audio:
    if audio.n_samples >= n_samples:
        return Audio(audio.data[:, :n_samples], audio.sr)
    pad = np.zeros((audio.channels, n_samples - audio.n_samples), dtype=np.float32)
    return Audio(np.concatenate([audio.data, pad], axis=1), audio.sr)


def mix(layers, sr: int = SR) -> Audio:
    """Суммирует список Audio/ndarray в один стерео Audio."""
    prepared = []
    for layer in layers:
        a = layer if isinstance(layer, Audio) else Audio(layer, sr)
        prepared.append(a.stereo())
    if not prepared:
        return silence(0.0, sr)
    n = max(a.n_samples for a in prepared)
    acc = np.zeros((2, n), dtype=np.float32)
    for a in prepared:
        acc[:, : a.n_samples] += a.data
    return Audio(acc, sr)
