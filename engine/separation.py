"""Разделение трека на дорожки (stems).

Основной путь — Demucs (htdemucs / htdemucs_6s), если он установлен.
Запасной — честный DSP-сплит на numpy: center-extraction для вокала +
HPSS для разделения ударных и гармонической части + фильтрация баса.
Так сервис отдаёт дорожки даже без GPU и тяжёлых моделей.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from . import SR
from .audio_io import Audio, save
from .dsp import bandpass, highpass, lowpass

logger = logging.getLogger(__name__)

@dataclass
class SeparationResult:
    stems: dict[str, Audio]
    backend: str
    model: str = ""

    def write(self, out_dir: str, fmt: str = "wav") -> dict[str, str]:
        paths = {}
        os.makedirs(out_dir, exist_ok=True)
        for name, audio in self.stems.items():
            paths[name] = save(os.path.join(out_dir, f"{name}.{fmt}"), audio)
        return paths


def demucs_available() -> bool:
    try:
        import demucs.api  # noqa: F401

        return True
    except Exception:
        return False


def separate(audio: Audio, model: str = "htdemucs", prefer_backend: str | None = None) -> SeparationResult:
    """Возвращает дорожки. prefer_backend: 'demucs' | 'dsp' | None (авто)."""
    backend = prefer_backend or ("demucs" if demucs_available() else "dsp")
    if backend == "demucs":
        try:
            return _separate_demucs(audio, model)
        except Exception as exc:  # pragma: no cover - зависит от наличия модели
            logger.warning("Demucs недоступен (%s), переключаюсь на DSP-разделение", exc)
    return _separate_dsp(audio)


def _separate_demucs(audio: Audio, model: str) -> SeparationResult:  # pragma: no cover - тяжёлая зависимость
    import torch
    from demucs.api import Separator

    separator = Separator(model=model)
    wav = torch.from_numpy(audio.stereo().data).float()
    _, sources = separator.separate_tensor(wav, sr=audio.sr)
    stems = {name: Audio(tensor.cpu().numpy(), audio.sr) for name, tensor in sources.items()}
    return SeparationResult(stems=stems, backend="demucs", model=model)


def _separate_dsp(audio: Audio) -> SeparationResult:
    """DSP-разделение: center/side для вокала, HPSS для ударных, фильтр для баса."""
    st = audio.stereo()
    sr = st.sr
    left, right = st.data[0], st.data[1]

    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    # 1. Вокал — центр канала в вокальном диапазоне
    vocal_band = bandpass(mid, 180.0, 8000.0, sr)
    vocal_mask = _spectral_center_mask(left, right, sr)
    vocals_mono = _apply_mask(vocal_band, vocal_mask, sr)
    vocals = Audio(np.stack([vocals_mono, vocals_mono]), sr)

    # 2. Инструментал = исходник минус выделенный центр
    instrumental = np.stack([left - vocals_mono * 0.9, right - vocals_mono * 0.9])

    # 3. Ударные и гармония — гармонико-перкуссионное разделение
    harmonic, percussive = hpss(instrumental.mean(axis=0), sr)

    drums_mono = percussive
    drums = Audio(np.stack([drums_mono, drums_mono]), sr)

    bass_mono = lowpass(harmonic, 220.0, sr)
    bass = Audio(np.stack([bass_mono, bass_mono]), sr)

    other_mono = highpass(harmonic, 200.0, sr)
    other = Audio(np.stack([other_mono + side * 0.5, other_mono - side * 0.5]), sr)

    stems = {"vocals": vocals, "drums": drums, "bass": bass, "other": other}
    return SeparationResult(stems=stems, backend="dsp", model="center+hpss")


def _spectral_center_mask(left: np.ndarray, right: np.ndarray, sr: int,
                          n_fft: int = 4096, hop: int = 1024) -> np.ndarray:
    """Маска «насколько бин сцентрирован» — грубый, но рабочий вокал-детектор."""
    def _stft(x):
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))
        window = np.hanning(n_fft).astype(np.float32)
        n_frames = 1 + (len(x) - n_fft) // hop
        idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
        return np.fft.rfft(x[idx] * window, axis=1)

    L, R = _stft(left), _stft(right)
    num = np.abs(L * np.conj(R))
    den = np.maximum(np.abs(L) * np.abs(R), 1e-9)
    coherence = np.clip(num / den, 0, 1)
    balance = 1.0 - np.abs(np.abs(L) - np.abs(R)) / np.maximum(np.abs(L) + np.abs(R), 1e-9)
    return (coherence * balance).astype(np.float32)


def _apply_mask(x: np.ndarray, mask: np.ndarray, sr: int, n_fft: int = 4096, hop: int = 1024) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    padded = np.pad(x, (0, n_fft))
    n_frames = min(mask.shape[0], 1 + (len(padded) - n_fft) // hop)
    out = np.zeros(len(padded), dtype=np.float32)
    win_sum = np.zeros(len(padded), dtype=np.float32)
    for i in range(n_frames):
        seg = padded[i * hop: i * hop + n_fft] * window
        spec = np.fft.rfft(seg) * mask[i]
        out[i * hop: i * hop + n_fft] += np.fft.irfft(spec, n=n_fft).astype(np.float32) * window
        win_sum[i * hop: i * hop + n_fft] += window ** 2
    out /= _safe_win_sum(win_sum)
    return out[: len(x)]


def _safe_win_sum(win_sum: np.ndarray) -> np.ndarray:
    """На краях сумма окон стремится к нулю — иначе деление взрывает амплитуду."""
    floor = max(float(win_sum.max()) * 0.1, 1e-6)
    return np.maximum(win_sum, floor)


def hpss(y: np.ndarray, sr: int = SR, n_fft: int = 2048, hop: int = 512,
         kernel: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """Гармонико-перкуссионное разделение медианной фильтрацией спектрограммы."""
    window = np.hanning(n_fft).astype(np.float32)
    padded = np.pad(y, (0, n_fft))
    n_frames = 1 + (len(padded) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(padded[idx] * window, axis=1)
    mag = np.abs(spec)

    harm = _median_filter_1d(mag, kernel, axis=0)   # медиана по времени
    perc = _median_filter_1d(mag, kernel, axis=1)   # медиана по частоте
    total = np.maximum(harm ** 2 + perc ** 2, 1e-12)
    mask_h = harm ** 2 / total
    mask_p = perc ** 2 / total

    def _istft(masked_spec):
        out = np.zeros(len(padded), dtype=np.float32)
        win_sum = np.zeros(len(padded), dtype=np.float32)
        frames = np.fft.irfft(masked_spec, n=n_fft, axis=1).astype(np.float32) * window
        for i in range(n_frames):
            out[i * hop: i * hop + n_fft] += frames[i]
            win_sum[i * hop: i * hop + n_fft] += window ** 2
        return (out / _safe_win_sum(win_sum))[: len(y)]

    return _istft(spec * mask_h), _istft(spec * mask_p)


def _median_filter_1d(mat: np.ndarray, size: int, axis: int) -> np.ndarray:
    try:
        from scipy.ndimage import median_filter

        shape = [1, 1]
        shape[axis] = size
        return median_filter(mat, size=tuple(shape), mode="nearest")
    except Exception:  # pragma: no cover
        kernel = np.ones(size, dtype=np.float32) / size
        return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), axis, mat)
