"""Разделение трека на дорожки (stems).

Основной путь — Demucs (htdemucs / htdemucs_6s), если он установлен.
Запасной — честный DSP-сплит на numpy: center-extraction для вокала +
HPSS для разделения ударных и гармонической части + фильтрация баса.
Так сервис отдаёт дорожки даже без GPU и тяжёлых моделей.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np

from . import SR
from .audio_io import Audio, resample, save
from .dsp import bandpass, highpass, lowpass
from .models import ModelUnavailable, demucs_available, get_demucs

logger = logging.getLogger(__name__)

DEMUCS_SR = 44100          # все модели htdemucs обучены на 44.1 кГц стерео


@dataclass
class SeparationResult:
    stems: dict[str, Audio]
    backend: str
    model: str = ""
    seconds: float = 0.0
    detail: str = ""

    def write(self, out_dir: str, fmt: str = "wav") -> dict[str, str]:
        paths = {}
        os.makedirs(out_dir, exist_ok=True)
        for name, audio in self.stems.items():
            paths[name] = save(os.path.join(out_dir, f"{name}.{fmt}"), audio)
        return paths


def separate(audio: Audio, model: str = "htdemucs",
             prefer_backend: str | None = None) -> SeparationResult:
    """Возвращает дорожки. prefer_backend: 'demucs' | 'dsp' | None (авто).

    Demucs используется, когда установлен; при любой его ошибке молча
    откатываемся на DSP-разделение — пользователь получит результат хуже,
    но получит.
    """
    started = time.time()
    reason = ""
    backend = prefer_backend or ("demucs" if demucs_available() else "dsp")
    if backend == "dsp" and prefer_backend != "dsp" and not demucs_available():
        reason = "пакет demucs не установлен"
    if backend == "demucs":
        try:
            result = _separate_demucs(audio, model)
            result.seconds = round(time.time() - started, 2)
            return result
        except ModelUnavailable as exc:
            logger.info("Demucs недоступен (%s) — DSP-разделение", exc)
            reason = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Demucs упал (%s) — DSP-разделение", exc, exc_info=True)
            reason = str(exc)
    result = _separate_dsp(audio)
    result.seconds = round(time.time() - started, 2)
    result.detail = reason
    return result


def _separate_demucs(audio: Audio, model: str) -> SeparationResult:
    """Разделение нейросетью Demucs. Модель кэшируется между задачами."""
    import torch

    separator = get_demucs(model)
    model_sr = int(getattr(separator, "samplerate", DEMUCS_SR) or DEMUCS_SR)

    source = audio.stereo()
    if source.sr != model_sr:
        source = resample(source, model_sr)

    wav = torch.from_numpy(np.ascontiguousarray(source.data)).float()
    with torch.no_grad():
        _, sources = separator.separate_tensor(wav, sr=model_sr)
    del wav, source                      # исходник больше не нужен, он весь в sources

    # Забираем дорожки по одной и сразу отпускаем тензор: иначе на трёхминутном
    # треке одновременно живут и четыре тензора Torch, и четыре копии numpy —
    # лишние сотни мегабайт ровно там, где память и заканчивается.
    stems: dict[str, Audio] = {}
    for name in list(sources):
        tensor = sources.pop(name)
        data = tensor.detach().cpu().numpy().astype(np.float32)
        del tensor
        stem = Audio(data, model_sr)
        stems[name] = resample(stem, audio.sr) if model_sr != audio.sr else stem

    return SeparationResult(stems=stems, backend="demucs", model=model,
                            detail=f"{len(stems)} дорожек, {model_sr} Гц")


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
                          n_fft: int = 4096, hop: int = 1024,
                          block: int = 128) -> np.ndarray:
    """Маска «насколько бин сцентрирован» — грубый, но рабочий вокал-детектор.

    Считается блоками кадров. Спектр целого трека — это комплексная матрица
    двойной точности на каждый канал: на трёх минутах выходило больше
    полугигабайта временных массивов, и весь пик памяти рендера создавала
    именно эта функция. Блоками результат тот же, а память — от размера
    блока, а не от длины трека.
    """
    window = np.hanning(n_fft).astype(np.float32)
    if len(left) < n_fft:
        left = np.pad(left, (0, n_fft - len(left)))
        right = np.pad(right, (0, n_fft - len(right)))
    n_frames = 1 + (len(left) - n_fft) // hop
    mask = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    offsets = np.arange(n_fft)[None, :]

    for start in range(0, n_frames, block):
        stop = min(start + block, n_frames)
        idx = offsets + hop * np.arange(start, stop)[:, None]
        spec_l = np.fft.rfft(left[idx] * window, axis=1)
        spec_r = np.fft.rfft(right[idx] * window, axis=1)
        abs_l, abs_r = np.abs(spec_l), np.abs(spec_r)
        num = np.abs(spec_l * np.conj(spec_r))
        den = np.maximum(abs_l * abs_r, 1e-9)
        coherence = np.clip(num / den, 0, 1)
        balance = 1.0 - np.abs(abs_l - abs_r) / np.maximum(abs_l + abs_r, 1e-9)
        mask[start:stop] = (coherence * balance).astype(np.float32)
    return mask


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
         kernel: int = 17, block: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Гармонико-перкуссионное разделение медианной фильтрацией спектрограммы.

    Считается блоками кадров и в одинарной точности. Раньше здесь разом
    жили спектр трека в complex128, его модуль, две медианы, две маски и
    полная матрица обратного БПФ — на трёх минутах больше гигабайта, весь
    пик памяти рендера. Слышимой разницы нет: аудио и так float32.
    """
    window = np.hanning(n_fft).astype(np.float32)
    padded = np.pad(y, (0, n_fft))
    n_frames = 1 + (len(padded) - n_fft) // hop
    offsets = np.arange(n_fft)[None, :]

    spec = np.empty((n_frames, n_fft // 2 + 1), dtype=np.complex64)
    for start in range(0, n_frames, block):
        stop = min(start + block, n_frames)
        idx = offsets + hop * np.arange(start, stop)[:, None]
        spec[start:stop] = np.fft.rfft(padded[idx] * window, axis=1)

    mag = np.abs(spec)                              # float32
    harm = _median_filter_1d(mag, kernel, axis=0)   # медиана по времени
    perc = _median_filter_1d(mag, kernel, axis=1)   # медиана по частоте
    del mag
    total = np.maximum(harm * harm + perc * perc, 1e-12)
    mask_h = (harm * harm / total).astype(np.float32)
    del harm
    mask_p = (perc * perc / total).astype(np.float32)
    del perc, total

    # сумма окон одна на оба прохода — она зависит только от сетки кадров
    win_sum = np.zeros(len(padded), dtype=np.float32)
    window_sq = window * window
    for i in range(n_frames):
        win_sum[i * hop: i * hop + n_fft] += window_sq
    win_sum = _safe_win_sum(win_sum)

    def _istft(mask):
        out = np.zeros(len(padded), dtype=np.float32)
        for start in range(0, n_frames, block):
            stop = min(start + block, n_frames)
            frames = np.fft.irfft(spec[start:stop] * mask[start:stop], n=n_fft, axis=1)
            frames = frames.astype(np.float32) * window
            for i in range(start, stop):
                out[i * hop: i * hop + n_fft] += frames[i - start]
        out /= win_sum
        return out[: len(y)]

    return _istft(mask_h), _istft(mask_p)


def _median_filter_1d(mat: np.ndarray, size: int, axis: int) -> np.ndarray:
    try:
        from scipy.ndimage import median_filter

        shape = [1, 1]
        shape[axis] = size
        return median_filter(mat, size=tuple(shape), mode="nearest")
    except Exception:  # pragma: no cover
        kernel = np.ones(size, dtype=np.float32) / size
        return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), axis, mat)
