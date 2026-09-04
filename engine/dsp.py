"""Базовый DSP: фильтры, динамика, эффекты, питч-шифт.

Всё на numpy/scipy — без внешних плагинов, чтобы движок работал в любом
окружении. Если установлен `pedalboard`, его цепочки можно подключить
поверх (см. engine.amp), но обязательным он не является.
"""
from __future__ import annotations

import numpy as np

from . import SR

try:  # scipy есть в requirements, но код не должен разваливаться без него
    from scipy.signal import butter, fftconvolve, sosfilt

    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :], True
    return arr, False


def _restore(arr: np.ndarray, was_1d: bool) -> np.ndarray:
    return arr[0] if was_1d else arr


def _fft_filter(x: np.ndarray, sr: int, kind: str, f_low: float, f_high: float) -> np.ndarray:
    """Запасной фильтр в частотной области (когда нет scipy)."""
    n = x.shape[-1]
    spec = np.fft.rfft(x, axis=-1)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    if kind == "low":
        gain = 1.0 / np.sqrt(1.0 + (freqs / max(f_high, 1e-6)) ** 4)
    elif kind == "high":
        gain = 1.0 / np.sqrt(1.0 + (max(f_low, 1e-6) / np.maximum(freqs, 1e-6)) ** 4)
    else:
        lo = 1.0 / np.sqrt(1.0 + (max(f_low, 1e-6) / np.maximum(freqs, 1e-6)) ** 4)
        hi = 1.0 / np.sqrt(1.0 + (freqs / max(f_high, 1e-6)) ** 4)
        gain = lo * hi
    return np.fft.irfft(spec * gain, n=n, axis=-1).astype(np.float32)


def _butter(x: np.ndarray, sr: int, kind: str, f_low: float, f_high: float, order: int) -> np.ndarray:
    nyq = sr / 2.0
    if kind == "low":
        wn = min(f_high / nyq, 0.999)
    elif kind == "high":
        wn = max(min(f_low / nyq, 0.999), 1e-4)
    else:
        wn = [max(min(f_low / nyq, 0.998), 1e-4), min(f_high / nyq, 0.999)]
        if wn[0] >= wn[1]:
            return x
    sos = butter(order, wn, btype=kind, output="sos")
    return sosfilt(sos, x, axis=-1).astype(np.float32)


def lowpass(x, cutoff: float, sr: int = SR, order: int = 4):
    arr, flat = _as_2d(x)
    out = _butter(arr, sr, "low", 0.0, cutoff, order) if HAS_SCIPY else _fft_filter(arr, sr, "low", 0.0, cutoff)
    return _restore(out, flat)


def highpass(x, cutoff: float, sr: int = SR, order: int = 4):
    arr, flat = _as_2d(x)
    out = _butter(arr, sr, "high", cutoff, 0.0, order) if HAS_SCIPY else _fft_filter(arr, sr, "high", cutoff, 0.0)
    return _restore(out, flat)


def bandpass(x, low: float, high: float, sr: int = SR, order: int = 4):
    arr, flat = _as_2d(x)
    out = _butter(arr, sr, "band", low, high, order) if HAS_SCIPY else _fft_filter(arr, sr, "band", low, high)
    return _restore(out, flat)


def peaking_eq(x, freq: float, gain_db: float, q: float = 1.0, sr: int = SR):
    """Классический RBJ peaking-фильтр (колокол)."""
    arr, flat = _as_2d(x)
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A], dtype=np.float64)
    a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A], dtype=np.float64)
    b /= a[0]
    a /= a[0]
    if HAS_SCIPY:
        from scipy.signal import lfilter

        out = lfilter(b, a, arr, axis=-1).astype(np.float32)
    else:  # pragma: no cover
        out = arr
    return _restore(out, flat)


def envelope(x: np.ndarray, sr: int, attack_ms: float, release_ms: float, block: int = 64) -> np.ndarray:
    """Блочный follower огибающей: быстрый в Python и достаточно точный."""
    mono = np.max(np.abs(x), axis=0) if x.ndim > 1 else np.abs(x)
    n = mono.shape[-1]
    n_blocks = max(1, int(np.ceil(n / block)))
    padded = np.zeros(n_blocks * block, dtype=np.float32)
    padded[:n] = mono
    peaks = padded.reshape(n_blocks, block).max(axis=1)

    a_att = float(np.exp(-block / (sr * max(attack_ms, 0.01) / 1000.0)))
    a_rel = float(np.exp(-block / (sr * max(release_ms, 0.01) / 1000.0)))
    env_blocks = np.empty_like(peaks)
    state = 0.0
    for i, p in enumerate(peaks):
        coef = a_att if p > state else a_rel
        state = coef * state + (1.0 - coef) * float(p)
        env_blocks[i] = state
    env = np.repeat(env_blocks, block)[:n]
    return env.astype(np.float32)


def compressor(x, threshold_db=-18.0, ratio=4.0, attack_ms=10.0, release_ms=120.0,
               makeup_db=0.0, knee_db=6.0, sr: int = SR):
    arr, flat = _as_2d(x)
    env = envelope(arr, sr, attack_ms, release_ms)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    over = env_db - threshold_db
    # мягкое колено
    gain_db = np.where(
        over <= -knee_db / 2,
        0.0,
        np.where(
            over >= knee_db / 2,
            over * (1.0 / ratio - 1.0),
            (1.0 / ratio - 1.0) * (over + knee_db / 2) ** 2 / (2 * knee_db),
        ),
    )
    gain = 10.0 ** ((gain_db + makeup_db) / 20.0)
    return _restore((arr * gain).astype(np.float32), flat)


def gate(x, threshold_db=-45.0, attack_ms=1.0, release_ms=90.0, range_db=-60.0, sr: int = SR):
    arr, flat = _as_2d(x)
    env = envelope(arr, sr, attack_ms, release_ms)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    open_amt = np.clip((env_db - threshold_db) / 6.0, 0.0, 1.0)
    floor = db_to_lin(range_db)
    gain = floor + (1.0 - floor) * open_amt
    return _restore((arr * gain).astype(np.float32), flat)


def limiter(x, ceiling_db=-0.3, lookahead_ms=2.0, release_ms=60.0, sr: int = SR):
    arr, flat = _as_2d(x)
    ceiling = db_to_lin(ceiling_db)
    env = envelope(arr, sr, 0.1, release_ms)
    gain = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    la = int(sr * lookahead_ms / 1000.0)
    if la > 0:  # смотрим вперёд, чтобы успеть придавить транзиент
        gain = np.concatenate([gain[la:], np.full(la, gain[-1], dtype=np.float32)])
    out = np.clip(arr * gain, -1.0, 1.0).astype(np.float32)
    return _restore(out, flat)


def soft_clip(x, drive: float = 1.0):
    arr = np.asarray(x, dtype=np.float32)
    return np.tanh(arr * max(drive, 1e-6)).astype(np.float32)


def waveshape(x, drive: float = 4.0, asymmetry: float = 0.12, mode: str = "tube"):
    """Нелинейность для гитары/баса: tube = мягкое асимметричное насыщение."""
    arr = np.asarray(x, dtype=np.float32) * drive
    if mode == "fuzz":
        shaped = np.sign(arr) * (1.0 - np.exp(-np.abs(arr)))
    elif mode == "hard":
        shaped = np.clip(arr, -0.8, 0.8)
    else:
        shaped = np.tanh(arr + asymmetry) - np.tanh(asymmetry)
    return (shaped / max(np.tanh(drive), 1e-6)).astype(np.float32)


def delay_fx(x, sr: int = SR, time_s: float = 0.28, feedback: float = 0.32,
             mix_amt: float = 0.22, pingpong: bool = True):
    arr, flat = _as_2d(x)
    d = max(1, int(time_s * sr))
    wet = np.zeros_like(arr)
    src = arr.copy()
    gain = 1.0
    for tap in range(1, 6):
        gain *= feedback
        if gain < 0.01:
            break
        shift = d * tap
        if shift >= arr.shape[-1]:
            break
        tap_sig = np.zeros_like(arr)
        tap_sig[:, shift:] = src[:, : arr.shape[-1] - shift] * gain
        if pingpong and arr.shape[0] == 2 and tap % 2 == 1:
            tap_sig = tap_sig[::-1]
        wet += tap_sig
    wet = lowpass(wet, 6000, sr)
    return _restore(((1 - mix_amt) * arr + mix_amt * wet).astype(np.float32), flat)


def _reverb_ir(sr: int, decay_s: float, pre_delay_ms: float, damping: float, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = int(sr * decay_s)
    t = np.arange(n) / sr
    noise = rng.standard_normal((2, n)).astype(np.float32)
    ir = noise * np.exp(-t * (6.0 / max(decay_s, 0.05)))[None, :]
    ir = lowpass(ir, max(1200.0, 12000.0 * (1.0 - damping)), sr)
    pre = int(sr * pre_delay_ms / 1000.0)
    if pre:
        ir = np.concatenate([np.zeros((2, pre), dtype=np.float32), ir], axis=1)
    ir /= max(float(np.max(np.abs(ir))), 1e-9)
    return ir.astype(np.float32)


def reverb(x, sr: int = SR, decay_s: float = 1.6, mix_amt: float = 0.18,
           pre_delay_ms: float = 20.0, damping: float = 0.4):
    arr, flat = _as_2d(x)
    if not HAS_SCIPY:  # pragma: no cover
        return _restore(arr, flat)
    ir = _reverb_ir(sr, decay_s, pre_delay_ms, damping)
    stereo = arr if arr.shape[0] == 2 else np.repeat(arr, 2, axis=0)
    wet = np.stack([fftconvolve(stereo[c], ir[c])[: stereo.shape[-1]] for c in range(2)])
    wet *= 0.35
    out = (1 - mix_amt) * stereo + mix_amt * wet
    if arr.shape[0] == 1:
        out = out.mean(axis=0, keepdims=True)
    return _restore(out.astype(np.float32), flat)


def stereo_width(x, width: float = 1.4):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 2:
        return arr
    mid = (arr[0] + arr[1]) * 0.5
    side = (arr[0] - arr[1]) * 0.5 * width
    return np.stack([mid + side, mid - side]).astype(np.float32)


def pan(x, position: float = 0.0):
    """position: -1 (лево) .. +1 (право). Равномощное панорамирование."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.repeat(arr[None, :], 2, axis=0)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 2, axis=0)
    angle = (np.clip(position, -1, 1) + 1) * np.pi / 4
    return np.stack([arr[0] * np.cos(angle), arr[1] * np.sin(angle)]).astype(np.float32) * np.sqrt(2)


def fade(x, sr: int = SR, fade_in_ms: float = 5.0, fade_out_ms: float = 30.0):
    arr, flat = _as_2d(x)
    n = arr.shape[-1]
    out = arr.copy()
    ni = min(int(sr * fade_in_ms / 1000.0), n)
    no = min(int(sr * fade_out_ms / 1000.0), n)
    if ni > 0:
        out[:, :ni] *= np.linspace(0, 1, ni, dtype=np.float32)
    if no > 0:
        out[:, -no:] *= np.linspace(1, 0, no, dtype=np.float32)
    return _restore(out, flat)


def rms_db(x) -> float:
    arr = np.asarray(x, dtype=np.float32)
    return lin_to_db(np.sqrt(np.mean(arr ** 2)) if arr.size else 1e-9)


def normalize_peak(x, peak_db: float = -1.0):
    arr = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak < 1e-9:
        return arr
    return (arr * (db_to_lin(peak_db) / peak)).astype(np.float32)


def normalize_active_rms(x, target_db: float = -18.0, floor_ratio: float = 0.05):
    """Нормализует по RMS «звучащих» участков — не задирает редкие партии."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0 or float(np.max(np.abs(arr))) < 1e-9:
        return arr
    mono = np.max(np.abs(arr), axis=0) if arr.ndim > 1 else np.abs(arr)
    # опорный уровень берём по перцентилю, а не по максимуму: одиночный
    # выброс не должен решать, что считать «звучащим» участком
    reference = float(np.percentile(mono, 99.0)) or float(np.max(mono))
    active = mono > reference * floor_ratio
    if not np.any(active):
        return arr
    sel = arr[:, active] if arr.ndim > 1 else arr[active]
    current = lin_to_db(np.sqrt(np.mean(sel ** 2)))
    gain_db = float(np.clip(target_db - current, -24.0, 24.0))
    return (arr * db_to_lin(gain_db)).astype(np.float32)


def normalize_loudness(x, target_db: float = -14.0, sr: int = SR):
    """Приближение к целевой громкости по K-взвешенному RMS."""
    arr, flat = _as_2d(x)
    weighted = highpass(arr, 80.0, sr, order=2)
    current = rms_db(weighted)
    gain = db_to_lin(target_db - current)
    out = np.clip(arr * gain, -4.0, 4.0).astype(np.float32)
    return _restore(out, flat)


def time_stretch(x, rate: float, sr: int = SR, frame: int = 2048, hop: int = 512):
    """Фазовый вокодер: меняет длительность, не трогая высоту."""
    arr, flat = _as_2d(x)
    if abs(rate - 1.0) < 1e-3:
        return _restore(arr, flat)
    window = np.hanning(frame).astype(np.float32)
    out_channels = []
    for ch in arr:
        n_frames = max(1, 1 + (len(ch) - frame) // hop)
        stft = np.stack([
            np.fft.rfft(ch[i * hop: i * hop + frame] * window)
            for i in range(n_frames)
            if i * hop + frame <= len(ch)
        ]) if len(ch) >= frame else np.zeros((1, frame // 2 + 1), dtype=complex)
        mag, phase = np.abs(stft), np.angle(stft)
        dphase = np.diff(phase, axis=0, prepend=phase[:1])
        positions = np.arange(0, stft.shape[0] - 1, rate)
        out_len = int(len(positions) * hop) + frame
        out = np.zeros(out_len, dtype=np.float32)
        win_sum = np.zeros(out_len, dtype=np.float32)
        acc_phase = phase[0].copy()
        for i, p in enumerate(positions):
            idx = int(np.floor(p))
            frac = p - idx
            m = (1 - frac) * mag[idx] + frac * mag[min(idx + 1, len(mag) - 1)]
            spec = m * np.exp(1j * acc_phase)
            grain = np.fft.irfft(spec, n=frame).astype(np.float32) * window
            start = i * hop
            out[start: start + frame] += grain
            win_sum[start: start + frame] += window ** 2
            acc_phase = acc_phase + dphase[min(idx + 1, len(dphase) - 1)]
        out /= np.maximum(win_sum, 1e-6)
        out_channels.append(out)
    n = max(len(c) for c in out_channels)
    stacked = np.stack([np.pad(c, (0, n - len(c))) for c in out_channels])
    return _restore(stacked.astype(np.float32), flat)


def pitch_shift(x, semitones: float, sr: int = SR):
    """Питч-шифт = растяжение времени + передискретизация обратно."""
    if abs(semitones) < 1e-3:
        return np.asarray(x, dtype=np.float32)
    arr, flat = _as_2d(x)
    ratio = 2.0 ** (semitones / 12.0)
    stretched = time_stretch(arr, 1.0 / ratio, sr)
    n_out = int(round(stretched.shape[-1] / ratio))
    src = np.linspace(0, stretched.shape[-1] - 1, n_out)
    out = np.stack([np.interp(src, np.arange(stretched.shape[-1]), ch) for ch in stretched])
    out = out.astype(np.float32)
    # страховка от выбросов фазового вокодера на стыках окон
    in_peak = float(np.max(np.abs(arr)))
    out_peak = float(np.max(np.abs(out))) if out.size else 0.0
    if in_peak > 1e-9 and out_peak > in_peak * 2.0:
        out *= (in_peak * 2.0) / out_peak
    return _restore(out, flat)


def formant_shift(x, semitones: float, sr: int = SR):
    """Грубый сдвиг формант: масштабируем спектральную огибающую."""
    arr, flat = _as_2d(x)
    ratio = 2.0 ** (semitones / 12.0)
    n = arr.shape[-1]
    spec = np.fft.rfft(arr, axis=-1)
    mag, phase = np.abs(spec), np.angle(spec)
    bins = np.arange(mag.shape[-1])
    shifted = np.stack([np.interp(bins, bins * ratio, m, left=0.0, right=0.0) for m in mag])
    out = np.fft.irfft(shifted * np.exp(1j * phase), n=n, axis=-1)
    return _restore(out.astype(np.float32), flat)


def sidechain_duck(target, trigger, sr: int = SR, amount_db: float = -6.0,
                   attack_ms: float = 8.0, release_ms: float = 180.0):
    """Приглушает target там, где играет trigger (для вокала поверх инструментала)."""
    tgt, flat = _as_2d(target)
    trg, _ = _as_2d(trigger)
    n = tgt.shape[-1]
    trg_env = envelope(trg[:, :n] if trg.shape[-1] >= n else
                       np.pad(trg, ((0, 0), (0, n - trg.shape[-1]))), sr, attack_ms, release_ms)
    trg_db = 20 * np.log10(np.maximum(trg_env, 1e-9))
    depth = np.clip((trg_db + 40.0) / 40.0, 0.0, 1.0)
    gain = 10 ** ((depth * amount_db) / 20.0)
    return _restore((tgt * gain).astype(np.float32), flat)
