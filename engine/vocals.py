"""Обработка вокала: записанный голос → готовая дорожка в миксе.

Что здесь есть:
  * автоматическая цепочка обработки под стиль (читка/скрим/чистый/шёпот);
  * компенсация задержки записи в браузере (кросс-корреляция с минусовкой);
  * дабл-треки и гармонии, в том числе «переворот» пола голоса;
  * мягкий автотюн по тональности трека.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import SR
from .analysis import PITCH_NAMES, TrackAnalysis
from .audio_io import Audio
from .dsp import (compressor, db_to_lin, delay_fx, formant_shift, gate, highpass,
                  limiter, lowpass, normalize_active_rms, normalize_peak, peaking_eq,
                  pan, pitch_shift, reverb, rms_db, waveshape)

SCALES = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
}


@dataclass
class VocalChainPreset:
    hpf: float = 90.0
    gate_db: float = -42.0
    deess_db: float = -5.0
    presence_db: float = 3.5
    body_db: float = 1.5
    comp_threshold: float = -20.0
    comp_ratio: float = 4.0
    drive: float = 1.0
    delay_mix: float = 0.14
    reverb_mix: float = 0.16
    doubling: float = 0.0
    autotune: float = 0.0        # 0..1 — сила притяжения к нотам тональности
    target_db: float = -16.0


# Пресеты намеренно сухие. Реверберация и автотюн — то, что слышно первым и
# раздражает первым: голос «в бочке» и «затюненный» ломают доверие к треку
# быстрее, чем любой другой дефект. Кому нужно больше — прибавит вручную.
PRESETS: dict[str, VocalChainPreset] = {
    "rap": VocalChainPreset(hpf=110, deess_db=-6, presence_db=4, comp_threshold=-22,
                            comp_ratio=5.0, drive=1.2, delay_mix=0.05, reverb_mix=0.03,
                            doubling=0.2, autotune=0.0, target_db=-14.5),
    "scream": VocalChainPreset(hpf=130, gate_db=-38, deess_db=-3, presence_db=3.5,
                               comp_threshold=-24, comp_ratio=6.0, drive=2.0,
                               delay_mix=0.07, reverb_mix=0.06, doubling=0.3,
                               autotune=0.0, target_db=-14.0),
    "clean": VocalChainPreset(hpf=85, deess_db=-6, presence_db=2.5, body_db=1.5,
                              comp_threshold=-19, comp_ratio=3.0, drive=0.6,
                              delay_mix=0.06, reverb_mix=0.07, doubling=0.15,
                              autotune=0.0, target_db=-15.5),
    "whisper": VocalChainPreset(hpf=140, gate_db=-48, presence_db=5, comp_threshold=-26,
                                comp_ratio=5.0, drive=0.5, delay_mix=0.10, reverb_mix=0.10,
                                autotune=0.0, target_db=-19.0),
}

# Ориентировочные сдвиги для «перекраски» голоса по полу
GENDER_SHIFT = {
    ("male", "female"): (7.0, 3.0),     # (питч, форманты) в полутонах
    ("female", "male"): (-7.0, -3.0),
    ("male", "male"): (0.0, 0.0),
    ("female", "female"): (0.0, 0.0),
}


@dataclass
class VocalTakeResult:
    audio: Audio
    latency_ms: float = 0.0
    peak_db: float = 0.0
    notes: list[str] = field(default_factory=list)


def _latency_envelope(x: np.ndarray, sr: int, decim: int = 64) -> np.ndarray:
    """Огибающая с прореживанием: разрешение ~1.5 мс при дешёвой корреляции."""
    mono = np.abs(x)
    n = (len(mono) // decim) * decim
    if n < decim:
        return np.zeros(0, dtype=np.float32)
    env = mono[:n].reshape(-1, decim).max(axis=1)
    env = env - env.mean()
    return env.astype(np.float32)


def estimate_latency(take: Audio, reference: Audio, max_ms: float = 500.0,
                     min_correlation: float = 0.35, decim: int = 64) -> float:
    """Задержка записи относительно минусовки, мс.

    Считаем нормированную взаимную корреляцию огибающих. Если совпадение
    слабое (в дубле нет «подпевающей» минусовки), возвращаем 0 — лучше не
    двигать дубль вовсе, чем сдвинуть его наугад.
    """
    sr = take.sr
    limit = int(min(take.n_samples, reference.n_samples, sr * 30))
    if limit < sr:
        return 0.0
    a = _latency_envelope(take.mono()[:limit], sr, decim)
    b = _latency_envelope(reference.mono()[:limit], sr, decim)
    if a.size < 8 or b.size < 8:
        return 0.0

    max_lag = min(int(sr * max_ms / 1000.0 / decim), a.size - 4)
    if max_lag < 1:
        return 0.0
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-9:
        return 0.0
    scores = np.array([float(np.dot(a[lag: lag + b.size - max_lag],
                                    b[: b.size - max_lag])) / norm
                       for lag in range(max_lag)])
    best = int(np.argmax(scores))
    if scores[best] < min_correlation:
        return 0.0
    return round(best * decim / sr * 1000.0, 2)


def align(take: Audio, latency_ms: float) -> Audio:
    shift = int(take.sr * latency_ms / 1000.0)
    if shift > 0:
        data = take.data[:, shift:]
    elif shift < 0:
        data = np.concatenate([np.zeros((take.channels, -shift), dtype=np.float32), take.data], axis=1)
    else:
        data = take.data
    return Audio(data, take.sr)


def _deesser(x: np.ndarray, sr: int, amount_db: float) -> np.ndarray:
    """Динамическое ослабление сибилянтов в 5–9 кГц."""
    from .dsp import bandpass, envelope

    sib = bandpass(x, 5000, 9000, sr)
    env = envelope(sib if sib.ndim > 1 else sib[None, :], sr, 2.0, 40.0)
    thresh = np.percentile(env, 90) + 1e-6
    reduction = np.clip((env - thresh) / thresh, 0, 1) * db_to_lin(amount_db)
    gain = 1.0 - reduction
    return (x - sib * (1.0 - gain)).astype(np.float32)


def _detect_f0(frame: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 900.0) -> float:
    """Автокорреляционный детектор основного тона (для автотюна)."""
    frame = frame - frame.mean()
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    lag_min, lag_max = int(sr / fmax), min(int(sr / fmin), len(corr) - 1)
    if lag_max <= lag_min:
        return 0.0
    seg = corr[lag_min:lag_max]
    lag = int(np.argmax(seg)) + lag_min
    if corr[lag] < corr[0] * 0.3:
        return 0.0
    return sr / lag


def autotune(audio: Audio, key: str, mode: str, strength: float = 0.6,
             frame_ms: float = 80.0) -> Audio:
    """Мягкое подтягивание к нотам тональности (покадровый питч-шифт)."""
    if strength <= 0:
        return audio
    sr = audio.sr
    frame = int(sr * frame_ms / 1000.0)
    hop = frame // 2
    mono = audio.mono()
    root_pc = PITCH_NAMES.index(key) if key in PITCH_NAMES else 9
    scale = [(root_pc + s) % 12 for s in SCALES.get(mode, SCALES["minor"])]

    window = np.hanning(frame).astype(np.float32)
    out = np.zeros(len(mono) + frame, dtype=np.float32)
    win_sum = np.zeros_like(out)
    for start in range(0, max(len(mono) - frame, 1), hop):
        seg = mono[start: start + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        f0 = _detect_f0(seg, sr)
        shifted = seg
        if f0 > 0:
            midi = 69 + 12 * np.log2(f0 / 440.0)
            pc = midi % 12
            target_pc = min(scale, key=lambda s: min(abs(pc - s), 12 - abs(pc - s)))
            diff = target_pc - pc
            if diff > 6:
                diff -= 12
            elif diff < -6:
                diff += 12
            correction = float(np.clip(diff, -1.5, 1.5)) * strength
            if abs(correction) > 0.02:
                shifted = pitch_shift(seg, correction, sr)
                shifted = np.resize(shifted, frame)
        out[start: start + frame] += shifted * window
        win_sum[start: start + frame] += window
    out /= np.maximum(win_sum, 1e-6)
    out = out[: audio.n_samples]
    return Audio(np.repeat(out[None, :], audio.channels, axis=0), sr)


def process_take(take: Audio, style: str = "rap", analysis: TrackAnalysis | None = None,
                 reference: Audio | None = None, autotune_strength: float | None = None,
                 gender: str = "male", target_gender: str | None = None,
                 preset_overrides: dict | None = None) -> VocalTakeResult:
    """Полная автоматическая обработка записанного дубля."""
    preset = PRESETS.get(style, PRESETS["rap"])
    if preset_overrides:
        preset = VocalChainPreset(**{**preset.__dict__, **preset_overrides})

    notes: list[str] = []
    sr = take.sr
    latency = 0.0
    if reference is not None:
        latency = estimate_latency(take, reference)
        if abs(latency) > 5:
            take = align(take, latency)
            notes.append(f"Компенсирована задержка записи: {latency:.0f} мс.")

    x = take.stereo().data.copy()

    if target_gender and target_gender != gender:
        p_shift, f_shift = GENDER_SHIFT.get((gender, target_gender), (0.0, 0.0))
        x = pitch_shift(x, p_shift, sr)
        x = formant_shift(x, f_shift, sr)
        notes.append(f"Голос перекрашен: {gender} → {target_gender}.")

    x = highpass(x, preset.hpf, sr)
    x = gate(x, threshold_db=preset.gate_db, sr=sr)
    x = _deesser(x, sr, preset.deess_db)
    x = peaking_eq(x, 250.0, preset.body_db, q=0.9, sr=sr)
    x = peaking_eq(x, 400.0, -2.0, q=1.2, sr=sr)          # убираем «картон»
    x = peaking_eq(x, 3500.0, preset.presence_db, q=1.0, sr=sr)
    x = compressor(x, threshold_db=preset.comp_threshold, ratio=preset.comp_ratio,
                   attack_ms=6, release_ms=90, makeup_db=3.0, sr=sr)
    if preset.drive > 1.0:
        x = 0.75 * x + 0.25 * waveshape(x, drive=preset.drive, mode="tube")
    x = compressor(x, threshold_db=-12, ratio=2.5, attack_ms=25, release_ms=180, sr=sr)

    result = Audio(x, sr)

    strength = preset.autotune if autotune_strength is None else autotune_strength
    if strength and analysis is not None:
        result = autotune(result, analysis.key, analysis.mode, strength)
        notes.append(f"Автотюн по тональности {analysis.key_name} (сила {strength:.1f}).")

    if preset.doubling > 0:
        result = Audio(_double(result.data, sr, preset.doubling), sr)
        notes.append("Добавлен дабл-трек.")

    wet = delay_fx(result.data, sr, time_s=0.26, feedback=0.28, mix_amt=preset.delay_mix)
    wet = reverb(wet, sr, decay_s=1.5, mix_amt=preset.reverb_mix)
    gain = db_to_lin(preset.target_db - rms_db(wet))
    wet = wet * gain
    wet = limiter(wet, ceiling_db=-1.5, sr=sr)          # без клиппинга в шину
    if float(np.max(np.abs(wet))) > db_to_lin(-1.0):
        wet = normalize_peak(wet, -1.0)
    out = Audio(wet.astype(np.float32), sr)
    peak = 20 * np.log10(max(out.peak(), 1e-9))
    return VocalTakeResult(audio=out, latency_ms=latency, peak_db=float(peak), notes=notes)


def _double(x: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """Дабл: копия со сдвигом ~18 мс и лёгкой расстройкой, разведённая по сторонам."""
    delay = int(sr * 0.018)
    dbl = pitch_shift(x, 0.12, sr)
    dbl = np.concatenate([np.zeros((x.shape[0], delay), dtype=np.float32), dbl], axis=1)[:, : x.shape[1]]
    left = x[0] + dbl[0] * amount
    right = x[1] + dbl[min(1, dbl.shape[0] - 1)] * amount * 0.9
    return np.stack([left, right]).astype(np.float32) / (1.0 + amount * 0.6)


def harmony(audio: Audio, semitones: float = 7.0, level_db: float = -9.0,
            formants: float = 0.0) -> Audio:
    """Гармония/бэк-вокал: копия голоса, сдвинутая по высоте."""
    shifted = pitch_shift(audio.data, semitones, audio.sr)
    if formants:
        shifted = formant_shift(shifted, formants, audio.sr)
    # фазовый вокодер умеет «выстрелить» по амплитуде — приводим к уровню оригинала
    shifted = normalize_active_rms(shifted, rms_db(audio.data))
    shifted = lowpass(shifted, 9000, audio.sr) * db_to_lin(level_db)
    return Audio(np.asarray(shifted, dtype=np.float32), audio.sr)


def build_vocal_bus(takes: list[Audio], pans: list[float] | None = None) -> Audio:
    """Складывает лид, даблы и гармонии в одну вокальную шину."""
    if not takes:
        return Audio(np.zeros((2, 1), dtype=np.float32))
    sr = takes[0].sr
    n = max(t.n_samples for t in takes)
    acc = np.zeros((2, n), dtype=np.float32)
    for i, take in enumerate(takes):
        data = take.stereo().data
        if pans and i < len(pans):
            data = pan(data.mean(axis=0), pans[i])
        acc[:, : data.shape[1]] += data
    return Audio(acc, sr)
