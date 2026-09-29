"""Синтез инструментов ню-метала.

Всё генерируется на numpy: аддитивный синтез струнных с последующей
эмуляцией усилителя/кабинета и синтезированная ударная установка.
Никаких сэмплов и SoundFont'ов не требуется, но если в MEDIA есть
пользовательские сэмплы — их можно подмешать через SampleKit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import SR
from .dsp import bandpass, highpass, lowpass, peaking_eq, waveshape, compressor


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _adsr(n: int, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    a = min(int(attack * sr), n)
    d = min(int(decay * sr), max(n - a, 0))
    r = min(int(release * sr), max(n - a - d, 0))
    s = max(n - a - d - r, 0)
    env = np.concatenate([
        np.linspace(0, 1, a, dtype=np.float32),
        np.linspace(1, sustain, d, dtype=np.float32),
        np.full(s, sustain, dtype=np.float32),
        np.linspace(sustain, 0, r, dtype=np.float32),
    ])
    return env[:n] if len(env) >= n else np.pad(env, (0, n - len(env)))


def _partials(freq: float, n: int, sr: int, n_partials: int = 14,
              detune_cents: float = 6.0, seed: int = 0) -> np.ndarray:
    """Аддитивная пила из двух расстроенных голосов — «толстый» струнный тон."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, dtype=np.float32)
    for voice, cents in enumerate((-detune_cents, detune_cents)):
        f0 = freq * 2 ** (cents / 1200.0)
        for k in range(1, n_partials + 1):
            f = f0 * k
            if f > sr * 0.45:
                break
            phase = rng.uniform(0, 2 * np.pi)
            out += (1.0 / k) * np.sin(2 * np.pi * f * t + phase)
    return (out / max(np.max(np.abs(out)), 1e-6)).astype(np.float32)


def ks_string(freq: float, dur: float, sr: int = SR, decay: float = 1.2,
              brightness: float = 4000.0, seed: int = 0) -> np.ndarray:
    """Струна по модели Карплуса-Стронга.

    Возбуждаем период шумом (щипок медиатором) и гоняем его по кольцевому
    буферу, усредняя соседние отсчёты — это физика реальной струны: высокие
    гармоники затухают быстрее низких. Аддитивный синтез так не умеет, оттого
    и звучал как синтезатор, а не как гитара.

    Считаем периодами, а не отсчётами: numpy делает всё кольцо за одну
    операцию, поэтому нота рендерится за микросекунды.
    """
    n = max(int(dur * sr), 64)
    period = max(8, int(round(sr / max(freq, 20.0))))
    rng = np.random.default_rng(seed)

    excitation = rng.standard_normal(period).astype(np.float32)
    excitation = lowpass(excitation, min(brightness, sr * 0.45), sr)
    excitation -= excitation.mean()          # без постоянной составляющей

    # за один оборот кольца амплитуда падает в g раз; отсюда время затухания
    periods_to_silence = max(decay * freq, 1.0)
    g = min(10.0 ** (-3.0 / periods_to_silence), 0.9995)

    blocks, produced = [], 0
    buf = excitation
    while produced < n:
        blocks.append(buf)
        produced += period
        buf = (g * 0.5) * (buf + np.roll(buf, 1))
    out = np.concatenate(blocks)[:n]
    peak = float(np.max(np.abs(out)))
    return (out / peak).astype(np.float32) if peak > 1e-9 else out


@dataclass
class GuitarTone:
    """Настройки гитарного тракта."""
    drive: float = 12.0
    palm_mute: bool = True
    cab_low: float = 90.0
    cab_high: float = 5200.0
    presence_db: float = 3.0
    scoop_db: float = -3.0        # характерный ню-метал «провал» в середине
    noise: float = 0.02


def pluck(freq: float, dur: float, sr: int = SR, velocity: float = 1.0,
          palm_mute: bool = False, seed: int = 0, bright: float = 1.0) -> np.ndarray:
    """Один щипок струны до усилителя."""
    n = max(int(dur * sr), 64)
    if palm_mute:
        # заглушённая ладонью струна: гаснет за доли секунды и заметно глуше
        body = ks_string(freq, dur, sr, decay=0.13, brightness=1400 * bright, seed=seed)
        env = _adsr(n, sr, 0.001, min(0.07, dur * 0.5), 0.10, min(0.05, dur * 0.4))
    else:
        body = ks_string(freq, dur, sr, decay=1.6, brightness=3200 * bright, seed=seed)
        env = _adsr(n, sr, 0.002, min(0.30, dur * 0.6), 0.60, min(0.12, dur * 0.35))

    # удар медиатора по струне
    rng = np.random.default_rng(seed + 1)
    click_n = min(int(0.004 * sr), n)
    click = np.zeros(n, dtype=np.float32)
    click[:click_n] = rng.standard_normal(click_n) * np.linspace(1, 0, click_n)
    click = highpass(click, 2000, sr) * 0.2
    return ((body * env + click) * velocity).astype(np.float32)


_CAB_CACHE: dict[tuple, np.ndarray] = {}


def cab_impulse(sr: int = SR, low: float = 85.0, high: float = 5200.0,
                seed: int = 17) -> np.ndarray:
    """Импульсная характеристика гитарного кабинета.

    Реальный кабинет — это не фильтр, а короткий отклик со своими
    резонансами и отражениями внутри корпуса. Синтезируем его: затухающий
    шум с резонансом корпуса и подъёмом на «разрыве» диффузора.

    Если в VIBETRACK_CAB_IR указан путь к настоящему импульсу (их полно
    бесплатных), берём его — это ещё один слышимый шаг вперёд.
    """
    key = (sr, low, high, seed)
    if key in _CAB_CACHE:
        return _CAB_CACHE[key]

    custom = os.getenv("VIBETRACK_CAB_IR")
    if custom and os.path.exists(custom):
        try:
            import soundfile as sf

            data, ir_sr = sf.read(custom, dtype="float32", always_2d=True)
            ir = data[:, 0]
            if ir_sr != sr:  # импульс короткий, линейной интерполяции хватает
                idx = np.linspace(0, len(ir) - 1, int(len(ir) * sr / ir_sr))
                ir = np.interp(idx, np.arange(len(ir)), ir).astype(np.float32)
            ir = ir[: int(sr * 0.05)]
            _CAB_CACHE[key] = ir / max(float(np.max(np.abs(ir))), 1e-9)
            return _CAB_CACHE[key]
        except Exception:  # некорректный файл не должен ломать рендер
            pass

    n = int(sr * 0.012)
    t = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(seed)
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-t * 900.0)
    ir[0] += 1.0                                    # прямой звук
    ir = bandpass(ir, low, high, sr)
    ir = peaking_eq(ir, 120.0, 5.0, q=1.4, sr=sr)   # резонанс корпуса
    ir = peaking_eq(ir, 2400.0, 1.0, q=1.8, sr=sr)  # разрыв диффузора
    ir = peaking_eq(ir, 6500.0, -8.0, q=1.0, sr=sr)  # завал сверху
    _CAB_CACHE[key] = ir / max(float(np.max(np.abs(ir))), 1e-9)
    return _CAB_CACHE[key]


def amp(signal: np.ndarray, tone: GuitarTone, sr: int = SR) -> np.ndarray:
    """Гитарный тракт: два каскада усиления и кабинет.

    Один каскад искажения звучит плоско — настоящий усилитель наращивает
    гейн ступенями, с коррекцией между ними. Отсюда «мясо» вместо жужжания.
    """
    x = highpass(signal, tone.cab_low, sr)
    x = peaking_eq(x, 550.0, -5.0, q=1.4, sr=sr)     # «каша» дроп-строя живёт выше, чем кажется

    x = np.tanh(x * (tone.drive * 0.45)).astype(np.float32)   # первый каскад
    x = peaking_eq(x, 800.0, tone.scoop_db, q=0.9, sr=sr)
    x = lowpass(x, 7500.0, sr)
    x = np.tanh(x * (tone.drive * 0.35)).astype(np.float32)   # второй каскад

    try:
        from scipy.signal import oaconvolve

        ir = cab_impulse(sr, tone.cab_low, tone.cab_high)
        x = oaconvolve(x, ir)[: len(x)].astype(np.float32)
    except Exception:                                 # без scipy остаёмся на фильтрах
        x = lowpass(x, tone.cab_high, sr)

    x = peaking_eq(x, 3200.0, tone.presence_db, q=1.1, sr=sr)
    x = peaking_eq(x, 160.0, 6.0, q=0.9, sr=sr)      # «мясо» чага живёт здесь
    peak = float(np.max(np.abs(x)))
    if peak > 1e-9:
        x = x / peak
    return compressor(x, threshold_db=-20, ratio=3.5, attack_ms=3, release_ms=80, sr=sr)


def bass_note(freq: float, dur: float, sr: int = SR, velocity: float = 1.0,
              drive: float = 3.0, seed: int = 0) -> np.ndarray:
    n = max(int(dur * sr), 64)
    t = np.arange(n, dtype=np.float32) / sr
    sub = np.sin(2 * np.pi * freq * t)
    saw = _partials(freq, n, sr, n_partials=10, detune_cents=3.0, seed=seed)
    env = _adsr(n, sr, 0.005, min(0.18, dur * 0.5), 0.7, min(0.08, dur * 0.3))
    body = (0.6 * sub + 0.5 * saw) * env * velocity
    grit = waveshape(highpass(body, 300, sr), drive=drive, mode="tube") * 0.35
    return (lowpass(body, 3500, sr) + grit).astype(np.float32)


# ---------------------------------------------------------------- ударные
def kick(sr: int = SR, velocity: float = 1.0, decay: float = 0.32) -> np.ndarray:
    """Бочка из трёх слоёв: подтон, тело и удар колотушки."""
    n = int(decay * sr)
    t = np.arange(n, dtype=np.float32) / sr

    sweep = 150.0 * np.exp(-t * 32.0) + 48.0          # тело с падением высоты
    body = np.sin(2 * np.pi * np.cumsum(sweep) / sr) * np.exp(-t * 8.0)
    sub = np.sin(2 * np.pi * 45.0 * t) * np.exp(-t * 5.0) * 0.7   # подтон

    rng = np.random.default_rng(3)
    click = np.zeros(n, dtype=np.float32)
    cn = int(0.006 * sr)
    click[:cn] = rng.standard_normal(cn) * np.exp(-np.linspace(0, 6, cn))
    click = bandpass(click, 1800, 6000, sr) * 0.45     # стук колотушки о пластик

    mix = body + sub + click
    mix = np.tanh(mix * 1.6) * 0.7                     # лёгкое насыщение склеивает слои
    return (mix * velocity).astype(np.float32)


def snare(sr: int = SR, velocity: float = 1.0, decay: float = 0.22) -> np.ndarray:
    """Рабочий: тело пластика, шум пружин и отдельный «щелчок» сверху.

    Один шумовой слой звучит как шипение — разделение на тело и щелчок с
    разной скоростью затухания и даёт узнаваемый удар.
    """
    n = int(decay * sr)
    t = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(n).astype(np.float32)

    body = bandpass(noise, 180, 1200, sr) * np.exp(-t * 22.0)      # пластик
    wires = bandpass(noise, 1200, 7000, sr) * np.exp(-t * 14.0)    # пружины
    crack = highpass(noise, 4000, sr) * np.exp(-t * 90.0) * 0.8    # щелчок
    tone = (0.5 * np.sin(2 * np.pi * 195 * t) +
            0.3 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t * 26.0)

    mix = body * 0.9 + wires * 0.7 + crack + tone
    return (np.tanh(mix * 1.3) * 0.75 * velocity).astype(np.float32)


def hihat(sr: int = SR, velocity: float = 1.0, open_hat: bool = False) -> np.ndarray:
    decay = 0.35 if open_hat else 0.06
    n = int(decay * sr)
    rng = np.random.default_rng(7)
    noise = highpass(rng.standard_normal(n).astype(np.float32), 7000, sr)
    env = np.exp(-np.arange(n, dtype=np.float32) / sr * (9.0 if open_hat else 60.0))
    return (noise * env * velocity * 0.5).astype(np.float32)


def crash(sr: int = SR, velocity: float = 1.0, decay: float = 1.8) -> np.ndarray:
    n = int(decay * sr)
    rng = np.random.default_rng(9)
    noise = bandpass(rng.standard_normal(n).astype(np.float32), 2500, 14000, sr)
    env = np.exp(-np.arange(n, dtype=np.float32) / sr * 2.2)
    return (noise * env * velocity * 0.45).astype(np.float32)


def tom(sr: int = SR, velocity: float = 1.0, pitch: float = 140.0, decay: float = 0.45) -> np.ndarray:
    n = int(decay * sr)
    t = np.arange(n, dtype=np.float32) / sr
    freq = pitch * (1 + 0.6 * np.exp(-t * 20))
    body = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 6.0)
    rng = np.random.default_rng(13)
    skin = bandpass(rng.standard_normal(n).astype(np.float32), 200, 3000, sr) * np.exp(-t * 30) * 0.3
    return ((body + skin) * velocity).astype(np.float32)


def bongo(sr: int = SR, velocity: float = 1.0, pitch: float = 300.0) -> np.ndarray:
    """Перкуссия в духе Korn — «живая» подложка под грув."""
    return tom(sr, velocity * 0.8, pitch=pitch, decay=0.22)


DRUM_VOICES = {
    "kick": lambda sr, v: kick(sr, v),
    "snare": lambda sr, v: snare(sr, v),
    "hat": lambda sr, v: hihat(sr, v, open_hat=False),
    "openhat": lambda sr, v: hihat(sr, v, open_hat=True),
    "crash": lambda sr, v: crash(sr, v),
    "tom_hi": lambda sr, v: tom(sr, v, pitch=190),
    "tom_lo": lambda sr, v: tom(sr, v, pitch=110),
    "bongo": lambda sr, v: bongo(sr, v),
}


# ------------------------------------------------------- клавиши и эффекты
def pad_note(freq: float, dur: float, sr: int = SR, velocity: float = 0.6,
             seed: int = 0) -> np.ndarray:
    n = max(int(dur * sr), 64)
    body = _partials(freq, n, sr, n_partials=8, detune_cents=12.0, seed=seed)
    env = _adsr(n, sr, min(0.35, dur * 0.3), 0.2, 0.8, min(0.6, dur * 0.4))
    return (lowpass(body, 2200, sr) * env * velocity).astype(np.float32)


def scratch(sr: int = SR, velocity: float = 1.0, dur: float = 0.35, seed: int = 0) -> np.ndarray:
    """Скретч тёрнтейбла: шум с быстрой модуляцией высоты."""
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(seed + 21)
    base = bandpass(rng.standard_normal(n).astype(np.float32), 400, 6000, sr)
    lfo = np.sin(2 * np.pi * 6.5 * t) * 0.5 + 0.5
    idx = np.clip((np.cumsum(0.4 + lfo * 1.6) ).astype(int), 0, n - 1)
    env = np.exp(-t * 4.0)
    return (base[idx] * env * velocity * 0.6).astype(np.float32)


def synth_stab(freq: float, dur: float, sr: int = SR, velocity: float = 0.8, seed: int = 0) -> np.ndarray:
    n = max(int(dur * sr), 64)
    t = np.arange(n, dtype=np.float32) / sr
    sig = np.sign(np.sin(2 * np.pi * freq * t)) * 0.4 + _partials(freq, n, sr, seed=seed) * 0.6
    cutoff = 400 + 4000 * np.exp(-t * 12)
    sig = lowpass(sig, float(cutoff.mean()), sr)
    env = _adsr(n, sr, 0.003, min(0.15, dur * 0.5), 0.3, min(0.1, dur * 0.3))
    return (sig * env * velocity).astype(np.float32)


@dataclass
class SampleKit:
    """Подмешивание пользовательских сэмплов вместо синтеза (опционально)."""
    samples: dict[str, np.ndarray] = field(default_factory=dict)

    def get(self, name: str) -> np.ndarray | None:
        return self.samples.get(name)
