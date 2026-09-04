"""Синтез инструментов ню-метала.

Всё генерируется на numpy: аддитивный синтез струнных с последующей
эмуляцией усилителя/кабинета и синтезированная ударная установка.
Никаких сэмплов и SoundFont'ов не требуется, но если в MEDIA есть
пользовательские сэмплы — их можно подмешать через SampleKit.
"""
from __future__ import annotations

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


@dataclass
class GuitarTone:
    """Настройки гитарного тракта."""
    drive: float = 12.0
    palm_mute: bool = True
    cab_low: float = 90.0
    cab_high: float = 5200.0
    presence_db: float = 4.0
    scoop_db: float = -3.0        # характерный ню-метал «провал» в середине
    noise: float = 0.02


def pluck(freq: float, dur: float, sr: int = SR, velocity: float = 1.0,
          palm_mute: bool = False, seed: int = 0, bright: float = 1.0) -> np.ndarray:
    """Один щипок струны (до усилителя)."""
    n = max(int(dur * sr), 64)
    body = _partials(freq, n, sr, seed=seed)
    if palm_mute:
        env = _adsr(n, sr, 0.002, min(0.09, dur * 0.5), 0.12, min(0.06, dur * 0.4))
        body = lowpass(body, 2600 * bright, sr)
    else:
        env = _adsr(n, sr, 0.004, min(0.25, dur * 0.6), 0.55, min(0.12, dur * 0.35))
    # шум медиатора
    rng = np.random.default_rng(seed + 1)
    click_n = min(int(0.006 * sr), n)
    click = np.zeros(n, dtype=np.float32)
    click[:click_n] = rng.standard_normal(click_n) * np.linspace(1, 0, click_n)
    click = highpass(click, 1500, sr) * 0.25
    return ((body * env + click) * velocity).astype(np.float32)


def amp(signal: np.ndarray, tone: GuitarTone, sr: int = SR) -> np.ndarray:
    """Эмуляция «стены» гитарного усилителя + кабинета."""
    x = highpass(signal, tone.cab_low, sr)
    x = peaking_eq(x, 800.0, tone.scoop_db, q=0.9, sr=sr)
    x = waveshape(x, drive=tone.drive, mode="tube")
    x = lowpass(x, tone.cab_high, sr)
    x = peaking_eq(x, 3200.0, tone.presence_db, q=1.1, sr=sr)
    x = peaking_eq(x, 120.0, 2.5, q=0.8, sr=sr)
    if tone.noise:
        rng = np.random.default_rng(11)
        x = x + rng.standard_normal(x.shape).astype(np.float32) * tone.noise * 0.01
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
    n = int(decay * sr)
    t = np.arange(n, dtype=np.float32) / sr
    freq = 120.0 * np.exp(-t * 28.0) + 46.0
    phase = 2 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * np.exp(-t * 7.0)
    click = np.zeros(n, dtype=np.float32)
    cn = int(0.004 * sr)
    click[:cn] = np.random.default_rng(3).standard_normal(cn) * np.linspace(1, 0, cn)
    click = highpass(click, 2500, sr) * 0.5
    return ((body + click) * velocity).astype(np.float32)


def snare(sr: int = SR, velocity: float = 1.0, decay: float = 0.22) -> np.ndarray:
    n = int(decay * sr)
    t = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(5)
    noise = bandpass(rng.standard_normal(n).astype(np.float32), 250, 9000, sr)
    tone = 0.5 * np.sin(2 * np.pi * 190 * t) + 0.35 * np.sin(2 * np.pi * 330 * t)
    env = np.exp(-t * 18.0)
    return ((noise * 0.9 + tone) * env * velocity).astype(np.float32)


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
