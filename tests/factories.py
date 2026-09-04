"""Генератор тестового аудио: аккорды Am–F–C–G под ровный бит."""
from __future__ import annotations

import numpy as np
import soundfile as sf

CHORDS = {0: [57, 60, 64], 1: [53, 57, 60], 2: [48, 52, 55], 3: [55, 59, 62]}


def make_track(path: str, seconds: float = 12.0, sr: int = 44100, tempo: float = 120.0,
               with_voice: bool = True) -> str:
    t = np.arange(int(sr * seconds)) / sr
    bar = 60.0 / tempo * 4
    y = np.zeros_like(t)
    for i in range(int(seconds / bar) + 1):
        seg = (t >= i * bar) & (t < (i + 1) * bar)
        for midi in CHORDS[i % 4]:
            freq = 440.0 * 2 ** ((midi - 69) / 12)
            y[seg] += 0.18 * np.sin(2 * np.pi * freq * t[seg])
    beat = 60.0 / tempo
    rng = np.random.default_rng(4)
    for i in range(int(seconds / beat)):
        s = int(i * beat * sr)
        e = min(s + int(0.06 * sr), len(y))
        y[s:e] += rng.standard_normal(e - s) * 0.35 * np.exp(-np.linspace(0, 8, e - s))
    if with_voice:
        f0 = 200 * (1 + 0.02 * np.sin(2 * np.pi * 4 * t))
        y += 0.12 * np.sin(2 * np.pi * np.cumsum(f0) / sr)
    stereo = np.stack([y, y * 0.97]).T.astype(np.float32)
    sf.write(path, stereo, sr)
    return path


def make_voice(path: str, seconds: float = 4.0, sr: int = 44100, f0: float = 165.0) -> str:
    t = np.arange(int(sr * seconds)) / sr
    freq = f0 * (1 + 0.015 * np.sin(2 * np.pi * 5 * t))
    voice = sum((1 / k) * np.sin(2 * np.pi * np.cumsum(freq * k) / sr) for k in range(1, 10))
    voice = (voice * 0.25).astype(np.float32)
    sf.write(path, np.stack([voice, voice]).T, sr)
    return path
