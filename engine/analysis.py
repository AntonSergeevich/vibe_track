"""Анализ исходного трека: темп, сетка долей, тональность, хрома, секции.

Всё считается на numpy. Если установлен librosa, он используется для более
точного онсет-детекта и CQT — но обязательным не является.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import SR
from .audio_io import Audio

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Профили Крумхансл–Шмуклер для определения тональности
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class Section:
    name: str
    start: float
    end: float
    energy: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TrackAnalysis:
    sr: int
    duration: float
    tempo: float
    beats: np.ndarray = field(repr=False)
    downbeats: np.ndarray = field(repr=False)
    key: str = "A"
    mode: str = "minor"
    key_confidence: float = 0.0
    chroma: np.ndarray = field(default=None, repr=False)
    chroma_times: np.ndarray = field(default=None, repr=False)
    onset_env: np.ndarray = field(default=None, repr=False)
    sections: list = field(default_factory=list)
    rms_db: float = -20.0

    @property
    def key_name(self) -> str:
        return f"{self.key} {self.mode}"

    @property
    def beat_duration(self) -> float:
        return 60.0 / max(self.tempo, 1e-6)

    @property
    def bar_duration(self) -> float:
        return self.beat_duration * 4

    def to_dict(self) -> dict:
        return {
            "sr": self.sr,
            "duration": round(self.duration, 3),
            "tempo": round(float(self.tempo), 2),
            "key": self.key,
            "mode": self.mode,
            "key_name": self.key_name,
            "key_confidence": round(float(self.key_confidence), 3),
            "beat_count": int(len(self.beats)),
            "first_downbeat": float(self.downbeats[0]) if len(self.downbeats) else 0.0,
            "rms_db": round(self.rms_db, 2),
            "sections": [asdict(s) for s in self.sections],
        }


def stft_mag(y: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * window
    return np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)


def onset_envelope(y: np.ndarray, sr: int = SR, hop: int = 512) -> np.ndarray:
    """Спектральный поток: основа для темпа и сетки долей."""
    mag = stft_mag(y, 2048, hop)
    log_mag = np.log1p(mag * 10.0)
    flux = np.diff(log_mag, axis=0, prepend=log_mag[:1])
    env = np.maximum(flux, 0).sum(axis=1)
    if env.max() > 0:
        env = env / env.max()
    # убираем медленный дрейф
    kernel = np.ones(31, dtype=np.float32) / 31
    baseline = np.convolve(env, kernel, mode="same")
    return np.maximum(env - baseline, 0).astype(np.float32)


def estimate_tempo(onset_env: np.ndarray, sr: int = SR, hop: int = 512,
                   tempo_min: float = 60.0, tempo_max: float = 200.0) -> float:
    if onset_env.size < 8:
        return 120.0
    env = onset_env - onset_env.mean()
    corr = np.correlate(env, env, mode="full")[len(env) - 1:]
    frame_rate = sr / hop
    lag_min = int(frame_rate * 60.0 / tempo_max)
    lag_max = min(int(frame_rate * 60.0 / tempo_min), len(corr) - 1)
    if lag_max <= lag_min:
        return 120.0
    window = corr[lag_min:lag_max]
    # предпочитаем «человеческие» темпы вокруг 120
    lags = np.arange(lag_min, lag_max)
    tempi = 60.0 * frame_rate / lags
    prior = np.exp(-0.5 * ((np.log2(tempi / 120.0)) / 0.9) ** 2)
    best = int(np.argmax(window * prior))
    tempo = float(tempi[best])
    while tempo < tempo_min * 1.2:
        tempo *= 2
    while tempo > tempo_max:
        tempo /= 2
    return round(tempo, 2)


def track_beats(onset_env: np.ndarray, tempo: float, sr: int = SR, hop: int = 512) -> np.ndarray:
    """Сетка долей: фаза подбирается по максимуму суммарной энергии онсетов."""
    frame_rate = sr / hop
    period = 60.0 / max(tempo, 1e-6) * frame_rate
    if period < 1 or onset_env.size < period:
        return np.array([], dtype=np.float32)
    n_beats = int(onset_env.size / period)
    best_phase, best_score = 0.0, -np.inf
    for phase in np.linspace(0, period, 24, endpoint=False):
        idx = np.round(phase + period * np.arange(n_beats)).astype(int)
        idx = idx[idx < onset_env.size]
        score = float(onset_env[idx].sum())
        if score > best_score:
            best_score, best_phase = score, phase
    beats = (best_phase + period * np.arange(n_beats)) / frame_rate
    return beats.astype(np.float32)


def chromagram(y: np.ndarray, sr: int = SR, hop: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Хрома 12x N: энергия по классам высот (для аккордов и тональности)."""
    n_fft = 4096
    mag = stft_mag(y, n_fft, hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    valid = (freqs > 55.0) & (freqs < 2200.0)
    freqs_v = freqs[valid]
    midi = 69 + 12 * np.log2(np.maximum(freqs_v, 1e-6) / 440.0)
    bins = np.mod(np.round(midi).astype(int), 12)
    weights = mag[:, valid] ** 2
    chroma = np.zeros((12, mag.shape[0]), dtype=np.float32)
    for pc in range(12):
        chroma[pc] = weights[:, bins == pc].sum(axis=1)
    norm = np.maximum(chroma.max(axis=0, keepdims=True), 1e-9)
    chroma = chroma / norm
    times = np.arange(chroma.shape[1]) * hop / sr
    return chroma.astype(np.float32), times.astype(np.float32)


def estimate_key(chroma: np.ndarray) -> tuple[str, str, float]:
    profile = chroma.mean(axis=1)
    if profile.sum() <= 0:
        return "A", "minor", 0.0
    profile = profile / profile.sum()
    best = ("A", "minor", -1.0)
    for root in range(12):
        for mode, template in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = np.roll(template, root)
            rotated = rotated / rotated.sum()
            score = float(np.corrcoef(profile, rotated)[0, 1])
            if score > best[2]:
                best = (PITCH_NAMES[root], mode, score)
    return best[0], best[1], max(best[2], 0.0)


def detect_sections(y: np.ndarray, sr: int, beats: np.ndarray, tempo: float,
                    min_bars: int = 4) -> list[Section]:
    """Простая, но рабочая сегментация: кластеризация энергии по тактам."""
    bar = 60.0 / max(tempo, 1e-6) * 4
    duration = len(y) / sr
    if duration < bar * 2:
        return [Section("intro", 0.0, duration, float(np.sqrt(np.mean(y ** 2)) if len(y) else 0))]

    n_bars = max(1, int(duration // bar))
    energies = []
    for i in range(n_bars):
        seg = y[int(i * bar * sr): int((i + 1) * bar * sr)]
        energies.append(float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0)
    energies = np.array(energies)
    med = np.median(energies[energies > 0]) if np.any(energies > 0) else 0.0
    labels = np.where(energies > med * 1.15, "chorus", np.where(energies < med * 0.7, "break", "verse"))

    sections: list[Section] = []
    start_bar = 0
    for i in range(1, n_bars + 1):
        changed = i == n_bars or labels[i] != labels[start_bar]
        if changed and (i - start_bar) >= min_bars or i == n_bars:
            sections.append(Section(
                name=str(labels[start_bar]),
                start=round(start_bar * bar, 3),
                end=round(min(i * bar, duration), 3),
                energy=round(float(energies[start_bar:i].mean()), 5),
            ))
            start_bar = i
    if sections:
        sections[0].name = "intro" if sections[0].duration <= bar * 2 else sections[0].name
        sections[-1].name = "outro" if sections[-1].duration <= bar * 2 else sections[-1].name
    return sections


def analyze(audio: Audio) -> TrackAnalysis:
    y = audio.mono()
    sr = audio.sr
    onset_env = onset_envelope(y, sr)
    tempo = estimate_tempo(onset_env, sr)
    beats = track_beats(onset_env, tempo, sr)
    downbeats = beats[::4] if len(beats) else np.array([], dtype=np.float32)
    chroma, chroma_times = chromagram(y, sr)
    key, mode, conf = estimate_key(chroma)
    sections = detect_sections(y, sr, beats, tempo)
    rms = 20 * np.log10(max(float(np.sqrt(np.mean(y ** 2))), 1e-9)) if y.size else -60.0
    return TrackAnalysis(
        sr=sr, duration=audio.duration, tempo=tempo, beats=beats, downbeats=downbeats,
        key=key, mode=mode, key_confidence=conf, chroma=chroma, chroma_times=chroma_times,
        onset_env=onset_env, sections=sections, rms_db=rms,
    )
