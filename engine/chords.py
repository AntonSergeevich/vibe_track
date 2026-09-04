"""Распознавание аккордов и построение аккордовой схемы.

Хрома + шаблоны аккордов (мажор, минор, power-chord, sus, 7-е ступени),
сглаживание по тактам, вывод в виде списка (время, длительность, аккорд).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .analysis import PITCH_NAMES, TrackAnalysis

# Шаблоны интервалов от основного тона
CHORD_TEMPLATES: dict[str, tuple[int, ...]] = {
    "5": (0, 7),            # power-chord — основа ню-метала
    "": (0, 4, 7),          # мажор
    "m": (0, 3, 7),         # минор
    "sus4": (0, 5, 7),
    "sus2": (0, 2, 7),
    "7": (0, 4, 7, 10),
    "m7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11),
    "dim": (0, 3, 6),
}

# Веса: power-chord часто «выигрывает» у трезвучий, слегка штрафуем его,
# чтобы не терять мажор/минор там, где терция реально звучит.
_QUALITY_BIAS = {"5": 0.92, "": 1.0, "m": 1.0, "sus4": 0.94, "sus2": 0.93,
                 "7": 0.97, "m7": 0.97, "maj7": 0.95, "dim": 0.9}


@dataclass
class ChordEvent:
    start: float
    end: float
    root: int          # 0..11, C=0
    quality: str
    confidence: float = 0.0

    @property
    def name(self) -> str:
        return f"{PITCH_NAMES[self.root % 12]}{self.quality}"

    @property
    def duration(self) -> float:
        return self.end - self.start

    def pitches(self, octave: int = 3) -> list[int]:
        base = 12 * (octave + 1) + self.root
        return [base + iv for iv in CHORD_TEMPLATES.get(self.quality, (0, 7))]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["name"] = self.name
        d["start"] = round(self.start, 3)
        d["end"] = round(self.end, 3)
        d["confidence"] = round(float(self.confidence), 3)
        return d


def _template_matrix() -> tuple[np.ndarray, list[tuple[int, str]]]:
    rows, labels = [], []
    for quality, intervals in CHORD_TEMPLATES.items():
        for root in range(12):
            vec = np.zeros(12, dtype=np.float32)
            for i, iv in enumerate(intervals):
                vec[(root + iv) % 12] = 1.0 if i == 0 else 0.85
            vec /= np.linalg.norm(vec)
            rows.append(vec * _QUALITY_BIAS[quality])
            labels.append((root, quality))
    return np.stack(rows), labels


_TEMPLATES, _LABELS = _template_matrix()


def recognize(analysis: TrackAnalysis, resolution: str = "bar") -> list[ChordEvent]:
    """Распознаёт аккорды по сетке долей/тактов исходного трека."""
    chroma, times = analysis.chroma, analysis.chroma_times
    if chroma is None or chroma.size == 0:
        return []

    grid = _build_grid(analysis, resolution)
    events: list[ChordEvent] = []
    for start, end in grid:
        mask = (times >= start) & (times < end)
        if not np.any(mask):
            continue
        vec = chroma[:, mask].mean(axis=1)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            continue
        scores = _TEMPLATES @ (vec / norm)
        best = int(np.argmax(scores))
        root, quality = _LABELS[best]
        events.append(ChordEvent(start=float(start), end=float(end), root=root,
                                 quality=quality, confidence=float(scores[best])))
    return _merge_repeats(events)


def _build_grid(analysis: TrackAnalysis, resolution: str) -> list[tuple[float, float]]:
    beats = analysis.beats
    if len(beats) < 2:
        step = analysis.bar_duration if resolution == "bar" else analysis.beat_duration
        n = max(1, int(analysis.duration / step))
        return [(i * step, min((i + 1) * step, analysis.duration)) for i in range(n)]
    stride = 4 if resolution == "bar" else (2 if resolution == "half-bar" else 1)
    edges = list(beats[::stride]) + [analysis.duration]
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)
            if edges[i + 1] > edges[i]]


def _merge_repeats(events: list[ChordEvent]) -> list[ChordEvent]:
    merged: list[ChordEvent] = []
    for ev in events:
        if merged and merged[-1].root == ev.root and merged[-1].quality == ev.quality:
            merged[-1].end = ev.end
            merged[-1].confidence = max(merged[-1].confidence, ev.confidence)
        else:
            merged.append(ev)
    return merged


def chord_at(events: list[ChordEvent], t: float) -> ChordEvent | None:
    for ev in events:
        if ev.start <= t < ev.end:
            return ev
    return events[-1] if events else None


def transpose(events: list[ChordEvent], semitones: int) -> list[ChordEvent]:
    return [ChordEvent(e.start, e.end, (e.root + semitones) % 12, e.quality, e.confidence)
            for e in events]


def to_power_chords(events: list[ChordEvent]) -> list[ChordEvent]:
    """Ню-метал говорит квинтами: превращаем трезвучия в power-chords."""
    return [ChordEvent(e.start, e.end, e.root, "5", e.confidence) for e in events]


def progression_string(events: list[ChordEvent], limit: int = 16) -> str:
    return " | ".join(e.name for e in events[:limit])
