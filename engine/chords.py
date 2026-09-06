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

# Словарь по умолчанию. Расширенный набор (септаккорды, sus) даёт формально
# более точное совпадение с хромой, но человек, снимающий песню, слышит там
# простое трезвучие — и на других сайтах будет написано именно оно.
SIMPLE_QUALITIES = ("", "m")
FULL_QUALITIES = tuple(CHORD_TEMPLATES)

# Трезвучия ступеней: не только какие тона «свои», но и какое у них
# наклонение. Без этого мажорная V ступень легко распознаётся как минорная —
# терцию в плотном миксе почти не слышно, и подсказка тональности решает.
_MAJOR_TRIADS = {0: "", 2: "m", 4: "m", 5: "", 7: "", 9: "m", 11: "dim"}
_MINOR_TRIADS = {0: "m", 2: "dim", 3: "", 5: "m", 7: "m", 8: "", 10: ""}


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


def _template_matrix(qualities=None) -> tuple[np.ndarray, list[tuple[int, str]]]:
    rows, labels = [], []
    for quality in (qualities or tuple(CHORD_TEMPLATES)):
        intervals = CHORD_TEMPLATES[quality]
        for root in range(12):
            vec = np.zeros(12, dtype=np.float32)
            for i, iv in enumerate(intervals):
                vec[(root + iv) % 12] = 1.0 if i == 0 else 0.85
            vec /= np.linalg.norm(vec)
            rows.append(vec * _QUALITY_BIAS[quality])
            labels.append((root, quality))
    return np.stack(rows), labels


def recognize(analysis: TrackAnalysis, resolution: str = "bar", simple: bool = True,
              change_penalty: float = 0.12, key_bonus: float = 0.12,
              min_duration: float | None = None) -> list[ChordEvent]:
    """Распознаёт аккорды по сетке тактов исходного трека.

    Три вещи, которых не хватало наивному подходу «взять лучший шаблон на
    такте», из-за чего аккордов получалось вдвое больше, чем слышит человек:

    * `simple` — словарь из мажора и минора. Септаккорд формально ближе к
      хроме, но в песеннике будет написано трезвучие;
    * `key_bonus` — приоритет ступеням найденной тональности, иначе
      случайный шум уводит в «чужие» аккорды;
    * `change_penalty` — плата за смену аккорда. Динамическое
      программирование выбирает не лучший аккорд на такте, а лучшую
      последовательность целиком, поэтому она перестаёт дёргаться.
    """
    chroma, times = analysis.chroma, analysis.chroma_times
    if chroma is None or chroma.size == 0:
        return []

    qualities = SIMPLE_QUALITIES if simple else FULL_QUALITIES
    templates, labels = _template_matrix(qualities)
    grid = _build_grid(analysis, resolution)

    segments, scores = [], []
    for start, end in grid:
        mask = (times >= start) & (times < end)
        if not np.any(mask):
            continue
        vec = chroma[:, mask].mean(axis=1)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            continue
        segments.append((start, end))
        scores.append(templates @ (vec / norm))
    if not segments:
        return []

    score_matrix = np.stack(scores)
    score_matrix += key_bonus * _key_prior(labels, analysis.key, analysis.mode)
    path = _viterbi(score_matrix, change_penalty)

    events = [ChordEvent(start=float(seg[0]), end=float(seg[1]), root=labels[idx][0],
                         quality=labels[idx][1], confidence=float(score_matrix[i, idx]))
              for i, (seg, idx) in enumerate(zip(segments, path))]
    events = _merge_repeats(events)
    return _drop_short(events, min_duration if min_duration is not None
                       else analysis.bar_duration * 0.75)


def _key_prior(labels: list[tuple[int, str]], key: str, mode: str) -> np.ndarray:
    """Прибавка аккордам тональности: полная — за совпадение и тона, и наклонения."""
    if key not in PITCH_NAMES:
        return np.zeros(len(labels), dtype=np.float32)
    root_pc = PITCH_NAMES.index(key)
    triads = _MINOR_TRIADS if mode == "minor" else _MAJOR_TRIADS
    diatonic = {(root_pc + step) % 12: quality for step, quality in triads.items()}

    prior = np.zeros(len(labels), dtype=np.float32)
    for i, (root, quality) in enumerate(labels):
        expected = diatonic.get(root)
        if expected is None:
            continue
        prior[i] = 1.0 if quality == expected else 0.35   # «свой» тон, чужое наклонение
    return prior


def _viterbi(scores: np.ndarray, change_penalty: float) -> list[int]:
    """Лучшая последовательность аккордов, а не лучший аккорд на каждом такте."""
    n_frames, n_states = scores.shape
    best = scores[0].copy()
    backtrack = np.zeros((n_frames, n_states), dtype=np.int16)
    for t in range(1, n_frames):
        stay = best                                   # остаться на том же аккорде
        switch = best.max() - change_penalty          # перейти на любой другой
        prev_best = int(np.argmax(best))
        choose_stay = stay > switch
        backtrack[t] = np.where(choose_stay, np.arange(n_states), prev_best)
        best = scores[t] + np.where(choose_stay, stay, switch)

    path = [int(np.argmax(best))]
    for t in range(n_frames - 1, 0, -1):
        path.append(int(backtrack[t][path[-1]]))
    return path[::-1]


def _drop_short(events: list[ChordEvent], min_duration: float) -> list[ChordEvent]:
    """Слишком короткие аккорды прилипают к соседям — так их слышит человек."""
    if min_duration <= 0 or len(events) < 2:
        return events
    out: list[ChordEvent] = []
    for ev in events:
        if ev.duration < min_duration and out:
            out[-1].end = ev.end
        else:
            out.append(ev)
    return _merge_repeats(out)


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
