"""Секвенсор: из аккордов и сетки долей — в ноты ню-метал аранжировки.

Именно здесь рождаются рифы. Побочный, но важный эффект: мы точно знаем,
какие ноты сыграны, поэтому табулатура получается не «угаданной», а
ровно той, что звучит в миксе.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .arrangement import ArrangementSpec, BASS_TUNINGS, TUNINGS
from .chords import ChordEvent
from .analysis import TrackAnalysis

STEPS_PER_BAR = 16  # шестнадцатые


@dataclass
class Note:
    start: float
    dur: float
    midi: int
    vel: float = 0.9
    palm_mute: bool = False
    string: int | None = None     # 0 = самая низкая струна
    fret: int | None = None
    accent: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = round(self.start, 4)
        d["dur"] = round(self.dur, 4)
        return d


@dataclass
class DrumHit:
    start: float
    voice: str
    vel: float = 0.9

    def to_dict(self) -> dict:
        return {"start": round(self.start, 4), "voice": self.voice, "vel": round(self.vel, 3)}


@dataclass
class Part:
    instrument_id: str
    kind: str = "pitched"          # pitched | drums
    notes: list[Note] = field(default_factory=list)
    hits: list[DrumHit] = field(default_factory=list)
    tuning: str = ""

    def to_dict(self) -> dict:
        return {
            "instrument_id": self.instrument_id,
            "kind": self.kind,
            "tuning": self.tuning,
            "notes": [n.to_dict() for n in self.notes],
            "hits": [h.to_dict() for h in self.hits],
        }


@dataclass
class Bar:
    index: int
    start: float
    duration: float
    chord: ChordEvent | None
    section: str = "verse"

    @property
    def step(self) -> float:
        return self.duration / STEPS_PER_BAR


@dataclass
class Arrangement:
    spec: ArrangementSpec
    bars: list[Bar]
    parts: list[Part]
    duration: float

    def part(self, instrument_id: str) -> Part | None:
        for p in self.parts:
            if p.instrument_id == instrument_id:
                return p
        return None

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "bars": [{"index": b.index, "start": round(b.start, 3), "section": b.section,
                      "chord": b.chord.name if b.chord else None} for b in self.bars],
            "parts": [p.to_dict() for p in self.parts],
        }


# 16-шаговые ритмические сетки (1 = удар). Несколько вариантов на грув,
# чтобы такты не звучали как копипаста.
GROOVE_PATTERNS: dict[str, list[str]] = {
    "syncopated": [
        "1001001000100100",
        "1000100100101000",
        "1001000010010010",
    ],
    "bounce": [
        "1000100010001010",
        "1010001010001000",
        "1000101010001001",
    ],
    "driving": [
        "1010101010101010",
        "1011101010111010",
        "1010101011101010",
    ],
    "halftime": [
        "1000000010000010",
        "1000001000000100",
        "1000000010001000",
    ],
}

SNARE_STEPS = {"syncopated": (4, 12), "bounce": (4, 12), "driving": (4, 12), "halftime": (8,)}

PENTATONIC = {"minor": [0, 3, 5, 7, 10], "major": [0, 2, 4, 7, 9]}


def build_bars(analysis: TrackAnalysis, chords: list[ChordEvent], spec: ArrangementSpec) -> list[Bar]:
    """Такты строим по реальной сетке долей исходника — так аранжировка попадает в грув."""
    beats = analysis.beats
    bar_dur = 60.0 / max(spec.tempo, 1e-6) * 4
    starts: list[float] = []
    if len(beats) >= 8:
        downbeats = list(beats[::4])
        starts = [float(b) for b in downbeats if b + bar_dur * 0.5 <= analysis.duration]
    if len(starts) < 2:
        n = max(1, int(analysis.duration / bar_dur))
        starts = [i * bar_dur for i in range(n)]

    bars: list[Bar] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else min(start + bar_dur, analysis.duration)
        duration = max(end - start, 0.2)
        chord = _chord_for(chords, start, end)
        bars.append(Bar(index=i, start=float(start), duration=float(duration), chord=chord,
                        section=_section_for(analysis, start)))
    return bars


def _chord_for(chords: list[ChordEvent], start: float, end: float) -> ChordEvent | None:
    if not chords:
        return None
    best, best_overlap = None, 0.0
    for ch in chords:
        overlap = min(ch.end, end) - max(ch.start, start)
        if overlap > best_overlap:
            best, best_overlap = ch, overlap
    return best or chords[0]


def _section_for(analysis: TrackAnalysis, t: float) -> str:
    for s in analysis.sections:
        if s.start <= t < s.end:
            return s.name
    return "verse"


def _tuning_notes(name: str) -> list[int]:
    return TUNINGS.get(name) or BASS_TUNINGS.get(name) or TUNINGS["drop_c"]


def _is_drop(tuning: list[int]) -> bool:
    """В drop-строе две нижние струны — чистая квинта (power-chord одним пальцем)."""
    return len(tuning) >= 2 and (tuning[1] - tuning[0]) == 7


def power_chord_shape(root_pc: int, tuning_name: str, octave_shift: int = 0) -> list[tuple[int, int, int]]:
    """Возвращает [(струна, лад, midi)] для power-chord от заданного тона."""
    tuning = _tuning_notes(tuning_name)
    open_low = tuning[0]
    fret = (root_pc - open_low) % 12 + 12 * octave_shift
    root_midi = open_low + fret
    shape = [(0, fret, root_midi)]
    if _is_drop(tuning):
        shape.append((1, fret, tuning[1] + fret))
        if len(tuning) > 2:
            shape.append((2, fret, tuning[2] + fret))
    else:
        fifth_fret = fret + 2
        shape.append((1, fifth_fret, tuning[1] + fifth_fret))
        if len(tuning) > 2:
            shape.append((2, fifth_fret, tuning[2] + fifth_fret))
    return shape


def _pattern_for(spec: ArrangementSpec, bar: Bar, rng: np.random.Generator) -> str:
    variants = GROOVE_PATTERNS.get(spec.groove, GROOVE_PATTERNS["syncopated"])
    pattern = variants[int(rng.integers(0, len(variants)))]
    if bar.section == "chorus":
        pattern = "".join("1" if (i % 2 == 0 or c == "1") else "0" for i, c in enumerate(pattern))
    elif bar.section == "break":
        pattern = "".join(c if i % 4 == 0 else "0" for i, c in enumerate(pattern))
    if spec.density < 0.6:
        pattern = "".join(c if (i % 2 == 0) else "0" for i, c in enumerate(pattern))
    return pattern


def sequence(analysis: TrackAnalysis, chords: list[ChordEvent], spec: ArrangementSpec) -> Arrangement:
    bars = build_bars(analysis, chords, spec)
    rng = np.random.default_rng(spec.seed)
    parts: list[Part] = []

    enabled = set(spec.enabled_ids())
    guitar_patterns: dict[int, str] = {}

    for iid in ("guitar_rhythm", "guitar_rhythm_r"):
        if iid in enabled:
            item = spec.get(iid)
            tuning = item.tuning if item and item.tuning else spec.tuning
            # правый дубль играет тот же рифф, но со своим «человеческим» разбросом
            offset_rng = np.random.default_rng(spec.seed + (0 if iid == "guitar_rhythm" else 7))
            parts.append(_rhythm_guitar(bars, spec, tuning, offset_rng, guitar_patterns,
                                        iid, humanize=iid.endswith("_r")))

    if "bass" in enabled:
        item = spec.get("bass")
        parts.append(_bass(bars, spec, item.tuning if item else spec.bass_tuning, guitar_patterns))
    if "drums" in enabled:
        parts.append(_drums(bars, spec, guitar_patterns, rng))
    if "percussion" in enabled:
        parts.append(_percussion(bars, spec, rng))
    if "guitar_lead" in enabled:
        item = spec.get("guitar_lead")
        parts.append(_lead(bars, spec, item.tuning if item else spec.tuning, rng))
    if "synth_pad" in enabled:
        parts.append(_pad(bars, spec))
    if "synth_stab" in enabled:
        parts.append(_stabs(bars, spec))
    if "turntables" in enabled:
        parts.append(_turntables(bars, spec, rng))

    duration = bars[-1].start + bars[-1].duration if bars else analysis.duration
    return Arrangement(spec=spec, bars=bars, parts=parts, duration=float(duration))


def _rhythm_guitar(bars, spec, tuning, rng, pattern_cache, instrument_id, humanize=False) -> Part:
    part = Part(instrument_id=instrument_id, tuning=tuning)
    for bar in bars:
        pattern = pattern_cache.get(bar.index)
        if pattern is None:
            pattern = _pattern_for(spec, bar, rng)
            pattern_cache[bar.index] = pattern
        root_pc = bar.chord.root if bar.chord else 0
        shape = power_chord_shape(root_pc, tuning)
        step = bar.step
        open_chord = bar.section == "chorus"
        for i, ch in enumerate(pattern):
            if ch != "1":
                continue
            jitter = float(rng.normal(0, 0.004)) if humanize else 0.0
            start = bar.start + i * step + jitter
            sustained = open_chord and i % 4 == 0
            dur = step * (3.6 if sustained else 0.85)
            vel = 0.95 if i % 4 == 0 else 0.78
            for s, fret, midi in shape:
                part.notes.append(Note(start=start, dur=dur, midi=midi, vel=vel,
                                       palm_mute=not sustained, string=s, fret=fret,
                                       accent=i % 4 == 0))
    return part


def _bass(bars, spec, tuning, pattern_cache) -> Part:
    part = Part(instrument_id="bass", tuning=tuning)
    strings = _tuning_notes(tuning)
    for bar in bars:
        pattern = pattern_cache.get(bar.index) or GROOVE_PATTERNS[spec.groove][0]
        root_pc = bar.chord.root if bar.chord else 0
        fret = (root_pc - strings[0]) % 12
        midi = strings[0] + fret
        step = bar.step
        for i, ch in enumerate(pattern):
            if ch != "1":
                continue
            dur = step * 1.4
            vel = 0.95 if i % 4 == 0 else 0.8
            part.notes.append(Note(start=bar.start + i * step, dur=dur, midi=midi, vel=vel,
                                   string=0, fret=fret, accent=i % 4 == 0))
        # октавный подъём в конце такта — типовой ню-метал ход
        if bar.section == "chorus":
            part.notes.append(Note(start=bar.start + 14 * step, dur=step * 1.6, midi=midi + 12,
                                   vel=0.85, string=min(2, len(strings) - 1), fret=fret))
    return part


def _drums(bars, spec, pattern_cache, rng) -> Part:
    part = Part(instrument_id="drums", kind="drums")
    snare_steps = SNARE_STEPS.get(spec.groove, (4, 12))
    prev_section = None
    for bar in bars:
        pattern = pattern_cache.get(bar.index) or GROOVE_PATTERNS[spec.groove][0]
        step = bar.step
        # бочка — под гитарные чаги
        for i, ch in enumerate(pattern):
            if ch == "1":
                part.hits.append(DrumHit(bar.start + i * step, "kick", 0.95 if i % 4 == 0 else 0.8))
        for s in snare_steps:
            part.hits.append(DrumHit(bar.start + s * step, "snare", 0.95))
        hat_step = 2 if spec.aggression < 0.8 else 1
        for i in range(0, STEPS_PER_BAR, hat_step):
            if bar.section == "break" and i % 4:
                continue
            part.hits.append(DrumHit(bar.start + i * step, "hat", 0.45 if i % 2 else 0.6))
        if bar.section != prev_section:
            part.hits.append(DrumHit(bar.start, "crash", 0.9))
            prev_section = bar.section
        # заполнение перед сменой секции
        if bar.index + 1 < len(bars) and bars[bar.index + 1].section != bar.section:
            for j, s in enumerate((12, 13, 14, 15)):
                voice = "tom_hi" if j < 2 else "tom_lo"
                part.hits.append(DrumHit(bar.start + s * step, voice, 0.85))
    return part


def _percussion(bars, spec, rng) -> Part:
    part = Part(instrument_id="percussion", kind="drums")
    for bar in bars:
        step = bar.step
        for i in (2, 6, 7, 10, 14):
            if rng.random() < 0.75:
                part.hits.append(DrumHit(bar.start + i * step, "bongo", 0.5 + 0.3 * rng.random()))
    return part


def _lead(bars, spec, tuning, rng) -> Part:
    part = Part(instrument_id="guitar_lead", tuning=tuning)
    scale = PENTATONIC.get(spec.mode, PENTATONIC["minor"])
    strings = _tuning_notes(tuning)
    for bar in bars:
        if bar.section not in ("chorus", "solo", "break"):
            continue
        root_pc = bar.chord.root if bar.chord else 0
        base = strings[0] + 24 + ((root_pc - strings[0]) % 12)
        step = bar.step
        i = 0
        while i < STEPS_PER_BAR:
            length = int(rng.choice([2, 2, 4]))
            degree = int(rng.choice(scale))
            octave = 12 if rng.random() < 0.25 else 0
            midi = base + degree + octave
            part.notes.append(Note(start=bar.start + i * step, dur=step * length * 0.9,
                                   midi=midi, vel=0.7, string=None, fret=None))
            i += length
    return part


def _pad(bars, spec) -> Part:
    part = Part(instrument_id="synth_pad")
    for bar in bars:
        if not bar.chord:
            continue
        for midi in bar.chord.pitches(octave=4)[:3]:
            part.notes.append(Note(start=bar.start, dur=bar.duration * 1.05, midi=midi, vel=0.45))
    return part


def _stabs(bars, spec) -> Part:
    part = Part(instrument_id="synth_stab")
    for bar in bars:
        if bar.section != "chorus" or not bar.chord:
            continue
        step = bar.step
        for i in (0, 8):
            part.notes.append(Note(start=bar.start + i * step, dur=step * 2,
                                   midi=bar.chord.root + 48, vel=0.6))
    return part


def _turntables(bars, spec, rng) -> Part:
    part = Part(instrument_id="turntables", kind="drums")
    prev_section = None
    for bar in bars:
        if bar.section != prev_section:
            part.hits.append(DrumHit(bar.start, "scratch", 0.8))
            prev_section = bar.section
        elif bar.index % 4 == 3 and rng.random() < 0.6:
            part.hits.append(DrumHit(bar.start + bar.step * 12, "scratch", 0.6))
    return part
