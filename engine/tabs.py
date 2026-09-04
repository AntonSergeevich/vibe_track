"""Табулатура и аккордовые схемы.

Табы строятся из тех же нот, что и звучат в миксе, поэтому совпадают
с результатом дословно. Отдельно рисуются аппликатуры аккордов.
"""
from __future__ import annotations

from dataclasses import dataclass

from .analysis import PITCH_NAMES
from .arrangement import BASS_TUNINGS, TUNINGS
from .chords import ChordEvent
from .sequencer import Arrangement, Note, Part, STEPS_PER_BAR

TUNING_LABELS = {
    "standard_e": "E A D G B E",
    "drop_d": "D A D G B E",
    "drop_c#": "C# G# C# F# A# D#",
    "drop_c": "C G C F A D",
    "drop_b": "B F# B E G# C#",
    "drop_a_7": "A E A D G B E",
    "standard_b_7": "B E A D G B E",
    "drop_g_7": "G D G C F A D",
    "drop_e_8": "E B E A D G B E",
    "bass_standard": "E A D G",
    "bass_drop_d": "D A D G",
    "bass_drop_c": "C G C F",
    "bass_5": "B E A D G",
    "bass_5_drop_a": "A E A D G",
}


def tuning_notes(name: str) -> list[int]:
    return TUNINGS.get(name) or BASS_TUNINGS.get(name) or TUNINGS["drop_c"]


def string_labels(name: str) -> list[str]:
    label = TUNING_LABELS.get(name)
    if label:
        return label.split()[::-1]  # сверху вниз: высокая струна первой
    return [PITCH_NAMES[n % 12] for n in tuning_notes(name)][::-1]


@dataclass
class TabBar:
    index: int
    section: str
    chord: str
    lines: list[str]


def _fret_for(note: Note, tuning: list[int]) -> tuple[int, int]:
    """Струна и лад: если секвенсор их не проставил, подбираем ближайшие."""
    if note.string is not None and note.fret is not None:
        return note.string, note.fret
    best = (len(tuning) - 1, 0, 999)
    for s in range(len(tuning) - 1, -1, -1):
        fret = note.midi - tuning[s]
        if 0 <= fret <= 22 and fret < best[2]:
            best = (s, fret, fret)
    return best[0], best[1]


def render_tab(arr: Arrangement, instrument_id: str = "guitar_rhythm",
               bars_per_line: int = 2, max_bars: int | None = 32) -> str:
    """ASCII-таб для указанной партии."""
    part: Part | None = arr.part(instrument_id)
    if part is None or not part.notes:
        return ""
    tuning_name = part.tuning or arr.spec.tuning
    tuning = tuning_notes(tuning_name)
    labels = string_labels(tuning_name)
    n_strings = len(tuning)

    bars = arr.bars[:max_bars] if max_bars else arr.bars
    out_lines: list[str] = [
        f"# {instrument_id} — строй {TUNING_LABELS.get(tuning_name, tuning_name)} "
        f"({tuning_name}), {arr.spec.tempo:.0f} BPM",
        "",
    ]

    for chunk_start in range(0, len(bars), bars_per_line):
        chunk = bars[chunk_start: chunk_start + bars_per_line]
        rows = [f"{labels[i]:>2}|" for i in range(n_strings)]
        header, pm_row = "  |", "PM|"
        for bar in chunk:
            grid = [["-" for _ in range(STEPS_PER_BAR)] for _ in range(n_strings)]
            muted = [False] * STEPS_PER_BAR
            step = bar.duration / STEPS_PER_BAR
            for note in part.notes:
                if not (bar.start - 1e-3 <= note.start < bar.start + bar.duration - 1e-3):
                    continue
                idx = max(0, min(STEPS_PER_BAR - 1,
                                 int(round((note.start - bar.start) / max(step, 1e-6)))))
                s_idx, fret = _fret_for(note, tuning)
                grid[n_strings - 1 - s_idx][idx] = str(fret)   # низкая струна снизу
                muted[idx] = muted[idx] or note.palm_mute
            for r in range(n_strings):
                rows[r] += "".join(c if len(c) == 2 else f"{c}-" for c in grid[r]) + "|"
            chord_name = bar.chord.name if bar.chord else "-"
            header += f"{chord_name:<{STEPS_PER_BAR * 2}}|"
            pm_row += "".join("*-" if m else "--" for m in muted) + "|"
        out_lines.append(header)
        out_lines.extend(rows)
        out_lines.append(pm_row + "   (* = palm mute)")
        out_lines.append("")
    return "\n".join(out_lines)


def chord_diagram(chord: ChordEvent, tuning_name: str = "drop_c") -> str:
    """Аппликатура power-chord'а в виде маленькой схемы."""
    from .sequencer import power_chord_shape

    tuning = tuning_notes(tuning_name)
    shape = power_chord_shape(chord.root, tuning_name)
    frets = {s: f for s, f, _ in shape}
    labels = string_labels(tuning_name)
    lines = [f"{chord.name} ({TUNING_LABELS.get(tuning_name, tuning_name)})"]
    for row, label in enumerate(labels):
        s = len(tuning) - 1 - row
        mark = f"{frets[s]}" if s in frets else "x"
        lines.append(f"{label:>2} |{mark}")
    return "\n".join(lines)


def chord_chart(chords: list[ChordEvent], bars_per_line: int = 4) -> str:
    """Аккордовая схема по тактам."""
    if not chords:
        return ""
    cells = [c.name for c in chords]
    lines = []
    for i in range(0, len(cells), bars_per_line):
        row = cells[i: i + bars_per_line]
        lines.append(" | ".join(f"{c:<7}" for c in row))
    return "\n".join(lines)


def tab_for_all(arr: Arrangement) -> dict[str, str]:
    out = {}
    for part in arr.parts:
        if part.kind != "pitched" or not part.notes:
            continue
        tab = render_tab(arr, part.instrument_id)
        if tab:
            out[part.instrument_id] = tab
    return out
