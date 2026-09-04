"""Рендер партий секвенсора в аудио-дорожки."""
from __future__ import annotations

import numpy as np

from . import SR
from .arrangement import ArrangementSpec
from .audio_io import Audio
from .dsp import (compressor, delay_fx, highpass, lowpass, normalize_active_rms, pan,
                  reverb, stereo_width)
from . import instruments as ins
from .sequencer import Arrangement, Part

# Единый номинальный уровень всех дорожек: дальше балансом рулит микшер,
# а не случайная плотность нот в партии.
NOMINAL_RMS_DB = -18.0

# Как звучит каждый инструмент в готовом миксе
PART_FX = {
    "guitar_rhythm": {"pan": -0.4, "reverb": 0.06},
    "guitar_rhythm_r": {"pan": 0.4, "reverb": 0.06},
    "guitar_lead": {"pan": 0.12, "reverb": 0.16, "delay": 0.18},
    "bass": {"pan": 0.0, "reverb": 0.0},
    "drums": {"pan": 0.0, "reverb": 0.12},
    "percussion": {"pan": 0.45, "reverb": 0.14},
    "turntables": {"pan": -0.55, "reverb": 0.1},
    "synth_pad": {"pan": 0.0, "reverb": 0.3},
    "synth_stab": {"pan": 0.3, "reverb": 0.12},
}


def render_arrangement(arr: Arrangement, sr: int = SR) -> dict[str, Audio]:
    """Возвращает {instrument_id: Audio} — по дорожке на инструмент."""
    total = int((arr.duration + 3.0) * sr)
    stems: dict[str, Audio] = {}
    for part in arr.parts:
        mono = _render_part(part, arr.spec, total, sr)
        if mono is None:
            continue
        placed = _place(part.instrument_id, mono, sr)
        stems[part.instrument_id] = Audio(normalize_active_rms(placed, NOMINAL_RMS_DB), sr)
    return stems


def _render_part(part: Part, spec: ArrangementSpec, total: int, sr: int) -> np.ndarray | None:
    if part.kind == "drums":
        if part.instrument_id == "turntables":
            return _render_scratches(part, total, sr)
        return _render_drums(part, total, sr)
    renderer = {
        "guitar_rhythm": _render_guitar,
        "guitar_rhythm_r": _render_guitar,
        "guitar_lead": _render_lead,
        "bass": _render_bass,
        "synth_pad": _render_pad,
        "synth_stab": _render_stabs,
    }.get(part.instrument_id)
    return renderer(part, spec, total, sr) if renderer else None


def _add(buf: np.ndarray, sig: np.ndarray, start_sample: int) -> None:
    if start_sample >= len(buf) or len(sig) == 0:
        return
    end = min(start_sample + len(sig), len(buf))
    buf[start_sample:end] += sig[: end - start_sample]


def _render_guitar(part: Part, spec: ArrangementSpec, total: int, sr: int) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    for i, note in enumerate(part.notes):
        sig = ins.pluck(ins.midi_to_hz(note.midi), note.dur, sr, velocity=note.vel,
                        palm_mute=note.palm_mute, seed=i,
                        bright=0.8 + 0.4 * spec.aggression)
        _add(buf, sig, int(note.start * sr))
    tone = ins.GuitarTone(
        drive=8.0 + 12.0 * spec.aggression,
        cab_high=4600 + 1200 * spec.aggression,
        scoop_db=-2.0 - 3.0 * spec.aggression,
        presence_db=3.0 + 3.0 * spec.aggression,
    )
    return ins.amp(buf, tone, sr)


def _render_lead(part: Part, spec: ArrangementSpec, total: int, sr: int) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    for i, note in enumerate(part.notes):
        sig = ins.pluck(ins.midi_to_hz(note.midi), note.dur, sr, velocity=note.vel,
                        palm_mute=False, seed=1000 + i)
        _add(buf, sig, int(note.start * sr))
    tone = ins.GuitarTone(drive=6.0 + 8.0 * spec.aggression, cab_high=6000, scoop_db=0.0,
                          presence_db=5.0)
    return ins.amp(buf, tone, sr)


def _render_bass(part: Part, spec: ArrangementSpec, total: int, sr: int) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    for i, note in enumerate(part.notes):
        sig = ins.bass_note(ins.midi_to_hz(note.midi), note.dur, sr, velocity=note.vel,
                            drive=2.0 + 3.0 * spec.aggression, seed=i)
        _add(buf, sig, int(note.start * sr))
    buf = compressor(buf, threshold_db=-22, ratio=4.5, attack_ms=8, release_ms=90, sr=sr)
    return lowpass(highpass(buf, 35, sr), 6000, sr)


def _render_drums(part: Part, total: int, sr: int) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    cache: dict[tuple[str, int], np.ndarray] = {}
    for hit in part.hits:
        key = (hit.voice, int(hit.vel * 20))
        sig = cache.get(key)
        if sig is None:
            voice = ins.DRUM_VOICES.get(hit.voice)
            if voice is None:
                continue
            sig = voice(sr, hit.vel)
            cache[key] = sig
        _add(buf, sig, int(hit.start * sr))
    return compressor(buf, threshold_db=-16, ratio=4.0, attack_ms=5, release_ms=110, sr=sr)


def _render_scratches(part: Part, total: int, sr: int) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    for i, hit in enumerate(part.hits):
        _add(buf, ins.scratch(sr, hit.vel, seed=i), int(hit.start * sr))
    return buf


def _render_pad(part: Part, spec: ArrangementSpec, total: int, sr: int) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    for i, note in enumerate(part.notes):
        _add(buf, ins.pad_note(ins.midi_to_hz(note.midi), note.dur, sr, note.vel, seed=i),
             int(note.start * sr))
    return lowpass(buf, 4000, sr)


def _render_stabs(part: Part, spec: ArrangementSpec, total: int, sr: int) -> np.ndarray:
    buf = np.zeros(total, dtype=np.float32)
    for i, note in enumerate(part.notes):
        _add(buf, ins.synth_stab(ins.midi_to_hz(note.midi), note.dur, sr, note.vel, seed=i),
             int(note.start * sr))
    return buf


def _place(instrument_id: str, mono: np.ndarray, sr: int) -> np.ndarray:
    cfg = PART_FX.get(instrument_id, {})
    stereo = pan(mono, cfg.get("pan", 0.0))
    if instrument_id in ("guitar_rhythm", "guitar_rhythm_r"):
        stereo = stereo_width(stereo, 1.15)
    if cfg.get("delay"):
        stereo = delay_fx(stereo, sr, time_s=0.3, feedback=0.3, mix_amt=cfg["delay"])
    if cfg.get("reverb"):
        stereo = reverb(stereo, sr, decay_s=1.4, mix_amt=cfg["reverb"])
    return stereo
