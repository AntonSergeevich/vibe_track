"""Расшифровка вокала в текст и раскладка текста с аккордами.

Провайдеры (по убыванию качества):
  1. faster-whisper (если установлен) — с таймкодами по словам;
  2. openai-whisper;
  3. ручной текст от пользователя — выравнивается по секциям трека.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict

from .analysis import TrackAnalysis
from .audio_io import Audio, save
from .chords import ChordEvent, chord_at

logger = logging.getLogger(__name__)


@dataclass
class LyricWord:
    text: str
    start: float
    end: float


@dataclass
class LyricLine:
    text: str
    start: float
    end: float
    words: list[LyricWord] = field(default_factory=list)
    section: str = ""

    def to_dict(self) -> dict:
        return {"text": self.text, "start": round(self.start, 2), "end": round(self.end, 2),
                "section": self.section,
                "words": [asdict(w) for w in self.words]}


@dataclass
class Lyrics:
    lines: list[LyricLine] = field(default_factory=list)
    language: str = ""
    backend: str = "none"
    text: str = ""

    def to_dict(self) -> dict:
        return {"language": self.language, "backend": self.backend,
                "text": self.text, "lines": [l.to_dict() for l in self.lines]}


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        try:
            import whisper  # noqa: F401

            return True
        except Exception:
            return False


def transcribe(vocals: Audio, language: str | None = None,
               model_size: str | None = None) -> Lyrics:
    """Расшифровывает вокальную дорожку. Без Whisper вернёт пустой результат."""
    model_size = model_size or os.getenv("VIBETRACK_WHISPER_MODEL", "small")
    try:
        return _transcribe_faster_whisper(vocals, language, model_size)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover
        logger.warning("faster-whisper упал: %s", exc)
    try:
        return _transcribe_whisper(vocals, language, model_size)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover
        logger.warning("whisper упал: %s", exc)
    return Lyrics(backend="unavailable")


def _tmp_wav(vocals: Audio) -> str:
    import tempfile

    path = tempfile.mktemp(suffix=".wav")
    return save(path, vocals)


def _transcribe_faster_whisper(vocals: Audio, language, model_size) -> Lyrics:  # pragma: no cover
    from faster_whisper import WhisperModel

    path = _tmp_wav(vocals)
    try:
        model = WhisperModel(model_size, device=os.getenv("VIBETRACK_WHISPER_DEVICE", "cpu"),
                             compute_type=os.getenv("VIBETRACK_WHISPER_COMPUTE", "int8"))
        segments, info = model.transcribe(path, language=language, word_timestamps=True,
                                          vad_filter=True)
        lines = []
        for seg in segments:
            words = [LyricWord(w.word.strip(), float(w.start), float(w.end))
                     for w in (seg.words or [])]
            lines.append(LyricLine(seg.text.strip(), float(seg.start), float(seg.end), words))
        return Lyrics(lines=lines, language=info.language, backend="faster-whisper",
                      text="\n".join(l.text for l in lines))
    finally:
        if os.path.exists(path):
            os.remove(path)


def _transcribe_whisper(vocals: Audio, language, model_size) -> Lyrics:  # pragma: no cover
    import whisper

    path = _tmp_wav(vocals)
    try:
        model = whisper.load_model(model_size)
        result = model.transcribe(path, language=language)
        lines = [LyricLine(seg["text"].strip(), float(seg["start"]), float(seg["end"]))
                 for seg in result.get("segments", [])]
        return Lyrics(lines=lines, language=result.get("language", ""), backend="whisper",
                      text="\n".join(l.text for l in lines))
    finally:
        if os.path.exists(path):
            os.remove(path)


def lyrics_from_text(text: str, analysis: TrackAnalysis) -> Lyrics:
    """Пользовательский текст раскладывается по секциям и тактам трека."""
    raw_lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if not raw_lines:
        return Lyrics(backend="manual")
    sections = analysis.sections or []
    total = analysis.duration or (len(raw_lines) * 4.0)
    lines: list[LyricLine] = []
    if sections:
        singable = [s for s in sections if s.name in ("verse", "chorus", "bridge")] or sections
        per_section = max(1, len(raw_lines) // len(singable))
        idx = 0
        for s in singable:
            chunk = raw_lines[idx: idx + per_section]
            idx += per_section
            if not chunk:
                break
            step = s.duration / len(chunk)
            for i, line in enumerate(chunk):
                lines.append(LyricLine(line, s.start + i * step, s.start + (i + 1) * step,
                                       section=s.name))
        for i, line in enumerate(raw_lines[idx:]):  # хвост — в конец трека
            start = min(total - 1, total - (len(raw_lines) - idx - i) * 3.0)
            lines.append(LyricLine(line, max(0.0, start), max(0.0, start) + 3.0))
    else:
        step = total / len(raw_lines)
        lines = [LyricLine(l, i * step, (i + 1) * step) for i, l in enumerate(raw_lines)]
    return Lyrics(lines=lines, backend="manual", text="\n".join(raw_lines))


def lyric_sheet(lyrics: Lyrics, chords: list[ChordEvent], line_width: int = 74) -> str:
    """Текст с аккордами над словами — как в песеннике."""
    if not lyrics.lines:
        return ""
    out: list[str] = []
    current_section = None
    for line in lyrics.lines:
        if line.section and line.section != current_section:
            current_section = line.section
            out.append(f"[{current_section.upper()}]")
        chord_row = [" "] * max(len(line.text), line_width)
        placed: list[tuple[int, str]] = []
        if chords:
            if line.words:
                for w in line.words:
                    ch = chord_at(chords, w.start)
                    if ch:
                        col = line.text.find(w.text)
                        placed.append((max(col, 0), ch.name))
            else:
                span = max(line.end - line.start, 0.1)
                inside = [c for c in chords if line.start <= c.start < line.end]
                for c in inside:
                    col = int((c.start - line.start) / span * max(len(line.text) - 1, 1))
                    placed.append((max(col, 0), c.name))
                if not inside:
                    ch = chord_at(chords, line.start)
                    if ch:
                        placed.append((0, ch.name))
        last_end = -2
        for col, name in sorted(placed):
            col = max(col, last_end + 2)
            if col + len(name) >= len(chord_row):
                chord_row.extend([" "] * (col + len(name) - len(chord_row) + 1))
            chord_row[col: col + len(name)] = list(name)
            last_end = col + len(name)
        row = "".join(chord_row).rstrip()
        if row:
            out.append(row)
        out.append(line.text)
        out.append("")
    return "\n".join(out).strip()
