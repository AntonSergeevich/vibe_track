"""Аранжировочная спецификация: из текстового описания — в структуру.

Пользователь пишет «сделай ню-метал в духе Korn, семиструнка в Drop A,
пятиструнный бас, скретчи, женский вокал в припеве» — здесь это
превращается в ArrangementSpec, который дальше рендерит секвенсор.

Парсер правил детерминированный (и потому тестируемый). Опционально
поверх него можно подключить LLM (см. `refine_with_llm`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from .analysis import TrackAnalysis

# --------------------------------------------------------------- строй
TUNINGS: dict[str, list[int]] = {
    "standard_e": [40, 45, 50, 55, 59, 64],
    "drop_d": [38, 45, 50, 55, 59, 64],
    "drop_c#": [37, 44, 49, 54, 58, 63],
    "drop_c": [36, 43, 48, 53, 57, 62],
    "drop_b": [35, 42, 47, 52, 56, 61],
    "drop_a_7": [33, 40, 45, 50, 55, 59, 64],
    "standard_b_7": [35, 40, 45, 50, 55, 59, 64],
    "drop_g_7": [31, 38, 43, 48, 53, 57, 62],
    "drop_e_8": [28, 35, 40, 45, 50, 55, 59, 64],
}
BASS_TUNINGS: dict[str, list[int]] = {
    "bass_standard": [28, 33, 38, 43],
    "bass_drop_d": [26, 33, 38, 43],
    "bass_drop_c": [24, 31, 36, 41],
    "bass_5": [23, 28, 33, 38, 43],
    "bass_5_drop_a": [21, 28, 33, 38, 43],
}

TUNING_ALIASES = {
    "drop a": "drop_a_7", "дроп а": "drop_a_7", "drop-a": "drop_a_7",
    "drop b": "drop_b", "дроп б": "drop_b", "дроп в": "drop_b",
    "drop c#": "drop_c#", "drop c": "drop_c", "дроп с": "drop_c", "дроп ц": "drop_c",
    "drop d": "drop_d", "дроп д": "drop_d",
    "drop g": "drop_g_7", "drop e": "drop_e_8",
    "standard": "standard_e", "стандарт": "standard_e",
}

# ------------------------------------------------------- каталог инструментов
INSTRUMENT_CATALOG: dict[str, dict] = {
    "guitar_rhythm": {"label": "Ритм-гитара", "role": "rhythm", "default": True,
                      "tuning": "drop_c", "gain_db": -3.0, "pan": -0.35},
    "guitar_rhythm_r": {"label": "Ритм-гитара (дубль)", "role": "rhythm", "default": True,
                        "tuning": "drop_c", "gain_db": -3.0, "pan": 0.35},
    "guitar_lead": {"label": "Соло-гитара", "role": "lead", "default": False,
                    "tuning": "drop_c", "gain_db": -7.0, "pan": 0.15},
    "bass": {"label": "Бас-гитара", "role": "low", "default": True,
             "tuning": "bass_5", "gain_db": -4.0, "pan": 0.0},
    "drums": {"label": "Барабаны", "role": "percussion", "default": True,
              "tuning": "", "gain_db": -2.0, "pan": 0.0},
    "percussion": {"label": "Перкуссия (бонго/конги)", "role": "percussion", "default": False,
                   "tuning": "", "gain_db": -12.0, "pan": 0.4},
    "turntables": {"label": "Тёрнтейблы / скретчи", "role": "fx", "default": False,
                   "tuning": "", "gain_db": -13.0, "pan": -0.5},
    "synth_pad": {"label": "Синт-пэд / атмосфера", "role": "atmo", "default": False,
                  "tuning": "", "gain_db": -16.0, "pan": 0.0},
    "synth_stab": {"label": "Синт-стэбы", "role": "atmo", "default": False,
                   "tuning": "", "gain_db": -14.0, "pan": 0.25},
}

# Ключевые слова -> инструменты (RU/EN)
INSTRUMENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "guitar_lead": ("соло", "lead", "лид", "solo", "мелодия гитары", "гитарное соло"),
    "percussion": ("бонго", "конги", "перкусс", "percussion", "bongo", "conga", "джембе"),
    "turntables": ("скретч", "scratch", "тёрнтейбл", "тернтейбл", "turntable", "dj", "диджей", "винил"),
    "synth_pad": ("пэд", "pad", "атмосфер", "эмбиент", "ambient", "подклад", "синт", "synth", "клавиш"),
    "synth_stab": ("стэб", "stab", "индастриал", "industrial", "электроник"),
    "bass": ("бас", "bass", "слэп", "slap"),
    "drums": ("барабан", "drums", "ударн", "бочка", "kick"),
    "guitar_rhythm": ("гитар", "guitar", "риф", "riff", "чаг", "chug"),
}

GROOVE_KEYWORDS = {
    "syncopated": ("синкоп", "korn", "корн", "рвано", "syncopat", "качающ"),
    "halftime": ("half-time", "халф", "медленн", "тяжёл", "тяжел", "sludge", "слоу"),
    "bounce": ("баунс", "bounce", "качёв", "качев", "limp", "лимп", "прыг"),
    "driving": ("драйв", "быстр", "агресс", "linkin", "линкин", "энергич"),
}

VOCAL_STYLE_KEYWORDS = {
    "rap": ("рэп", "читка", "rap", "речитатив"),
    "scream": ("скрим", "scream", "гроул", "growl", "крик", "ор"),
    "clean": ("чистый", "clean", "мелодич", "напев"),
    "whisper": ("шёпот", "шепот", "whisper"),
}

# Жанры. Честно говоря, это варианты одного движка: меняются грув, строй,
# набор инструментов, плотность и характер вокала. Разные жанры звучат
# по-разному, но это не разные продакшены — общий синтез остаётся общим.
GENRES: dict[str, dict] = {
    "nu_metal": {
        "label": "Ню-метал", "hint": "Korn, Limp Bizkit — синкопы, дроп-строй, читка",
        "groove": "syncopated", "tuning": "drop_c", "bass_tuning": "bass_5",
        "aggression": 0.8, "density": 0.75, "instruments": ["turntables", "percussion"],
        "male_style": "rap", "female_style": "clean",
    },
    "metalcore": {
        "label": "Металкор", "hint": "плотные чаги, скрим в куплете, чистый припев",
        "groove": "driving", "tuning": "drop_b", "bass_tuning": "bass_5",
        "aggression": 0.95, "density": 0.9, "instruments": ["guitar_lead"],
        "male_style": "scream", "female_style": "clean",
    },
    "alt_rock": {
        "label": "Альт-рок", "hint": "Linkin Park — мелодика, пэды, умеренный вес",
        "groove": "driving", "tuning": "drop_d", "bass_tuning": "bass_standard",
        "aggression": 0.55, "density": 0.6, "instruments": ["synth_pad", "guitar_lead"],
        "male_style": "clean", "female_style": "clean",
    },
    "grunge": {
        "label": "Гранж", "hint": "Nirvana, Soundgarden — грязный тон, свободная подача",
        "groove": "halftime", "tuning": "drop_d", "bass_tuning": "bass_standard",
        "aggression": 0.6, "density": 0.5, "instruments": [],
        "male_style": "clean", "female_style": "clean",
    },
    "punk": {
        "label": "Панк", "hint": "быстро, прямо, без лишнего",
        "groove": "driving", "tuning": "standard_e", "bass_tuning": "bass_standard",
        "aggression": 0.7, "density": 0.95, "instruments": [],
        "male_style": "clean", "female_style": "clean",
    },
    "industrial": {
        "label": "Индастриал", "hint": "Rammstein, NIN — машинный грув, синт-стэбы",
        "groove": "driving", "tuning": "drop_c", "bass_tuning": "bass_5",
        "aggression": 0.85, "density": 0.8,
        "instruments": ["synth_stab", "synth_pad"],
        "male_style": "scream", "female_style": "clean",
    },
    "trap_metal": {
        "label": "Трэп-метал", "hint": "рваный бит, читка, редкие тяжёлые гитары",
        "groove": "halftime", "tuning": "drop_a_7", "bass_tuning": "bass_5_drop_a",
        "aggression": 0.7, "density": 0.45, "instruments": ["synth_stab", "turntables"],
        "male_style": "rap", "female_style": "rap",
    },
    "hard_rock": {
        "label": "Хард-рок", "hint": "классический риффовый рок без дроп-строя",
        "groove": "bounce", "tuning": "standard_e", "bass_tuning": "bass_standard",
        "aggression": 0.5, "density": 0.6, "instruments": ["guitar_lead"],
        "male_style": "clean", "female_style": "clean",
    },
}

GENRE_KEYWORDS = {
    "nu_metal": ("ню-метал", "ню метал", "nu-metal", "nu metal", "нюметал"),
    "metalcore": ("металкор", "metalcore", "дэткор", "deathcore"),
    "alt_rock": ("альт-рок", "альтернатив", "alt-rock", "alternative"),
    "grunge": ("гранж", "grunge"),
    "punk": ("панк", "punk"),
    "industrial": ("индастриал", "industrial", "раммштайн", "rammstein"),
    "trap_metal": ("трэп", "трап", "trap"),
    "hard_rock": ("хард-рок", "хард рок", "hard rock", "классический рок"),
}


REFERENCE_ARTISTS = {
    "korn": {"groove": "syncopated", "tuning": "drop_a_7", "instruments": ["percussion"], "aggression": 0.8},
    "корн": {"groove": "syncopated", "tuning": "drop_a_7", "instruments": ["percussion"], "aggression": 0.8},
    "slipknot": {"groove": "driving", "tuning": "drop_b", "instruments": ["turntables", "percussion"], "aggression": 0.95},
    "слипкнот": {"groove": "driving", "tuning": "drop_b", "instruments": ["turntables", "percussion"], "aggression": 0.95},
    "limp bizkit": {"groove": "bounce", "tuning": "drop_c", "instruments": ["turntables"], "aggression": 0.7},
    "linkin park": {"groove": "driving", "tuning": "drop_d", "instruments": ["synth_pad", "turntables"], "aggression": 0.6},
    "линкин": {"groove": "driving", "tuning": "drop_d", "instruments": ["synth_pad", "turntables"], "aggression": 0.6},
    "deftones": {"groove": "halftime", "tuning": "drop_c#", "instruments": ["synth_pad"], "aggression": 0.55},
    "дефтонс": {"groove": "halftime", "tuning": "drop_c#", "instruments": ["synth_pad"], "aggression": 0.55},
    "papa roach": {"groove": "bounce", "tuning": "drop_d", "instruments": [], "aggression": 0.7},
    "system of a down": {"groove": "driving", "tuning": "drop_c", "instruments": [], "aggression": 0.85},
}


@dataclass
class InstrumentSpec:
    id: str
    label: str = ""
    role: str = "rhythm"
    enabled: bool = True
    tuning: str = ""
    gain_db: float = -6.0
    pan: float = 0.0
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VocalSpec:
    """Кто и как поёт. Роли раскладываются по секциям."""
    male: bool = True
    female: bool = False
    male_style: str = "rap"          # rap | scream | clean | whisper
    female_style: str = "clean"
    male_sections: list[str] = field(default_factory=lambda: ["verse", "break"])
    female_sections: list[str] = field(default_factory=lambda: ["chorus"])
    harmony: bool = False        # бэк-вокал только по явной просьбе
    doubling: bool = True
    autotune: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArrangementSpec:
    title: str = "Nu-metal version"
    tempo: float = 100.0
    key: str = "A"
    mode: str = "minor"
    tuning: str = "drop_c"
    bass_tuning: str = "bass_5"
    genre: str = "nu_metal"
    groove: str = "syncopated"
    aggression: float = 0.75          # 0..1 — гейн, плотность, скорость
    density: float = 0.7              # плотность нот в риффе
    swing: float = 0.0
    transpose: int = 0                # сдвиг тональности в полутонах, -12..+12
    tempo_scale: float = 1.0          # растяжение исходного темпа
    seed: int = 1337
    instruments: list[InstrumentSpec] = field(default_factory=list)
    vocals: VocalSpec = field(default_factory=VocalSpec)
    references: list[str] = field(default_factory=list)
    prompt: str = ""
    notes: list[str] = field(default_factory=list)

    def enabled_ids(self) -> list[str]:
        return [i.id for i in self.instruments if i.enabled]

    def get(self, instrument_id: str) -> InstrumentSpec | None:
        for i in self.instruments:
            if i.id == instrument_id:
                return i
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["instruments"] = [i.to_dict() for i in self.instruments]
        d["vocals"] = self.vocals.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ArrangementSpec":
        data = dict(data or {})
        instruments = [InstrumentSpec(**i) for i in data.pop("instruments", [])]
        vocals = VocalSpec(**data.pop("vocals", {})) if data.get("vocals") is not None else VocalSpec()
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        spec = cls(**clean)
        if instruments:
            spec.instruments = instruments
        spec.vocals = vocals
        return spec


def default_instruments(tuning: str, bass_tuning: str, extra: list[str] | None = None) -> list[InstrumentSpec]:
    extra = extra or []
    out = []
    for iid, meta in INSTRUMENT_CATALOG.items():
        enabled = meta["default"] or iid in extra
        t = meta["tuning"]
        if t.startswith("bass"):
            t = bass_tuning
        elif t:
            t = tuning
        out.append(InstrumentSpec(id=iid, label=meta["label"], role=meta["role"], enabled=enabled,
                                  tuning=t, gain_db=meta["gain_db"], pan=meta["pan"]))
    return out


def apply_genre(spec: ArrangementSpec, genre: str) -> ArrangementSpec:
    """Раскладывает пресет жанра в спецификацию."""
    preset = GENRES.get(genre)
    if not preset:
        return spec
    spec.genre = genre
    spec.groove = preset["groove"]
    spec.tuning = preset["tuning"]
    spec.bass_tuning = preset["bass_tuning"]
    spec.aggression = preset["aggression"]
    spec.density = preset["density"]
    spec.instruments = default_instruments(spec.tuning, spec.bass_tuning,
                                           preset["instruments"])
    spec.vocals.male_style = preset["male_style"]
    spec.vocals.female_style = preset["female_style"]
    return spec


def parse_prompt(prompt: str, analysis: TrackAnalysis | None = None,
                 overrides: dict | None = None) -> ArrangementSpec:
    """Главный вход: текст описания -> ArrangementSpec."""
    text = (prompt or "").lower()
    notes: list[str] = []

    spec = ArrangementSpec(prompt=prompt or "")
    if analysis is not None:
        spec.tempo = float(analysis.tempo)
        spec.key, spec.mode = analysis.key, analysis.mode

    # 0. Жанр задаёт основу, всё остальное её уточняет
    for genre, keys in GENRE_KEYWORDS.items():
        if any(k in text for k in keys):
            spec = apply_genre(spec, genre)
            notes.append(f"Жанр: {GENRES[genre]['label']}.")
            break

    # 1. Референсы задают базовый характер
    extra_instruments: list[str] = []
    for artist, cfg in REFERENCE_ARTISTS.items():
        if artist in text:
            spec.references.append(artist)
            spec.groove = cfg["groove"]
            spec.tuning = cfg["tuning"]
            spec.aggression = cfg["aggression"]
            extra_instruments.extend(cfg["instruments"])
            notes.append(f"Референс «{artist}»: грув {cfg['groove']}, строй {cfg['tuning']}.")

    # 2. Явно указанный строй важнее референса
    for alias, tuning in TUNING_ALIASES.items():
        if alias in text:
            spec.tuning = tuning
            notes.append(f"Строй из описания: {tuning}.")
            break
    if re.search(r"(7|семи)[- ]?струн", text):
        if not spec.tuning.endswith("_7"):
            spec.tuning = "drop_a_7"
        notes.append("Семиструнная гитара.")
    if re.search(r"(8|восьми)[- ]?струн", text):
        spec.tuning = "drop_e_8"
        notes.append("Восьмиструнная гитара.")
    if re.search(r"(5|пяти)[- ]?струн", text) and ("бас" in text or "bass" in text):
        spec.bass_tuning = "bass_5"
        notes.append("Пятиструнный бас.")

    # 3. Грув и агрессия
    for groove, keys in GROOVE_KEYWORDS.items():
        if any(k in text for k in keys):
            spec.groove = groove
            break
    if any(k in text for k in ("очень тяжёл", "очень тяжел", "максимально агрессив", "brutal", "жёстк", "жестк")):
        spec.aggression = min(1.0, spec.aggression + 0.2)
    if any(k in text for k in ("мягч", "спокойн", "меланхол", "атмосферн")):
        spec.aggression = max(0.25, spec.aggression - 0.25)

    # 3.5 Тональность
    semitones = re.search(r"(?:на\s*)?(\d+)\s*полутон", text)
    tones = re.search(r"(?:на\s*)?(\d+)\s*тон[аоуы]?\b", text)
    shift = 0
    if semitones:
        shift = int(semitones.group(1))
    elif tones:
        shift = int(tones.group(1)) * 2
    elif re.search(r"на\s*пол\s?тона", text):
        shift = 1
    elif re.search(r"на\s*тон\b", text):
        shift = 2
    if shift:
        direction = -1 if ("ниже" in text or "низк" in text) else 1
        spec.transpose = max(-12, min(12, shift * direction))
        notes.append(f"Тональность сдвинута на {spec.transpose:+d} полутона.")

    # 4. Темп
    bpm = re.search(r"(\d{2,3})\s*(?:bpm|бпм|уд/?мин)", text)
    if bpm:
        target = float(bpm.group(1))
        if spec.tempo > 0:
            spec.tempo_scale = target / spec.tempo
        spec.tempo = target
        notes.append(f"Темп задан вручную: {target:.0f} BPM.")
    elif "быстрее" in text:
        spec.tempo_scale, spec.tempo = 1.12, spec.tempo * 1.12
    elif "медленнее" in text:
        spec.tempo_scale, spec.tempo = 0.9, spec.tempo * 0.9

    # 5. Инструменты
    for iid, keys in INSTRUMENT_KEYWORDS.items():
        if any(k in text for k in keys):
            extra_instruments.append(iid)
    if "без гитар" in text:
        extra_instruments = [i for i in extra_instruments if not i.startswith("guitar")]
    spec.instruments = default_instruments(spec.tuning, spec.bass_tuning, extra_instruments)
    for iid in ("guitar_rhythm", "guitar_rhythm_r", "guitar_lead"):
        item = spec.get(iid)
        if item:
            item.tuning = spec.tuning
    if "без гитар" in text:
        for iid in ("guitar_rhythm", "guitar_rhythm_r", "guitar_lead"):
            item = spec.get(iid)
            if item:
                item.enabled = False
    if "без барабан" in text or "без ударн" in text:
        drums = spec.get("drums")
        if drums:
            drums.enabled = False

    # 6. Вокал
    spec.vocals = _parse_vocals(text)

    # 7. Ручные переопределения из формы (важнее текста)
    if overrides:
        spec = apply_overrides(spec, overrides)

    spec.density = min(1.0, 0.45 + spec.aggression * 0.5)
    spec.notes = notes
    return spec


def _parse_vocals(text: str) -> VocalSpec:
    v = VocalSpec()
    has_female = any(k in text for k in ("женск", "female", "девуш", "женс. вокал"))
    has_male = any(k in text for k in ("мужск", "male", "мужс. вокал"))
    if has_female and not has_male:
        v.male, v.female = False, True
        v.female_sections = ["verse", "chorus", "break"]
    elif has_female and has_male:
        v.male, v.female = True, True
    elif has_male:
        v.male, v.female = True, False
        v.male_sections = ["verse", "chorus", "break"]
    if any(k in text for k in ("дуэт", "оба вокал", "и мужской и женский")):
        v.male = v.female = True

    for style, keys in VOCAL_STYLE_KEYWORDS.items():
        if any(k in text for k in keys):
            v.male_style = style
            break
    if "женский чистый" in text or "женский мелодич" in text:
        v.female_style = "clean"
    if "женский скрим" in text or "женский гроул" in text:
        v.female_style = "scream"
    if "автотюн" in text or "autotune" in text:
        v.autotune = True
    if "гармони" in text or "бэк-вокал" in text or "бэк вокал" in text:
        v.harmony = True
    if "без вокала" in text or "инструментал" in text:
        v.male = v.female = False
    return v


def apply_overrides(spec: ArrangementSpec, overrides: dict) -> ArrangementSpec:
    """Явные значения из UI/API перекрывают то, что вытащено из текста."""
    simple = ("tempo", "key", "mode", "groove", "aggression", "density", "swing",
              "tuning", "bass_tuning", "seed", "title", "tempo_scale", "transpose")

    genre = overrides.get("genre")
    if genre in GENRES:
        spec = apply_genre(spec, genre)
    for field_name in simple:
        if overrides.get(field_name) not in (None, ""):
            setattr(spec, field_name, overrides[field_name])

    if overrides.get("tuning"):
        for iid in ("guitar_rhythm", "guitar_rhythm_r", "guitar_lead"):
            item = spec.get(iid)
            if item:
                item.tuning = overrides["tuning"]
    if overrides.get("bass_tuning"):
        bass = spec.get("bass")
        if bass:
            bass.tuning = overrides["bass_tuning"]

    instruments = overrides.get("instruments")
    if isinstance(instruments, list) and instruments:
        wanted = set(instruments)
        if not spec.instruments:
            spec.instruments = default_instruments(spec.tuning, spec.bass_tuning)
        for item in spec.instruments:
            item.enabled = item.id in wanted
    if isinstance(overrides.get("instrument_params"), dict):
        for iid, params in overrides["instrument_params"].items():
            item = spec.get(iid)
            if item:
                item.gain_db = float(params.get("gain_db", item.gain_db))
                item.pan = float(params.get("pan", item.pan))
                item.params.update(params.get("params", {}))

    vocals = overrides.get("vocals")
    if isinstance(vocals, dict):
        current = spec.vocals.to_dict()
        current.update({k: v for k, v in vocals.items() if k in current})
        spec.vocals = VocalSpec(**current)
    return spec


def refine_with_llm(prompt: str, spec: ArrangementSpec) -> ArrangementSpec:
    """Опционально: уточняет спецификацию через LLM.

    Работает, только если задан ANTHROPIC_API_KEY и установлен пакет
    `anthropic`. При любой ошибке возвращает исходную спецификацию —
    правило-ориентированный парсер остаётся источником истины.
    """
    import json
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return spec
    try:  # pragma: no cover - внешний сервис
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        schema_hint = {
            "groove": list(GROOVE_KEYWORDS),
            "tuning": list(TUNINGS),
            "bass_tuning": list(BASS_TUNINGS),
            "instruments": list(INSTRUMENT_CATALOG),
            "vocals": spec.vocals.to_dict(),
        }
        msg = client.messages.create(
            model=os.getenv("VIBETRACK_LLM_MODEL", "claude-sonnet-5"),
            max_tokens=800,
            system=("Ты аранжировщик ню-метала. Верни ТОЛЬКО JSON с полями "
                    "groove, tuning, bass_tuning, aggression (0..1), instruments (список id), "
                    "vocals {male, female, male_style, female_style}. Доступные значения: "
                    + json.dumps(schema_hint, ensure_ascii=False)),
            messages=[{"role": "user", "content": prompt}],
        )
        payload = json.loads(msg.content[0].text)
        return apply_overrides(spec, payload)
    except Exception:
        return spec
