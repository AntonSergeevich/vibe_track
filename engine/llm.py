"""Разбор свободного описания аранжировки через Claude.

Правило-ориентированный парсер (`engine.arrangement.parse_prompt`) остаётся
источником истины и работает всегда. LLM — уточняющий слой поверх него:
понимает то, чего нет в словаре ключевых слов («злое, как ранний Slipknot,
но с трип-хоп битом»). Любая ошибка вызова — молча возвращаем исходную
спецификацию, рендер не должен падать из-за внешнего сервиса.

Порядок приоритетов: правила → LLM → явные переключатели из формы.
Пользовательский выбор в интерфейсе всегда главнее любой модели.
"""
from __future__ import annotations

import json
import logging
import os

from .analysis import TrackAnalysis
from .arrangement import (ArrangementSpec, BASS_TUNINGS, GROOVE_KEYWORDS,
                          INSTRUMENT_CATALOG, TUNINGS, VOCAL_STYLE_KEYWORDS,
                          apply_overrides)
from .models import ModelUnavailable, get_anthropic_client, llm_available

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 2000

SYSTEM_PROMPT = """Ты — аранжировщик тяжёлой музыки. По описанию от музыканта ты
подбираешь параметры аранжировки из фиксированного набора значений.

Правила:
- Выбирай ТОЛЬКО из перечисленных значений, ничего не выдумывай.
- aggression: 0.2 — сдержанно и атмосферно, 1.0 — предельно жёстко.
- density: 0.3 — редкие акценты, 1.0 — сплошной поток шестнадцатых.
- instruments: перечисли всё, что должно звучать. Ритм-гитары, бас и барабаны
  нужны почти всегда; guitar_rhythm_r — второй дубль гитары для широкой панорамы.
- vocals: male/female — кто поёт; стили: rap (читка), scream (скрим/гроул),
  clean (чистый), whisper (шёпот).
- notes: одно короткое предложение по-русски о том, как ты понял задачу.

Справочник значений:
"""


def _catalog() -> str:
    return json.dumps({
        "groove": list(GROOVE_KEYWORDS),
        "tuning": list(TUNINGS),
        "bass_tuning": list(BASS_TUNINGS),
        "instruments": {k: v["label"] for k, v in INSTRUMENT_CATALOG.items()},
        "vocal_styles": list(VOCAL_STYLE_KEYWORDS),
    }, ensure_ascii=False, indent=1)


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "groove": {"type": "string", "enum": list(GROOVE_KEYWORDS)},
            "tuning": {"type": "string", "enum": list(TUNINGS)},
            "bass_tuning": {"type": "string", "enum": list(BASS_TUNINGS)},
            "aggression": {"type": "number", "minimum": 0.2, "maximum": 1.0},
            "density": {"type": "number", "minimum": 0.2, "maximum": 1.0},
            "instruments": {
                "type": "array",
                "items": {"type": "string", "enum": list(INSTRUMENT_CATALOG)},
            },
            "vocals": {
                "type": "object",
                "properties": {
                    "male": {"type": "boolean"},
                    "female": {"type": "boolean"},
                    "male_style": {"type": "string", "enum": list(VOCAL_STYLE_KEYWORDS)},
                    "female_style": {"type": "string", "enum": list(VOCAL_STYLE_KEYWORDS)},
                    "autotune": {"type": "boolean"},
                },
                "required": ["male", "female", "male_style", "female_style", "autotune"],
                "additionalProperties": False,
            },
            "notes": {"type": "string"},
        },
        "required": ["groove", "tuning", "bass_tuning", "aggression", "density",
                     "instruments", "vocals", "notes"],
        "additionalProperties": False,
    }


def enabled() -> bool:
    flag = os.getenv("VIBETRACK_LLM_ENABLED", "auto").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    return llm_available()          # auto: включаем, если есть ключ и пакет


def _user_message(prompt: str, spec: ArrangementSpec, analysis: TrackAnalysis | None) -> str:
    facts = [f"Описание музыканта: {prompt.strip() or '(пусто)'}"]
    if analysis is not None:
        facts.append(f"Исходный трек: {analysis.tempo:.0f} BPM, тональность {analysis.key_name}, "
                     f"длительность {analysis.duration:.0f} с, "
                     f"секций {len(analysis.sections)}.")
    facts.append("Предварительный разбор по ключевым словам (можешь исправить): "
                 + json.dumps({"groove": spec.groove, "tuning": spec.tuning,
                               "bass_tuning": spec.bass_tuning,
                               "aggression": round(spec.aggression, 2),
                               "instruments": spec.enabled_ids(),
                               "vocals": spec.vocals.to_dict()}, ensure_ascii=False))
    return "\n\n".join(facts)


def parse_description(prompt: str, spec: ArrangementSpec,
                      analysis: TrackAnalysis | None = None) -> dict | None:
    """Возвращает словарь переопределений или None, если LLM недоступна."""
    if not prompt.strip() or not enabled():
        return None
    try:
        client = get_anthropic_client()
    except ModelUnavailable as exc:
        logger.info("LLM-разбор пропущен: %s", exc)
        return None

    model = os.getenv("VIBETRACK_LLM_MODEL", DEFAULT_MODEL)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            # справочник значений одинаков от запроса к запросу — кэшируем его
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT + _catalog(),
                "cache_control": {"type": "ephemeral"},
            }],
            output_config={
                "effort": "low",                       # задача простая, платить за глубину незачем
                "format": {"type": "json_schema", "schema": _schema()},
            },
            messages=[{"role": "user", "content": _user_message(prompt, spec, analysis)}],
        )
    except Exception as exc:  # noqa: BLE001 — внешний сервис не должен ронять рендер
        logger.warning("LLM-разбор не удался (%s), остаёмся на правилах", exc)
        return None

    if getattr(response, "stop_reason", None) == "refusal":
        logger.warning("LLM отказалась разбирать описание")
        return None
    try:
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
    except (StopIteration, ValueError, AttributeError) as exc:
        logger.warning("Не разобрал ответ LLM: %s", exc)
        return None

    usage = getattr(response, "usage", None)
    if usage is not None:
        logger.info("LLM-разбор: %s вход / %s выход / %s из кэша",
                    getattr(usage, "input_tokens", "?"),
                    getattr(usage, "output_tokens", "?"),
                    getattr(usage, "cache_read_input_tokens", 0))
    return data


def refine(spec: ArrangementSpec, prompt: str, analysis: TrackAnalysis | None = None,
           user_overrides: dict | None = None) -> ArrangementSpec:
    """Уточняет спецификацию через LLM, сохраняя приоритет ручных настроек."""
    data = parse_description(prompt, spec, analysis)
    if not data:
        return spec

    note = data.pop("notes", "")
    spec = apply_overrides(spec, data)
    if note:
        spec.notes.append(f"LLM: {note}")
    if user_overrides:
        # то, что пользователь выставил руками, важнее того, что решила модель
        spec = apply_overrides(spec, user_overrides)
    return spec
