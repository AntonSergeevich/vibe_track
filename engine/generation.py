"""Генерация кавера через внешний API.

Локально генеративную модель не запустить: нужна видеокарта и Python 3.11.
Зато почти все сервисы устроены одинаково — отправить трек, дождаться
готовности, скачать результат. Поэтому здесь один настраиваемый адаптер, а
не пять разных: провайдер меняется переменными окружения.

Важное ограничение, о котором легко забыть: генеративная модель возвращает
готовый микс. Ни отдельных дорожек, ни нот для табулатуры из него не
достать — поэтому кавер живёт рядом с нашим разбором, а не вместо него.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

from .audio_io import Audio, load

logger = logging.getLogger(__name__)

# Ориентиры стоимости за трек, доллары. Нужны, чтобы считать себестоимость
# рендера ещё до счёта от провайдера.
COST_HINTS = {
    "suno_proxy": 0.10,
    "elevenlabs": 0.53,
    "stability": 0.20,
    "custom": 0.20,
}


@dataclass
class CoverRequest:
    source_path: str
    style_prompt: str
    genre: str = ""
    duration: float = 0.0
    keep_vocals: bool = True


@dataclass
class CoverResult:
    audio: Audio | None = None
    provider: str = "none"
    cost_usd: float = 0.0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.audio is not None


class ProviderNotConfigured(RuntimeError):
    pass


def provider_name() -> str:
    return os.getenv("VIBETRACK_MUSIC_PROVIDER", "none").strip().lower()


def is_configured() -> bool:
    return provider_name() not in ("", "none") and bool(os.getenv("VIBETRACK_MUSIC_API_KEY"))


def generate_cover(request: CoverRequest, session=None) -> CoverResult:
    """Отправляет трек провайдеру и возвращает готовый кавер."""
    name = provider_name()
    if not is_configured():
        return CoverResult(provider=name or "none",
                           error="Провайдер генерации не настроен: задайте "
                                 "VIBETRACK_MUSIC_PROVIDER и VIBETRACK_MUSIC_API_KEY.")
    started = time.time()
    try:
        if name == "stability":
            audio = stability_generate(request, session=session)
        else:
            audio = _http_generate(request, session=session)
    except Exception as exc:  # noqa: BLE001 — внешний сервис не должен ронять рендер
        logger.warning("Генерация через %s не удалась: %s", name, exc)
        return CoverResult(provider=name, error=str(exc),
                           seconds=round(time.time() - started, 2))
    return CoverResult(audio=audio, provider=name,
                       cost_usd=COST_HINTS.get(name, COST_HINTS["custom"]),
                       seconds=round(time.time() - started, 2),
                       notes=[f"Кавер сгенерирован провайдером {name}"])


# ------------------------------------------------------------- Stability AI
STABILITY_URL = "https://api.stability.ai/v2beta/audio/stable-audio-2/audio-to-audio"
# Жёсткий предел модели: 190 секунд. Больше она не принимает вообще, поэтому
# длинный трек приходится обрезать — и честно об этом предупреждать.
STABILITY_MAX_SECONDS = 190.0


def stability_limit() -> float:
    return min(float(os.getenv("VIBETRACK_STABILITY_MAX_SECONDS", STABILITY_MAX_SECONDS)),
               STABILITY_MAX_SECONDS)


def prepare_input(source_path: str, max_seconds: float, sr: int = 44100) -> str:
    """Готовит вход для audio-to-audio: обрезка и выравнивание громкости.

    Провайдер работает с короткими фрагментами и заметно лучше отвечает на
    нормализованный по громкости материал, чем на сырой файл с диска.
    """
    import tempfile

    from .audio_io import save
    from .dsp import normalize_loudness

    audio = load(source_path, sr=sr)
    if audio.duration > max_seconds:
        audio = Audio(audio.data[:, : int(max_seconds * audio.sr)], audio.sr)
    audio = Audio(normalize_loudness(audio.data, -16.0, audio.sr), audio.sr)
    path = tempfile.mktemp(suffix=".wav")
    return save(path, audio)


def stability_generate(request: CoverRequest, session=None) -> Audio:
    """Stable Audio 2: превращает загруженный трек в другой стиль.

    Эндпоинт синхронный — аудио приходит телом ответа, опрашивать нечего.
    Поля запроса подтверждаются первым же реальным вызовом: при ошибке
    Stability отвечает понятным JSON, который команда `check_music_api`
    печатает целиком.
    """
    import requests

    session = session or requests.Session()
    url = os.getenv("VIBETRACK_STABILITY_URL", STABILITY_URL)
    key = os.getenv("VIBETRACK_MUSIC_API_KEY", "")
    max_seconds = stability_limit()
    duration = int(min(request.duration or max_seconds, max_seconds))

    prepared = prepare_input(request.source_path, max_seconds)
    try:
        with open(prepared, "rb") as fh:
            response = session.post(
                url,
                headers={"Authorization": f"Bearer {key}", "Accept": "audio/*"},
                files={"audio": fh},
                data={
                    "prompt": request.style_prompt,
                    # 0.5-0.6 — проверено слухом на реальном материале:
                    # инструменты уже звучат, а темп и форма ещё держатся
                    "strength": os.getenv("VIBETRACK_STABILITY_STRENGTH", "0.55"),
                    "duration": duration,
                    "output_format": os.getenv("VIBETRACK_STABILITY_FORMAT", "mp3"),
                    # stable-audio-2 принимает только 30-100: меньше — отказ 400
                    "steps": os.getenv("VIBETRACK_STABILITY_STEPS", "50"),
                },
                timeout=int(os.getenv("VIBETRACK_STABILITY_TIMEOUT", "600")),
            )
    finally:
        if os.path.exists(prepared):
            os.remove(prepared)

    if response.status_code == 422:
        # Фильтр авторских прав. Срабатывает и на чужой музыке, и на своей —
        # достаточно похожего фрагмента в их базе. Пользователю нужен не код
        # ответа, а понимание, что делать дальше.
        raise RuntimeError(
            "Нейросеть отказалась обрабатывать этот трек: её фильтр счёл "
            "материал защищённым авторским правом. Так бывает и со своими "
            "записями. Попробуйте другой фрагмент — короткий кусок проходит "
            "чаще, чем целая песня."
            f" (ответ сервиса: {_error_text(response)})")
    if response.status_code != 200:
        raise RuntimeError(f"Stability ответил {response.status_code}: "
                           f"{_error_text(response)}")

    import tempfile

    suffix = ".mp3" if os.getenv("VIBETRACK_STABILITY_FORMAT", "mp3") == "mp3" else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    try:
        return load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _error_text(response) -> str:
    """Текст ошибки провайдера в читаемом виде."""
    try:
        data = response.json()
    except Exception:
        return (getattr(response, "text", "") or str(response.content[:300]))[:500]
    for key in ("errors", "message", "error", "name"):
        if key in data:
            return json.dumps(data[key], ensure_ascii=False)[:500]
    return json.dumps(data, ensure_ascii=False)[:500]


def _config() -> dict:
    """Описание запросов провайдера.

    Живёт в переменной окружения как JSON, чтобы подключение нового сервиса
    не требовало правки кода:

        VIBETRACK_MUSIC_API_CONFIG='{"submit_url": "...", "audio_field": "audio",
          "prompt_field": "prompt", "job_id_path": "id",
          "status_url": "https://.../{job_id}", "status_path": "status",
          "done_values": ["succeeded"], "result_path": "output.audio"}'

    Для RunPod Serverless (и похожих JSON-only API без своего хостинга
    файлов) добавляются input_mode/output_mode/wrap_input_key -- пример
    целиком в .env.example, рядом с настоящими значениями полей ACE-Step
    впишутся, когда будет развёрнут реальный endpoint и виден его ответ.
    """
    raw = os.getenv("VIBETRACK_MUSIC_API_CONFIG", "").strip()
    config = json.loads(raw) if raw else {}
    config.setdefault("submit_url", os.getenv("VIBETRACK_MUSIC_API_URL", ""))
    config.setdefault("audio_field", "audio")
    config.setdefault("prompt_field", "prompt")
    config.setdefault("job_id_path", "id")
    config.setdefault("status_path", "status")
    config.setdefault("done_values", ["succeeded", "completed", "done", "complete",
                                       "COMPLETED"])
    config.setdefault("failed_values", ["failed", "error", "canceled", "FAILED"])
    config.setdefault("result_path", "output")
    config.setdefault("poll_seconds", 5)
    config.setdefault("timeout_seconds", 600)
    # JSON-only сервисы вроде RunPod Serverless не принимают multipart и не
    # хостят файлы у себя -- аудио едет и приходит base64-строкой прямо в
    # теле, а не по отдельной ссылке. "input_mode"/"output_mode" переключают
    # транспорт, не трогая опрос статуса и остальную логику.
    config.setdefault("input_mode", "multipart")     # или "json_base64"
    config.setdefault("output_mode", "url")           # или "base64"
    # RunPod оборачивает тело запроса в {"input": {...}} -- у большинства
    # остальных сервисов этого нет, поэтому по умолчанию не оборачиваем.
    config.setdefault("wrap_input_key", "")
    if not config["submit_url"]:
        raise ProviderNotConfigured("Не задан адрес API генерации")
    return config


def _dig(data, path: str):
    """Достаёт значение по пути вида 'output.audio.url'."""
    value = data
    for key in path.split("."):
        if isinstance(value, list):
            value = value[int(key)] if key.isdigit() else (value[0] if value else None)
            continue
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None
    return value


def _http_generate(request: CoverRequest, session=None) -> Audio:
    """
    Отправить → дождаться → скачать. Общая схема почти всех сервисов.

    Два входных и два выходных режима (config["input_mode"]/["output_mode"]):
    классический REST шлёт файл multipart-формой и отдаёт результат по
    ссылке, а JSON-only сервисы вроде RunPod Serverless своих файлов не
    хостят вовсе -- всё, включая исходное аудио и готовый кавер, едет
    base64-строкой прямо внутри JSON. Опрос статуса и обработка ошибок от
    этого не зависят и написаны один раз.
    """
    import requests

    session = session or requests.Session()
    config = _config()
    key = os.getenv("VIBETRACK_MUSIC_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}"}
    extra = os.getenv("VIBETRACK_MUSIC_API_HEADERS", "").strip()
    if extra:
        headers.update(json.loads(extra))

    payload = {
        config["prompt_field"]: request.style_prompt,
        "genre": request.genre,
        "duration": request.duration,
        "keep_vocals": request.keep_vocals,
    }
    payload.update(json.loads(os.getenv("VIBETRACK_MUSIC_API_EXTRA", "{}") or "{}"))

    if config["input_mode"] == "json_base64":
        import base64

        with open(request.source_path, "rb") as fh:
            payload[config["audio_field"]] = base64.b64encode(fh.read()).decode("ascii")
        body = {config["wrap_input_key"]: payload} if config["wrap_input_key"] else payload
        response = session.post(config["submit_url"], headers=headers, json=body, timeout=120)
    else:
        with open(request.source_path, "rb") as fh:
            response = session.post(config["submit_url"], headers=headers, data=payload,
                                    files={config["audio_field"]: fh}, timeout=120)
    response.raise_for_status()
    submitted = response.json()

    audio_bytes = _extract_audio(submitted, config, session, headers)
    if audio_bytes is None:                  # синхронного ответа не было — опрашиваем
        audio_bytes = _poll(session, headers, config, submitted)

    return _bytes_to_audio(audio_bytes)


def _extract_audio(data: dict, config: dict, session, headers: dict) -> bytes | None:
    """
    Достать готовое аудио из ответа -- по ссылке или из base64 в теле.

    None означает "результата тут нет" -- не ошибка, а сигнал опрашивать
    статус дальше: у синхронных сервисов результат уже в первом ответе,
    у асинхронных появится только когда задача досчитает.
    """
    value = _dig(data, config["result_path"])
    if not value:
        return None
    if config["output_mode"] == "base64":
        import base64

        if isinstance(value, str) and value.startswith("data:") and "," in value:
            value = value.split(",", 1)[1]    # data:audio/wav;base64,XXXX — срезаем префикс
        return base64.b64decode(value)
    downloaded = session.get(value, headers=headers, timeout=300)
    downloaded.raise_for_status()
    return downloaded.content


def _bytes_to_audio(data: bytes) -> Audio:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _poll(session, headers: dict, config: dict, submitted: dict) -> bytes:
    job_id = _dig(submitted, config["job_id_path"])
    if job_id is None:
        raise RuntimeError(f"В ответе нет идентификатора задачи: {submitted}")

    status_url = config.get("status_url", "").format(job_id=job_id)
    if not status_url:
        raise RuntimeError("Не задан адрес проверки статуса (status_url)")

    deadline = time.time() + config["timeout_seconds"]
    while time.time() < deadline:
        response = session.get(status_url, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        status = str(_dig(data, config["status_path"]) or "").lower()

        if status in [s.lower() for s in config["failed_values"]]:
            raise RuntimeError(f"Провайдер вернул статус «{status}»: {data}")
        if status in [s.lower() for s in config["done_values"]]:
            audio_bytes = _extract_audio(data, config, session, headers)
            if audio_bytes is None:
                raise RuntimeError(f"Задача готова, но нет результата: {data}")
            return audio_bytes
        time.sleep(config["poll_seconds"])
    raise TimeoutError("Провайдер не ответил за отведённое время")


# Модель охотно выкидывает исходную музыку и играет «мясо» с первой секунды.
# Эта приписка возвращает ей то, ради чего человек и принёс свой трек:
# его мелодию, гармонию и форму песни — включая тихое вступление.
FAITHFUL_SUFFIX = ("follow the original melody, chord progression and song structure, "
                   "keep the dynamics of the source: quiet intro stays quiet, "
                   "heavy sections hit hard")


def style_prompt_for(genre: str, spec_notes: str = "") -> str:
    """Текстовое описание стиля на английском — его понимают все сервисы."""
    prompts = {
        "nu_metal": "aggressive nu metal, downtuned seven-string guitars, syncopated groove, heavy drums",
        "metalcore": "modern metalcore, tight palm-muted riffs, double kick drums, breakdown",
        "alt_rock": "alternative rock, melodic guitars, driving drums, anthemic",
        "grunge": "90s grunge, dirty distorted guitars, loose drums, raw production",
        "punk": "fast punk rock, simple power chords, energetic drums",
        "industrial": "industrial metal, mechanical groove, synth stabs, heavy guitars",
        "trap_metal": "trap metal, 808 bass, sparse heavy guitars, half-time beat",
        "hard_rock": "classic hard rock, riff driven, live drums",
    }
    base = prompts.get(genre, "heavy rock with distorted guitars")
    return f"{base}. {FAITHFUL_SUFFIX}. {spec_notes}".strip().rstrip(".") + "."
