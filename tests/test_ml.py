"""Тесты слоя нейросетей: кэш моделей, откаты, Whisper, LLM-разбор.

Реальные веса здесь не нужны — модели подменяются заглушками. Проверяем
именно интеграцию: что кэш работает, что при отсутствии модели сервис не
падает, а откатывается, и что ручные настройки главнее решений LLM.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from engine import models
from engine.analysis import analyze
from engine.arrangement import parse_prompt
from engine.audio_io import Audio, load
from engine.llm import enabled as llm_enabled, parse_description, refine
from engine.separation import separate
from engine.transcription import assign_sections, transcribe

from .factories import make_track


def _noise(seconds: float = 3.0, sr: int = 44100) -> Audio:
    rng = np.random.default_rng(3)
    return Audio((rng.standard_normal((2, int(sr * seconds))) * 0.1).astype(np.float32), sr)


class TestModelCache(unittest.TestCase):
    def setUp(self):
        models.clear()

    def tearDown(self):
        models.clear()

    def test_model_is_built_once(self):
        calls = []

        def builder():
            calls.append(1)
            return object()

        first = models._cached("test:model", builder)
        second = models._cached("test:model", builder)
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1, "модель должна собираться один раз на процесс")

    def test_status_reports_availability(self):
        status = models.status()
        self.assertEqual(set(status), {"demucs", "whisper", "llm"})
        for info in status.values():
            self.assertIsInstance(info.available, bool)
            self.assertFalse(info.loaded)

    def test_torch_threads_are_capped(self):
        with mock.patch.dict(os.environ, {"VIBETRACK_TORCH_THREADS": "2"}):
            self.assertEqual(models.configure_torch_threads(), 2)


class TestSeparationFallback(unittest.TestCase):
    def setUp(self):
        models.clear()

    def test_falls_back_to_dsp_when_demucs_missing(self):
        with mock.patch("engine.separation.demucs_available", return_value=False):
            result = separate(_noise())
        self.assertEqual(result.backend, "dsp")
        self.assertEqual(set(result.stems), {"vocals", "drums", "bass", "other"})
        self.assertGreater(result.seconds, 0)

    def test_falls_back_when_demucs_raises(self):
        with mock.patch("engine.separation.demucs_available", return_value=True), \
             mock.patch("engine.separation._separate_demucs",
                        side_effect=RuntimeError("нет весов")):
            result = separate(_noise())
        self.assertEqual(result.backend, "dsp", "падение Demucs не должно ронять рендер")

    def test_fallback_explains_the_real_reason(self):
        """Пользователь должен видеть, почему дорожки хуже: пакета нет или веса не скачались."""
        with mock.patch("engine.separation.demucs_available", return_value=False):
            self.assertIn("не установлен", separate(_noise()).detail)
        with mock.patch("engine.separation.demucs_available", return_value=True), \
             mock.patch("engine.separation._separate_demucs",
                        side_effect=RuntimeError("Tunnel connection failed: 403")):
            self.assertIn("403", separate(_noise()).detail)
        self.assertEqual(separate(_noise(), prefer_backend="dsp").detail, "",
                         "осознанный выбор DSP — не повод для предупреждения")

    def test_demucs_result_is_used_when_available(self):
        stems = {"vocals": _noise(), "drums": _noise(), "bass": _noise(), "other": _noise()}
        fake = mock.Mock(return_value=type("R", (), {
            "stems": stems, "backend": "demucs", "model": "htdemucs",
            "seconds": 0.0, "detail": "",
        })())
        with mock.patch("engine.separation.demucs_available", return_value=True), \
             mock.patch("engine.separation._separate_demucs", fake):
            result = separate(_noise(), model="htdemucs")
        self.assertEqual(result.backend, "demucs")
        fake.assert_called_once()


class _FakeWord:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _FakeSegment:
    def __init__(self, text, start, end, words=None):
        self.text, self.start, self.end = text, start, end
        self.words = words or []


class TestWhisperIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="vibetrack-ml-")
        cls.audio = load(make_track(os.path.join(cls.tmp, "src.wav"), seconds=12))
        cls.analysis = analyze(cls.audio)

    def setUp(self):
        models.clear()

    def tearDown(self):
        models.clear()

    def test_transcribe_builds_lines_with_words(self):
        segments = [
            _FakeSegment("Я иду сквозь этот шум", 0.5, 3.0,
                         [_FakeWord("Я", 0.5, 0.7), _FakeWord("иду", 0.8, 1.2)]),
            _FakeSegment("  ", 3.0, 3.2),                     # пустой сегмент отбрасываем
            _FakeSegment("Мой голос рвётся", 3.5, 6.0),
        ]
        info = type("Info", (), {"language": "ru"})()
        fake_model = mock.Mock()
        fake_model.transcribe.return_value = (iter(segments), info)

        with mock.patch("engine.transcription.get_whisper", return_value=fake_model):
            lyrics = transcribe(self.audio, analysis=self.analysis)

        self.assertEqual(lyrics.backend, "faster-whisper")
        self.assertEqual(lyrics.language, "ru")
        self.assertEqual(len(lyrics.lines), 2)
        self.assertEqual(lyrics.lines[0].words[1].text, "иду")
        self.assertTrue(all(l.section for l in lyrics.lines), "строкам проставлена секция трека")
        kwargs = fake_model.transcribe.call_args.kwargs
        self.assertTrue(kwargs["word_timestamps"])
        self.assertFalse(kwargs["condition_on_previous_text"],
                         "иначе Whisper зацикливается на повторяющихся припевах")

    def test_missing_whisper_returns_unavailable(self):
        with mock.patch("engine.transcription.get_whisper",
                        side_effect=models.ModelUnavailable("нет пакета")):
            lyrics = transcribe(self.audio)
        self.assertEqual(lyrics.backend, "unavailable")
        self.assertEqual(lyrics.lines, [])

    def test_short_audio_is_skipped(self):
        self.assertEqual(transcribe(_noise(0.3)).backend, "none")

    def test_assign_sections_uses_track_structure(self):
        from engine.transcription import Lyrics, LyricLine

        lyrics = Lyrics(lines=[LyricLine("строка", 1.0, 2.0)])
        assign_sections(lyrics, self.analysis)
        self.assertEqual(lyrics.lines[0].section, self.analysis.sections[0].name)


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    stop_reason = "end_turn"

    def __init__(self, payload):
        self.content = [_FakeBlock(json.dumps(payload, ensure_ascii=False))]
        self.usage = type("U", (), {"input_tokens": 1500, "output_tokens": 300,
                                    "cache_read_input_tokens": 1200})()


class TestLlmParsing(unittest.TestCase):
    def setUp(self):
        models.clear()
        self.spec = parse_prompt("тяжёлый трек")
        self.payload = {
            "groove": "halftime",
            "tuning": "drop_b",
            "bass_tuning": "bass_5",
            "aggression": 0.9,
            "density": 0.8,
            "instruments": ["guitar_rhythm", "guitar_rhythm_r", "bass", "drums", "synth_pad"],
            "vocals": {"male": True, "female": True, "male_style": "scream",
                       "female_style": "clean", "autotune": False},
            "notes": "Медленно и тяжело, женский вокал в припеве.",
        }

    def tearDown(self):
        models.clear()

    def _client(self):
        client = mock.Mock()
        client.messages.create.return_value = _FakeResponse(self.payload)
        return client

    def test_enabled_flag_respects_env(self):
        with mock.patch.dict(os.environ, {"VIBETRACK_LLM_ENABLED": "0"}):
            self.assertFalse(llm_enabled())
        with mock.patch.dict(os.environ, {"VIBETRACK_LLM_ENABLED": "1"}):
            self.assertTrue(llm_enabled())

    def test_refine_applies_llm_answer(self):
        client = self._client()
        with mock.patch.dict(os.environ, {"VIBETRACK_LLM_ENABLED": "1"}), \
             mock.patch("engine.llm.get_anthropic_client", return_value=client):
            spec = refine(self.spec, "медленно и тяжело, как Deftones, женский припев")

        self.assertEqual(spec.groove, "halftime")
        self.assertEqual(spec.tuning, "drop_b")
        self.assertIn("synth_pad", spec.enabled_ids())
        self.assertTrue(spec.vocals.female)
        self.assertEqual(spec.vocals.male_style, "scream")
        self.assertTrue(any("LLM:" in n for n in spec.notes))

    def test_user_overrides_beat_the_model(self):
        client = self._client()
        with mock.patch.dict(os.environ, {"VIBETRACK_LLM_ENABLED": "1"}), \
             mock.patch("engine.llm.get_anthropic_client", return_value=client):
            spec = refine(self.spec, "любое описание",
                          user_overrides={"tuning": "drop_a_7", "groove": "bounce"})

        self.assertEqual(spec.tuning, "drop_a_7", "выбор в форме важнее ответа модели")
        self.assertEqual(spec.groove, "bounce")

    def test_request_uses_cache_and_schema(self):
        client = self._client()
        with mock.patch.dict(os.environ, {"VIBETRACK_LLM_ENABLED": "1"}), \
             mock.patch("engine.llm.get_anthropic_client", return_value=client):
            parse_description("что-то злое", self.spec)

        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["system"][0]["cache_control"], {"type": "ephemeral"},
                         "справочник должен кэшироваться — иначе платим за него каждый раз")
        self.assertEqual(kwargs["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(kwargs["output_config"]["effort"], "low")

    def test_api_failure_keeps_rule_based_spec(self):
        client = mock.Mock()
        client.messages.create.side_effect = RuntimeError("503")
        with mock.patch.dict(os.environ, {"VIBETRACK_LLM_ENABLED": "1"}), \
             mock.patch("engine.llm.get_anthropic_client", return_value=client):
            spec = refine(self.spec, "описание")
        self.assertEqual(spec.tuning, self.spec.tuning, "сбой LLM не меняет аранжировку")

    def test_refusal_is_ignored(self):
        response = _FakeResponse(self.payload)
        response.stop_reason = "refusal"
        client = mock.Mock()
        client.messages.create.return_value = response
        with mock.patch.dict(os.environ, {"VIBETRACK_LLM_ENABLED": "1"}), \
             mock.patch("engine.llm.get_anthropic_client", return_value=client):
            self.assertIsNone(parse_description("описание", self.spec))

    def test_disabled_llm_makes_no_call(self):
        client = self._client()
        with mock.patch.dict(os.environ, {"VIBETRACK_LLM_ENABLED": "0"}), \
             mock.patch("engine.llm.get_anthropic_client", return_value=client):
            refine(self.spec, "описание")
        client.messages.create.assert_not_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestDemucsContract(unittest.TestCase):
    """Проверяет наш код вокруг Demucs с настоящим torch, но с заглушкой весов.

    Веса Demucs (сотни мегабайт) в тестах не качаем, но контракт проверяем:
    тензор нужной формы и типа, ресемплинг в 44.1 кГц и обратно, выход в float32.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except Exception:
            raise unittest.SkipTest("torch не установлен — тест контракта пропущен")

    def setUp(self):
        models.clear()

    def test_tensor_contract_and_resampling(self):
        import torch

        from engine.separation import _separate_demucs

        captured = {}

        class FakeSeparator:
            samplerate = 44100

            def separate_tensor(self, wav, sr):
                captured["shape"] = tuple(wav.shape)
                captured["dtype"] = wav.dtype
                captured["sr"] = sr
                stems = {name: torch.zeros_like(wav) + 0.01
                         for name in ("vocals", "drums", "bass", "other")}
                return wav, stems

        source = Audio((np.random.default_rng(1).standard_normal((2, 22050 * 4)) * 0.1)
                       .astype(np.float32), 22050)
        with mock.patch("engine.separation.get_demucs", return_value=FakeSeparator()):
            result = _separate_demucs(source, "htdemucs")

        self.assertEqual(captured["sr"], 44100, "Demucs ждёт 44.1 кГц")
        self.assertEqual(captured["shape"][0], 2, "и ровно два канала")
        self.assertEqual(captured["dtype"], torch.float32)
        self.assertEqual(result.backend, "demucs")
        for name, stem in result.stems.items():
            self.assertEqual(stem.sr, source.sr, f"{name}: вернули в исходной частоте")
            self.assertEqual(stem.data.dtype, np.float32)
            self.assertAlmostEqual(stem.duration, source.duration, delta=0.05)

    def test_real_missing_weights_fall_back(self):
        """Реальный Demucs без весов не должен ронять рендер (проверено вживую)."""
        from engine.models import ModelUnavailable, get_demucs

        with mock.patch("engine.models._cached",
                        side_effect=ModelUnavailable("нет весов")):
            with self.assertRaises(ModelUnavailable):
                get_demucs("htdemucs")


class TestSampler(unittest.TestCase):
    """Разбор пользовательских сэмплов: луп → удары, гитара → нота."""

    @classmethod
    def setUpClass(cls):
        from engine import instruments as ins

        cls.sr = 44100
        loop = np.zeros(int(cls.sr * 4), dtype=np.float32)
        step = int(cls.sr * 0.25)
        voices = [ins.kick, lambda: ins.hihat(), ins.snare, lambda: ins.hihat()]
        for bar in range(4):
            for i, make in enumerate(voices):
                pos = bar * 4 * step + i * step
                hit = make()
                end = min(pos + len(hit), len(loop))
                loop[pos:end] += hit[: end - pos]
        cls.loop = Audio(loop, cls.sr)
        note = ins.amp(ins.pluck(ins.midi_to_hz(40), 1.0, cls.sr), ins.GuitarTone())
        cls.note = Audio(note, cls.sr)

    def test_loop_is_split_into_roles(self):
        from engine.sampler import slice_drum_loop

        kit = slice_drum_loop(self.loop, source="drums.wav")
        for role in ("kick", "snare", "hat"):
            self.assertIn(role, kit.hits, f"в лупе не найден {role}")
            self.assertTrue(all(h.size > 0 for h in kit.hits[role]))
        self.assertLessEqual(len(kit.hits["kick"]), 3, "храним не больше трёх вариантов на роль")

    def test_classifier_recognises_each_voice(self):
        from engine import instruments as ins
        from engine.sampler import classify_hit

        cases = {"kick": ins.kick(), "snare": ins.snare(), "hat": ins.hihat(),
                 "crash": ins.crash()}
        for expected, signal in cases.items():
            role, _ = classify_hit(signal, self.sr)
            self.assertEqual(role, expected)

    def test_guitar_pitch_is_detected(self):
        from engine import instruments as ins
        from engine.sampler import extract_guitar_note

        sample = extract_guitar_note(self.note, source="guitar.wav")
        self.assertIsNotNone(sample)
        self.assertAlmostEqual(sample.base_freq, ins.midi_to_hz(40), delta=3.0)

    def test_repitched_note_lands_on_target(self):
        from engine import instruments as ins
        from engine.sampler import extract_guitar_note, detect_pitch

        sample = extract_guitar_note(self.note, source="guitar.wav")
        target = ins.midi_to_hz(47)                     # на квинту выше
        played = sample.play(target, 0.6, self.sr)
        self.assertAlmostEqual(detect_pitch(played, self.sr), target, delta=target * 0.05)

    def test_palm_mute_is_shorter(self):
        from engine.sampler import extract_guitar_note

        sample = extract_guitar_note(self.note, source="guitar.wav")
        muted = sample.play(110.0, 0.5, self.sr, palm_mute=True)
        open_note = sample.play(110.0, 0.5, self.sr, palm_mute=False)
        self.assertLess(float(np.abs(muted).sum()), float(np.abs(open_note).sum()),
                        "заглушённая нота должна нести меньше энергии")

    def test_missing_directory_is_not_an_error(self):
        from engine.sampler import load_library

        library = load_library("/nonexistent/samples")
        self.assertFalse(library.has_drums)
        self.assertFalse(library.has_guitar)

    def test_library_reads_named_files(self):
        import tempfile

        from engine.audio_io import save
        from engine.sampler import load_library

        with tempfile.TemporaryDirectory() as tmp:
            save(os.path.join(tmp, "bitcrushed-drums.wav"), self.loop)
            save(os.path.join(tmp, "distorted-guitar.wav"), self.note)
            library = load_library(tmp)

        self.assertTrue(library.has_drums, "файл со словом drums должен стать китом")
        self.assertTrue(library.has_guitar, "файл со словом guitar — гитарой")
        self.assertEqual(len(library.notes), 2)

    def test_render_uses_samples_when_available(self):
        from engine.analysis import analyze
        from engine.arrangement import parse_prompt
        from engine.chords import recognize, to_power_chords
        from engine.rendering import render_arrangement
        from engine.sampler import load_library, slice_drum_loop, extract_guitar_note, SampleLibrary
        from engine.sequencer import sequence

        analysis = analyze(self.loop)
        spec = parse_prompt("ню-метал drop C", analysis)
        arrangement = sequence(analysis, to_power_chords(recognize(analysis)), spec)

        library = SampleLibrary(kit=slice_drum_loop(self.loop),
                                guitar=extract_guitar_note(self.note))
        with_samples = render_arrangement(arrangement, self.sr, library=library)
        synthesized = render_arrangement(arrangement, self.sr, library=None)

        self.assertIn("drums", with_samples)
        self.assertFalse(np.allclose(with_samples["drums"].data, synthesized["drums"].data),
                         "с сэмплами барабаны должны звучать иначе, чем синтез")
