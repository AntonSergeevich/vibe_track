"""Тесты аудиодвижка: анализ, аккорды, аранжировка, вокал, полный конвейер."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from engine.analysis import analyze
from engine.arrangement import parse_prompt
from engine.audio_io import Audio, load, save
from engine.chords import recognize, to_power_chords
from engine.dsp import normalize_active_rms, pan, pitch_shift, rms_db, sidechain_duck
from engine.mixing import build_gains, mixdown
from engine.render import RenderOptions, add_vocal_take, transform
from engine.rendering import render_arrangement
from engine.separation import separate
from engine.sequencer import power_chord_shape, sequence
from engine.tabs import chord_chart, render_tab
from engine.transcription import lyric_sheet, lyrics_from_text
from engine.vocals import estimate_latency, process_take

from .factories import make_track, make_voice


class EngineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="vibetrack-tests-")
        cls.track_path = make_track(os.path.join(cls.tmp, "src.wav"), seconds=12)
        cls.voice_path = make_voice(os.path.join(cls.tmp, "voice.wav"))
        cls.audio = load(cls.track_path)
        cls.analysis = analyze(cls.audio)
        cls.chords = recognize(cls.analysis)


class TestAnalysis(EngineTestCase):
    def test_tempo_detected(self):
        self.assertAlmostEqual(self.analysis.tempo, 120.0, delta=6.0)

    def test_key_detected(self):
        # Am и C — параллельные тональности, обе считаем верным ответом
        self.assertIn(self.analysis.key_name, ("A minor", "C major"))

    def test_beat_grid(self):
        self.assertGreater(len(self.analysis.beats), 10)
        deltas = np.diff(self.analysis.beats)
        self.assertAlmostEqual(float(deltas.mean()), 0.5, delta=0.06)


class TestChords(EngineTestCase):
    def test_progression(self):
        names = [c.name for c in self.chords][:4]
        self.assertEqual(names, ["Am", "F", "C", "G"])

    def test_power_chords(self):
        power = to_power_chords(self.chords)
        self.assertTrue(all(c.quality == "5" for c in power))
        self.assertEqual(power[0].name, "A5")


class TestArrangement(EngineTestCase):
    def test_prompt_parsing(self):
        spec = parse_prompt(
            "ню-метал в духе Korn, семиструнная гитара в drop A, пятиструнный бас, "
            "скретчи и бонго, мужская читка и женский чистый вокал, 95 bpm",
            self.analysis)
        self.assertEqual(spec.tuning, "drop_a_7")
        self.assertEqual(spec.bass_tuning, "bass_5")
        self.assertEqual(spec.groove, "syncopated")
        self.assertAlmostEqual(spec.tempo, 95.0)
        self.assertIn("turntables", spec.enabled_ids())
        self.assertIn("percussion", spec.enabled_ids())
        self.assertTrue(spec.vocals.male and spec.vocals.female)
        self.assertEqual(spec.vocals.male_style, "rap")

    def test_overrides_win_over_text(self):
        spec = parse_prompt("drop C, только гитары", self.analysis,
                            {"tuning": "drop_b", "instruments": ["drums", "bass"]})
        self.assertEqual(spec.tuning, "drop_b")
        self.assertEqual(sorted(spec.enabled_ids()), ["bass", "drums"])

    def test_instrumental_prompt_disables_vocals(self):
        spec = parse_prompt("инструментал без вокала", self.analysis)
        self.assertFalse(spec.vocals.male)
        self.assertFalse(spec.vocals.female)


class TestSequencerAndTabs(EngineTestCase):
    def setUp(self):
        self.spec = parse_prompt("ню-метал drop C, соло-гитара", self.analysis)
        self.arrangement = sequence(self.analysis, to_power_chords(self.chords), self.spec)

    def test_parts_generated(self):
        ids = {p.instrument_id for p in self.arrangement.parts}
        self.assertIn("guitar_rhythm", ids)
        self.assertIn("drums", ids)
        guitar = self.arrangement.part("guitar_rhythm")
        self.assertGreater(len(guitar.notes), 20)
        self.assertTrue(any(n.palm_mute for n in guitar.notes))

    def test_power_chord_shape_in_drop_tuning(self):
        shape = power_chord_shape(9, "drop_c")           # A5 в строе Drop C
        frets = {s: f for s, f, _ in shape}
        self.assertEqual(frets[0], 9)
        self.assertEqual(frets[1], 9)                    # квинта на том же ладу

    def test_tab_matches_notes(self):
        tab = render_tab(self.arrangement, "guitar_rhythm")
        self.assertIn("drop_c", tab)
        self.assertIn("palm mute", tab)
        self.assertIn("|", tab)

    def test_chord_chart(self):
        self.assertIn("Am", chord_chart(self.chords))


class TestSeparation(EngineTestCase):
    def test_dsp_backend_returns_four_stems(self):
        result = separate(self.audio, prefer_backend="dsp")
        self.assertEqual(result.backend, "dsp")
        self.assertEqual(set(result.stems), {"vocals", "drums", "bass", "other"})
        for name, stem in result.stems.items():
            self.assertAlmostEqual(stem.duration, self.audio.duration, delta=0.2, msg=name)
            self.assertLess(stem.peak(), 4.0, msg=f"{name} перегружен")


class TestDsp(EngineTestCase):
    def test_normalize_active_rms_ignores_spikes(self):
        sig = np.zeros((1, 44100), dtype=np.float32)
        sig[0, :22050] = 0.1
        sig[0, 30000] = 3.0                              # выброс
        out = normalize_active_rms(sig, -18.0)
        self.assertLess(abs(rms_db(out[0, :22050]) + 18.0), 3.0)

    def test_pitch_shift_keeps_length_and_level(self):
        sig = self.audio.data[:, :44100]
        shifted = pitch_shift(sig, 5.0)
        self.assertAlmostEqual(shifted.shape[-1], sig.shape[-1], delta=2000)
        self.assertLess(float(np.max(np.abs(shifted))), float(np.max(np.abs(sig))) * 2.1)

    def test_sidechain_duck_reduces_level(self):
        target = np.ones((2, 44100), dtype=np.float32) * 0.5
        trigger = np.ones((2, 44100), dtype=np.float32) * 0.8
        ducked = sidechain_duck(target, trigger, amount_db=-6.0)
        self.assertLess(rms_db(ducked), rms_db(target) - 3.0)


class TestVocals(EngineTestCase):
    def test_latency_estimation(self):
        # дубль пишется вместе с минусовкой в наушниках: у него та же ритмика
        reference = self.audio
        shift = int(reference.sr * 0.12)
        delayed = Audio(np.concatenate(
            [np.zeros((2, shift), dtype=np.float32), reference.data], axis=1), reference.sr)
        self.assertAlmostEqual(estimate_latency(delayed, reference), 120.0, delta=8.0)

    def test_uncorrelated_take_reports_no_latency(self):
        take = load(self.voice_path)
        noise = Audio((np.random.default_rng(1).standard_normal((2, take.n_samples)) * 0.1)
                      .astype(np.float32), take.sr)
        self.assertEqual(estimate_latency(take, noise), 0.0)

    def test_process_take_is_clean_and_loud_enough(self):
        take = load(self.voice_path)
        result = process_take(take, style="rap", analysis=self.analysis)
        self.assertLessEqual(result.peak_db, 0.0)
        self.assertGreater(rms_db(result.audio.data), -30.0)

    def test_gender_flip_changes_spectrum(self):
        take = load(self.voice_path)
        male = process_take(take, style="clean", analysis=self.analysis).audio
        female = process_take(take, style="clean", analysis=self.analysis,
                              gender="male", target_gender="female").audio
        centroid = lambda a: float(np.sum(np.abs(np.fft.rfft(a.mono()))
                                          * np.fft.rfftfreq(a.n_samples, 1 / a.sr))
                                   / max(np.sum(np.abs(np.fft.rfft(a.mono()))), 1e-9))
        self.assertGreater(centroid(female), centroid(male))


class TestMixing(EngineTestCase):
    def test_build_gains_applies_user_trim(self):
        spec = parse_prompt("ню-метал drop C", self.analysis)
        spec.get("bass").gain_db += 3.0                  # пользователь поднял бас
        gains = build_gains(spec)
        self.assertAlmostEqual(gains["bass"], -3.0 + 3.0)

    def test_consume_frees_input_stems(self):
        """consume=True не должен оставлять в памяти вторую копию всего микса."""
        spec = parse_prompt("ню-метал drop C", self.analysis)
        arrangement = sequence(self.analysis, to_power_chords(self.chords), spec)
        stems = render_arrangement(arrangement)
        names = set(stems)

        master, processed = mixdown(stems, consume=True)
        self.assertEqual(stems, {}, "исходные дорожки освобождены по мере сведения")
        self.assertEqual(set(processed), names, "но результат содержит все дорожки")
        self.assertLessEqual(master.peak(), 1.0)

    def test_master_does_not_clip(self):
        spec = parse_prompt("ню-метал drop C, скретчи", self.analysis)
        arrangement = sequence(self.analysis, to_power_chords(self.chords), spec)
        stems = render_arrangement(arrangement)
        master, processed = mixdown(stems)
        self.assertLessEqual(master.peak(), 1.0)
        self.assertGreater(rms_db(master.data), -20.0)
        for name, stem in processed.items():
            self.assertLessEqual(stem.peak(), 1.0, msg=name)


class TestLyrics(EngineTestCase):
    def test_manual_lyrics_get_chords(self):
        lyrics = lyrics_from_text("Первая строка\nВторая строка", self.analysis)
        sheet = lyric_sheet(lyrics, self.chords)
        self.assertIn("Первая строка", sheet)
        self.assertTrue(any(c.name in sheet for c in self.chords))


class TestFullPipeline(EngineTestCase):
    def test_transform_gives_analysis_without_synthesising_a_band(self):
        """По умолчанию синтеза нет: он не звучит группой и в релиз не годится.

        Остаётся то, что действительно полезно: дорожки исходника, аккорды,
        табы и текст.
        """
        out_dir = os.path.join(self.tmp, "render")
        result = transform(self.track_path,
                           "ню-метал в духе Korn, drop C, скретчи, женский вокал в припеве",
                           out_dir=out_dir,
                           options=RenderOptions(manual_lyrics="Строка раз\nСтрока два",
                                                 separation_backend="dsp", use_llm=False))
        self.assertTrue(os.path.exists(result.master_path))
        self.assertIn("source_vocals", result.stem_paths)
        self.assertNotIn("guitar_rhythm", result.stem_paths,
                         "синтезированные дорожки больше не отдаём")
        self.assertTrue(result.tabs.get("guitar_rhythm"), "табы считаются без синтеза")
        self.assertIn("Строка раз", result.lyric_sheet)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "score", "lyrics_chords.txt")))

    def test_synth_arrangement_still_available_on_request(self):
        """Старый путь никуда не делся — он просто не по умолчанию."""
        out_dir = os.path.join(self.tmp, "render-synth")
        result = transform(self.track_path, "ню-метал drop C", out_dir=out_dir,
                           options=RenderOptions(separation_backend="dsp", use_llm=False,
                                                 transcribe_lyrics=False,
                                                 synth_arrangement=True))
        self.assertIn("guitar_rhythm", result.stem_paths)
        self.assertIn("drums", result.stem_paths)
        master = load(result.master_path)
        self.assertLessEqual(master.peak(), 1.0)
        self.assertAlmostEqual(master.duration, self.audio.duration, delta=4.0)

    def test_add_vocal_take_mixes_into_instrumental(self):
        out_dir = os.path.join(self.tmp, "take")
        result = add_vocal_take(self.track_path, self.voice_path, out_dir,
                                style="scream", gender="male", target_gender="female")
        self.assertTrue(os.path.exists(result["master_path"]))
        self.assertTrue(os.path.exists(result["vocal_path"]))
        self.assertLessEqual(load(result["master_path"]).peak(), 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class TestTranspose(EngineTestCase):
    """Смена тональности: и из описания, и регулятором."""

    def test_prompt_parsing(self):
        self.assertEqual(parse_prompt("ню-метал, на 2 полутона ниже").transpose, -2)
        self.assertEqual(parse_prompt("ню-метал, на тон выше").transpose, 2)
        self.assertEqual(parse_prompt("ню-метал, на полтона ниже").transpose, -1)
        self.assertEqual(parse_prompt("ню-метал drop C").transpose, 0)

    def test_override_wins(self):
        self.assertEqual(parse_prompt("на тон выше", None, {"transpose": -5}).transpose, -5)

    def test_riff_notes_move_with_key(self):
        from engine.chords import transpose as transpose_chords

        base = parse_prompt("ню-метал drop C", self.analysis)
        power = to_power_chords(self.chords)
        low = sequence(self.analysis, transpose_chords(power, -3), base)
        high = sequence(self.analysis, transpose_chords(power, 2), base)

        low_note = low.part("guitar_rhythm").notes[0]
        high_note = high.part("guitar_rhythm").notes[0]
        self.assertNotEqual(low_note.fret, high_note.fret,
                            "сдвиг тональности должен менять лад, а не только звук")


class TestChordSmoothing(EngineTestCase):
    """Аккордов должно быть столько, сколько слышит человек, а не сколько видит спектр."""

    def test_viterbi_holds_chord_through_flicker(self):
        """Один дрогнувший такт не должен рождать лишний аккорд."""
        from engine.chords import _viterbi

        # четыре такта: на третьем шум чуть перевесил в сторону второго аккорда
        scores = np.array([[0.90, 0.60],
                           [0.88, 0.62],
                           [0.70, 0.74],
                           [0.89, 0.61]], dtype=np.float32)

        self.assertEqual(_viterbi(scores, change_penalty=0.0), [0, 0, 1, 0],
                         "без штрафа каждый такт выбирается сам по себе")
        self.assertEqual(_viterbi(scores, change_penalty=0.12), [0, 0, 0, 0],
                         "со штрафом последовательность остаётся целой")

    def test_viterbi_keeps_real_chord_change(self):
        """Настоящая смена аккорда штрафом не затирается."""
        from engine.chords import _viterbi

        scores = np.array([[0.92, 0.40],
                           [0.90, 0.42],
                           [0.38, 0.95],
                           [0.40, 0.93]], dtype=np.float32)
        self.assertEqual(_viterbi(scores, change_penalty=0.12), [0, 0, 1, 1])

    def test_simple_vocabulary_has_no_extensions(self):
        from engine.chords import recognize

        for chord in recognize(self.analysis):
            self.assertIn(chord.quality, ("", "m"),
                          f"{chord.name}: в песеннике пишут трезвучия")

    def test_key_prior_prefers_diatonic_quality(self):
        from engine.chords import _key_prior

        labels = [(7, ""), (7, "m"), (6, "")]      # G, Gm, F# в тональности до мажор
        prior = _key_prior(labels, "C", "major")
        self.assertGreater(prior[0], prior[1], "V ступень мажорная, а не минорная")
        self.assertEqual(prior[2], 0.0, "чужой тон подсказки не получает")

    def test_short_chords_are_absorbed(self):
        from engine.chords import ChordEvent, _drop_short

        events = [ChordEvent(0.0, 2.0, 9, "m"), ChordEvent(2.0, 2.1, 5, ""),
                  ChordEvent(2.1, 4.0, 0, "")]
        merged = _drop_short(events, min_duration=1.0)
        self.assertEqual([e.name for e in merged], ["Am", "C"])


class MemoryGuardTests(unittest.TestCase):
    """Проверка памяти перед Demucs: молчаливая смерть процесса недопустима."""

    def test_short_track_passes_on_a_roomy_machine(self):
        from unittest import mock

        from engine import system

        with mock.patch.object(system, "available_memory_mb", return_value=16000):
            ok, detail = system.enough_memory_for_demucs(200)
        self.assertTrue(ok)
        self.assertEqual(detail, "")

    def test_long_track_on_a_tight_machine_is_refused_with_numbers(self):
        from unittest import mock

        from engine import system

        with mock.patch.object(system, "available_memory_mb", return_value=1500):
            ok, detail = system.enough_memory_for_demucs(200)
        self.assertFalse(ok)
        self.assertIn("ГБ", detail, "человеку нужны цифры, а не «мало памяти»")

    def test_unknown_memory_never_blocks(self):
        from unittest import mock

        from engine import system

        with mock.patch.object(system, "available_memory_mb", return_value=None):
            ok, _ = system.enough_memory_for_demucs(600)
        self.assertTrue(ok, "не знать объём памяти — не повод отказывать")

    def test_separation_falls_back_to_dsp_and_says_why(self):
        from unittest import mock

        from engine import separation
        from engine.audio_io import Audio

        audio = Audio(np.zeros((2, 44100 * 3), dtype=np.float32), 44100)
        with mock.patch.object(separation, "enough_memory_for_demucs",
                               return_value=(False, "свободно 1.0 ГБ, нужно 2.5 ГБ")), \
             mock.patch.object(separation, "demucs_available", return_value=True):
            result = separation.separate(audio)
        self.assertEqual(result.backend, "dsp")
        self.assertIn("памяти", result.detail)


class SeparationMemoryTests(unittest.TestCase):
    """Маска центра считается блоками — результат обязан не измениться."""

    def test_blocked_mask_matches_whole_track_math(self):
        from engine.separation import _spectral_center_mask

        rng = np.random.default_rng(3)
        left = (rng.standard_normal(44100 * 3) * 0.2).astype(np.float32)
        right = (left * 0.8 + rng.standard_normal(44100 * 3) * 0.05).astype(np.float32)

        whole = _spectral_center_mask(left, right, 44100, block=10 ** 6)
        blocked = _spectral_center_mask(left, right, 44100, block=16)
        self.assertEqual(whole.shape, blocked.shape)
        # расхождение только на уровне шума самого БПФ (порядок 1e-7):
        # значения маски лежат в 0..1, слышать там нечего
        self.assertLess(float(np.max(np.abs(whole - blocked))), 1e-6,
                        "разбиение на блоки изменило маску")
        self.assertEqual(blocked.dtype, np.float32)

    def test_hpss_blocks_do_not_change_the_result(self):
        from engine.separation import hpss

        rng = np.random.default_rng(11)
        y = (rng.standard_normal(44100 * 2) * 0.15).astype(np.float32)
        y += (np.sin(np.linspace(0, 3000, y.size)) * 0.3).astype(np.float32)

        whole_h, whole_p = hpss(y, 44100, block=10 ** 6)
        blocked_h, blocked_p = hpss(y, 44100, block=32)
        for name, a, b in (("гармоника", whole_h, blocked_h),
                           ("перкуссия", whole_p, blocked_p)):
            self.assertLess(float(np.max(np.abs(a - b))), 1e-5, name)

    def test_hpss_separates_tone_from_clicks(self):
        """Смысл разделения: тон уходит в гармонику, щелчки — в перкуссию."""
        from engine.dsp import rms_db
        from engine.separation import hpss

        sr = 44100
        t = np.arange(sr * 2) / sr
        tone = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.4
        clicks = np.zeros_like(tone)
        clicks[::sr // 4] = 1.0
        harmonic, percussive = hpss(tone + clicks, sr)
        self.assertGreater(rms_db(harmonic), rms_db(percussive) + 6.0,
                           "тон должен остаться в гармонической части")

    def test_mask_survives_track_shorter_than_window(self):
        from engine.separation import _spectral_center_mask

        short = np.ones(1000, dtype=np.float32)
        mask = _spectral_center_mask(short, short, 44100)
        self.assertEqual(mask.shape[0], 1, "короткий трек — ровно один кадр")


class DtypeTests(unittest.TestCase):
    """Аудио живёт во float32.

    Одного float64-скаляра хватает, чтобы numpy повысил весь массив: на
    трёхминутном стерео это лишние 135 МБ на дорожку, и рендер падает с
    нехваткой памяти уже у пользователя, а не в тестах.
    """

    def test_pan_keeps_float32(self):
        mono = np.ones(4096, dtype=np.float32)
        stereo = np.ones((2, 4096), dtype=np.float32)
        for source in (mono, stereo):
            for position in (-1.0, -0.4, 0.0, 0.4, 1.0):
                out = pan(source, position)
                self.assertEqual(out.dtype, np.float32,
                                 f"панорама {position} повысила тип")
                self.assertEqual(out.shape, (2, 4096))

    def test_pan_is_equal_power(self):
        stereo = np.ones((2, 128), dtype=np.float32)
        for position in (-1.0, -0.5, 0.0, 0.5, 1.0):
            out = pan(stereo, position)
            power = float(out[0, 0] ** 2 + out[1, 0] ** 2)
            self.assertAlmostEqual(power, 2.0, places=4,
                                   msg=f"громкость гуляет при панораме {position}")

    def test_effect_chain_stays_float32(self):
        from engine.dsp import delay_fx, reverb, stereo_width

        signal = np.repeat(np.sin(np.linspace(0, 200, 44100, dtype=np.float32))[None, :], 2, axis=0)
        for name, fx in (("pan", lambda a: pan(a, -0.4)),
                         ("stereo_width", lambda a: stereo_width(a, 1.15)),
                         ("delay", lambda a: delay_fx(a, 44100, mix_amt=0.18)),
                         ("reverb", lambda a: reverb(a, 44100, mix_amt=0.12))):
            self.assertEqual(fx(signal).dtype, np.float32, name)
