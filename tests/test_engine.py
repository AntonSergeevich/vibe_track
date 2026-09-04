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
from engine.dsp import normalize_active_rms, pitch_shift, rms_db, sidechain_duck
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
    def test_transform_produces_stems_score_and_master(self):
        out_dir = os.path.join(self.tmp, "render")
        result = transform(self.track_path,
                           "ню-метал в духе Korn, drop C, скретчи, женский вокал в припеве",
                           out_dir=out_dir,
                           options=RenderOptions(manual_lyrics="Строка раз\nСтрока два"))
        self.assertTrue(os.path.exists(result.master_path))
        self.assertIn("guitar_rhythm", result.stem_paths)
        self.assertIn("drums", result.stem_paths)
        self.assertIn("source_vocals", result.stem_paths)
        self.assertTrue(result.tabs.get("guitar_rhythm"))
        self.assertIn("Строка раз", result.lyric_sheet)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "score", "lyrics_chords.txt")))
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
