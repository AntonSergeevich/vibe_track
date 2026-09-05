"""Тесты HTTP-слоя: загрузка, рендер, партитура, вокальные дубли."""
from __future__ import annotations

import os
import shutil
import tempfile

from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from tests.factories import make_track, make_voice

from .models import AudioFile, RenderJob, Stem, Track, VocalTake

MEDIA = tempfile.mkdtemp(prefix="vibetrack-media-")


@override_settings(
    MEDIA_ROOT=MEDIA, CELERY_TASK_ALWAYS_EAGER=True, CELERY_TASK_EAGER_PROPAGATES=True,
    # в тестах не тянем веса нейросетей из сети — проверяем HTTP-слой, а не модели
    VIBETRACK={**settings.VIBETRACK, 'SEPARATION_BACKEND': 'dsp'})
class StudioApiTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Celery читает конфиг один раз при старте, поэтому override_settings
        # до него не доходит — переключаем приложение в eager напрямую.
        from vibetrack_site.celery import app as celery_app

        cls._celery_conf = (celery_app.conf.task_always_eager,
                            celery_app.conf.task_eager_propagates,
                            celery_app.conf.broker_url)
        celery_app.conf.task_always_eager = True
        celery_app.conf.task_eager_propagates = True
        # даже в eager-режиме Celery берёт продюсера у брокера — в тестах он в памяти
        celery_app.conf.broker_url = "memory://"
        cls.tmp = tempfile.mkdtemp(prefix="vibetrack-src-")
        cls.track_path = make_track(os.path.join(cls.tmp, "src.wav"), seconds=10)
        cls.voice_path = make_voice(os.path.join(cls.tmp, "voice.wav"), seconds=3)

    @classmethod
    def tearDownClass(cls):
        from vibetrack_site.celery import app as celery_app

        (celery_app.conf.task_always_eager,
         celery_app.conf.task_eager_propagates,
         celery_app.conf.broker_url) = cls._celery_conf
        shutil.rmtree(cls.tmp, ignore_errors=True)
        shutil.rmtree(MEDIA, ignore_errors=True)
        super().tearDownClass()

    def _upload(self, name="src.wav"):
        with open(self.track_path, "rb") as fh:
            payload = SimpleUploadedFile(name, fh.read(), content_type="audio/wav")
        return self.client.post("/api/tracks/upload/", {"file": payload, "title": "Тест"})

    def test_studio_page_renders(self):
        response = self.client.get(reverse("studio"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "VIBE")

    def test_capabilities_lists_instruments_and_backends(self):
        data = self.client.get("/api/capabilities/").json()
        self.assertIn("guitar_rhythm", data["instruments"])
        self.assertIn("drop_a_7", data["tunings"])
        self.assertIn("demucs", data["backends"])

    def test_parse_description_endpoint(self):
        response = self.client.post(
            "/api/parse-description/",
            {"prompt": "ню-метал korn, семиструнка drop A, скретчи"},
            content_type="application/json")
        spec = response.json()
        self.assertEqual(spec["tuning"], "drop_a_7")
        self.assertIn("turntables", [i["id"] for i in spec["instruments"] if i["enabled"]])

    def test_upload_creates_track_with_source(self):
        response = self._upload()
        self.assertEqual(response.status_code, 201, response.content)
        track = Track.objects.get(pk=response.json()["id"])
        self.assertEqual(track.audio_files.filter(kind=AudioFile.KIND_SOURCE).count(), 1)

    def test_upload_rejects_wrong_extension(self):
        bad = SimpleUploadedFile("song.txt", b"not audio", content_type="text/plain")
        response = self.client.post("/api/tracks/upload/", {"file": bad})
        self.assertEqual(response.status_code, 400)
        self.assertIn("не поддерживается", response.json()["detail"])

    def test_transform_validates_unknown_instrument(self):
        track_id = self._upload().json()["id"]
        response = self.client.post(
            f"/api/tracks/{track_id}/transform/",
            {"overrides": {"instruments": ["theremin"]}}, content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_full_transform_flow(self):
        track_id = self._upload().json()["id"]
        response = self.client.post(
            f"/api/tracks/{track_id}/transform/",
            {"prompt": "ню-метал в духе Korn, drop C, скретчи, женский вокал",
             "overrides": {"instruments": ["guitar_rhythm", "guitar_rhythm_r", "bass", "drums"]},
             "options": {"manual_lyrics": "Первая строка\nВторая строка"}},
            content_type="application/json")
        self.assertEqual(response.status_code, 202, response.content)

        job = RenderJob.objects.get(pk=response.json()["id"])
        self.assertEqual(job.status, RenderJob.STATUS_DONE, job.error)
        self.assertEqual(job.progress, 100)
        self.assertTrue(job.master.name)
        self.assertTrue(os.path.exists(job.master.path))

        names = set(job.stems.values_list("name", flat=True))
        self.assertIn("guitar_rhythm", names)
        self.assertIn("drums", names)
        self.assertTrue(any(n.startswith("source_") for n in names))
        for stem in job.stems.all():
            self.assertTrue(os.path.exists(stem.file.path), stem.name)

        score = self.client.get(f"/api/renders/{job.pk}/score/").json()
        self.assertIn("Первая строка", score["lyric_sheet"])
        self.assertIn("guitar_rhythm", score["tabs"])
        self.assertTrue(score["chord_chart"])
        self.assertEqual(score["tuning"], "drop_c")

        status = self.client.get(f"/api/renders/{job.pk}/status/").json()
        self.assertEqual(status["status"], "done")

        download = self.client.get(f"/api/renders/{job.pk}/download/master/")
        self.assertEqual(download.status_code, 200)
        self.assertIn("attachment", download["Content-Disposition"])

        detail = self.client.get(f"/api/renders/{job.pk}/").json()
        self.assertTrue(detail["master_url"])
        self.assertTrue(detail["stems"][0]["file_url"])

    def test_transform_without_source_is_rejected(self):
        track = Track.objects.create(project=None_project(), title="Пустой")
        response = self.client.post(f"/api/tracks/{track.pk}/transform/", {},
                                    content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_vocal_take_is_processed_and_mixed(self):
        track_id = self._upload().json()["id"]
        job_id = self.client.post(
            f"/api/tracks/{track_id}/transform/",
            {"prompt": "ню-метал drop C",
             "overrides": {"instruments": ["guitar_rhythm", "drums"]}},
            content_type="application/json").json()["id"]

        with open(self.voice_path, "rb") as fh:
            take_file = SimpleUploadedFile("take.wav", fh.read(), content_type="audio/wav")
        response = self.client.post("/api/vocal-takes/", {
            "raw_file": take_file, "track": track_id, "job": job_id,
            "style": "scream", "gender": "male", "target_gender": "female",
        })
        self.assertEqual(response.status_code, 201, response.content)

        take = VocalTake.objects.get(pk=response.json()["id"])
        self.assertEqual(take.status, RenderJob.STATUS_DONE, take.error)
        self.assertTrue(os.path.exists(take.processed_file.path))
        self.assertTrue(os.path.exists(take.mixed_file.path))

        detail = self.client.get(f"/api/vocal-takes/{take.pk}/").json()
        self.assertTrue(detail["mixed_url"])
        self.assertTrue(detail["processed_url"])

    def test_vocal_take_rejects_unknown_style(self):
        track_id = self._upload().json()["id"]
        with open(self.voice_path, "rb") as fh:
            take_file = SimpleUploadedFile("take.wav", fh.read(), content_type="audio/wav")
        response = self.client.post("/api/vocal-takes/", {
            "raw_file": take_file, "track": track_id, "style": "opera", "gender": "male"})
        self.assertEqual(response.status_code, 400)


@override_settings(MEDIA_ROOT=MEDIA)
class BillingTests(TestCase):
    """Тарифы, лимиты, списания и возвраты."""

    def setUp(self):
        from .models import Project, Track, User

        self.user = User.objects.create_user(username="musician", password="pass12345")
        self.project = Project.objects.create(owner=self.user, title="Мои треки")
        self.track = Track.objects.create(project=self.project, title="Трек")

    def _job(self):
        return RenderJob.objects.create(track=self.track, prompt="ню-метал")

    def test_new_user_is_on_free_plan(self):
        from . import billing

        plan = billing.plan_for(self.user)
        self.assertEqual(plan.slug, "free")
        self.assertEqual(billing.remaining(self.user), 2)

    def test_charge_and_refund_accounting(self):
        from . import billing

        job = self._job()
        billing.charge(self.user, job)
        self.assertEqual(billing.remaining(self.user), 1)

        billing.refund(self.user, job, "рендер упал")
        self.assertEqual(billing.remaining(self.user), 2, "возврат вернул трек в лимит")

        billing.refund(self.user, job, "повторный возврат")
        self.assertEqual(billing.remaining(self.user), 2, "дважды возвращать нельзя")

    def test_quota_exceeded_raises_with_offer(self):
        from . import billing

        for _ in range(2):
            billing.charge(self.user, self._job())
        with self.assertRaises(billing.QuotaExceeded) as ctx:
            billing.check_quota(self.user)
        self.assertIn("190", str(ctx.exception), "в отказе должно быть предложение купить")

    def test_paid_plan_raises_limit_and_unlocks_wav(self):
        from datetime import timedelta

        from django.utils import timezone

        from . import billing
        from .models import Subscription

        Subscription.objects.create(user=self.user, plan="studio",
                                    expires_at=timezone.now() + timedelta(days=30))
        plan = billing.plan_for(self.user)
        self.assertEqual(plan.slug, "studio")
        self.assertEqual(plan.tracks, 90)
        options = billing.render_options_for(plan)
        self.assertEqual(options["export_format"], "wav")
        self.assertEqual(options["separation_backend"], "auto")

    def test_free_plan_gets_mp3_and_dsp(self):
        from . import billing

        options = billing.render_options_for(billing.PLANS["free"])
        self.assertEqual(options["export_format"], "mp3")
        self.assertEqual(options["separation_backend"], "dsp",
                         "бесплатный тариф не должен занимать Demucs")

    def test_expired_subscription_falls_back_to_free(self):
        from datetime import timedelta

        from django.utils import timezone

        from . import billing
        from .models import Subscription

        Subscription.objects.create(user=self.user, plan="pro",
                                    started_at=timezone.now() - timedelta(days=60),
                                    expires_at=timezone.now() - timedelta(days=1))
        self.assertEqual(billing.plan_for(self.user).slug, "free")

    def test_transform_returns_402_when_quota_is_out(self):
        from . import billing

        for _ in range(2):
            billing.charge(self.user, self._job())
        AudioFile.objects.create(track=self.track, file="audio/x.wav",
                                 kind=AudioFile.KIND_SOURCE, status="done")
        self.client.force_login(self.user)
        response = self.client.post(f"/api/tracks/{self.track.pk}/transform/", {},
                                    content_type="application/json")
        self.assertEqual(response.status_code, 402)
        self.assertEqual(response.json()["code"], "quota_exceeded")

    def test_checkout_creates_pending_payment(self):
        from .models import Payment

        self.client.force_login(self.user)
        response = self.client.post("/api/billing/checkout/", {"plan": "studio"},
                                    content_type="application/json")
        self.assertEqual(response.status_code, 202)
        payment = Payment.objects.get(user=self.user)
        self.assertEqual(payment.amount_rub, 690)
        self.assertEqual(payment.status, Payment.STATUS_PENDING)
        self.assertEqual(billing_plan_slug(self.user), "free",
                         "неоплаченный счёт не должен включать тариф")

    @override_settings(VIBETRACK_PAYMENTS_TEST_MODE=True)
    def test_checkout_in_test_mode_activates_subscription(self):
        self.client.force_login(self.user)
        response = self.client.post("/api/billing/checkout/", {"plan": "start"},
                                    content_type="application/json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(billing_plan_slug(self.user), "start")
        self.assertEqual(response.json()["billing"]["remaining"], 40)

    def test_checkout_rejects_unknown_plan(self):
        self.client.force_login(self.user)
        for bad in ("free", "platinum", ""):
            response = self.client.post("/api/billing/checkout/", {"plan": bad},
                                        content_type="application/json")
            self.assertEqual(response.status_code, 400, bad)


@override_settings(MEDIA_ROOT=MEDIA)
class CabinetTests(TestCase):
    """Личный кабинет и вход."""

    def setUp(self):
        from .models import User

        self.user = User.objects.create_user(username="singer", password="pass12345")

    def test_cabinet_requires_login(self):
        response = self.client.get("/cabinet/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])

    def test_cabinet_shows_plan_and_tracks(self):
        from .models import Project, RenderJob, Track

        project = Project.objects.create(owner=self.user, title="Мои треки")
        track = Track.objects.create(project=project, title="Мой трек")
        RenderJob.objects.create(track=track, prompt="ню-метал в духе Korn",
                                 status=RenderJob.STATUS_DONE)

        self.client.force_login(self.user)
        response = self.client.get("/cabinet/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Проба")
        self.assertContains(response, "Мой трек")
        self.assertContains(response, "690")      # витрина тарифов на месте

    def test_registration_creates_user_and_logs_in(self):
        response = self.client.post("/accounts/register/", {
            "username": "newbie",
            "password1": "verystrongpass123",
            "password2": "verystrongpass123",
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response["Location"], "/cabinet/")
        self.assertTrue(self.client.session.get("_auth_user_id"))

    def test_billing_summary_endpoint(self):
        self.client.force_login(self.user)
        data = self.client.get("/api/billing/summary/").json()
        self.assertEqual(data["plan"]["slug"], "free")
        self.assertEqual(data["remaining"], 2)
        self.assertEqual(len(data["plans"]), 5)


def billing_plan_slug(user) -> str:
    from . import billing

    return billing.plan_for(user).slug


def None_project():
    """Проект-заглушка для теста трека без исходника."""
    from .models import Project, User

    user, _ = User.objects.get_or_create(username="tester")
    project, _ = Project.objects.get_or_create(owner=user, title="Пустой проект")
    return project
