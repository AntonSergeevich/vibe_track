"""Тесты HTTP-слоя: загрузка, рендер, партитура, вокальные дубли."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from unittest import mock

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
class EnvFileTests(TestCase):
    """Чтение .env: ключ задаётся один раз файлом, а не в каждом терминале."""

    def _write(self, text: str):
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / ".env"
        path.write_text(text, encoding="utf-8-sig")
        return path

    def test_reads_values_and_ignores_junk(self):
        from vibetrack_site.settings import load_env_file

        path = self._write(
            "# комментарий\n"
            "VIBETRACK_MUSIC_API_KEY=\"sk-test\"\n"
            "VIBETRACK_STABILITY_STRENGTH=0.75  # ближе к оригиналу\n"
            "СЛОМАННАЯ СТРОКА\n")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VIBETRACK_MUSIC_API_KEY", None)
            os.environ.pop("VIBETRACK_STABILITY_STRENGTH", None)
            load_env_file(path)
            self.assertEqual(os.environ["VIBETRACK_MUSIC_API_KEY"], "sk-test")
            self.assertEqual(os.environ["VIBETRACK_STABILITY_STRENGTH"], "0.75",
                             "комментарий в конце строки не должен попадать в значение")

    def test_environment_wins_over_file(self):
        """На сервере настройки приходят из окружения, файл их не перебивает."""
        from vibetrack_site.settings import load_env_file

        path = self._write("VIBETRACK_MUSIC_API_KEY=sk-from-file\n")
        with mock.patch.dict(os.environ, {"VIBETRACK_MUSIC_API_KEY": "sk-real"}):
            load_env_file(path)
            self.assertEqual(os.environ["VIBETRACK_MUSIC_API_KEY"], "sk-real")

    def test_missing_file_is_not_an_error(self):
        from pathlib import Path

        from vibetrack_site.settings import load_env_file

        load_env_file(Path("/nonexistent/.env"))


class QueueFallbackTests(TestCase):
    """Недоступный Redis не должен ронять загруженный трек."""

    def test_task_runs_in_thread_when_broker_is_down(self):
        import threading

        from .tasks import enqueue

        done = threading.Event()
        seen = []

        class _DeadQueue:
            name = "core.tasks.render_track"

            def delay(self, *args):
                raise ConnectionError("Error 11001 connecting to redis:6379")

            def __call__(self, *args):
                seen.append(args)
                done.set()

        with override_settings(VIBETRACK_INLINE_WORKER=False):
            self.assertIsNone(enqueue(_DeadQueue(), 42))
        self.assertTrue(done.wait(5), "задача должна была уйти в фоновый поток")
        self.assertEqual(seen, [(42,)])

    def test_inline_worker_does_not_touch_the_queue(self):
        import threading

        from .tasks import enqueue

        done = threading.Event()

        class _Queue:
            def delay(self, *args):
                raise AssertionError("при включённом потоке очередь не нужна")

            def __call__(self, *args):
                done.set()

        with override_settings(VIBETRACK_INLINE_WORKER=True):
            enqueue(_Queue(), 1)
        self.assertTrue(done.wait(5))


class MemoryErrorTests(TestCase):
    """Нехватка памяти должна объясняться по-человечески и возвращать списание."""

    def test_render_explains_memory_error_and_refunds(self):
        from .models import Project, RenderJob, Track, User
        from . import billing
        from .tasks import render_track

        user = User.objects.create_user(username="drummer", password="pass12345")
        project = Project.objects.create(owner=user, title="Мои треки")
        track = Track.objects.create(project=project, title="Длинный трек")
        AudioFile.objects.create(track=track, file="audio/x.wav",
                                 kind=AudioFile.KIND_SOURCE, status="done")
        job = RenderJob.objects.create(track=track, options={"generate_cover": True})
        billing.charge(user, job)
        billing.charge_cover(user, job)

        with mock.patch("core.tasks.transform",
                        side_effect=MemoryError("Unable to allocate 135. MiB")):
            render_track(job.pk)

        job.refresh_from_db()
        self.assertEqual(job.status, RenderJob.STATUS_ERROR)
        self.assertIn("памяти", job.error)
        self.assertIn("покороче", job.error, "человеку нужен выход, а не диагноз")
        self.assertEqual(billing.used_this_period(user), 0, "списание вернули")
        self.assertEqual(billing.covers_used_this_period(user), 0)


class StalledJobTests(TestCase):
    """Оборванный рендер должен закрываться, а не висеть на 25% вечно."""

    def setUp(self):
        from .models import Project, Track, User

        self.user = User.objects.create_user(username="bassist", password="pass12345")
        self.project = Project.objects.create(owner=self.user, title="Мои треки")
        self.track = Track.objects.create(project=self.project, title="Трек")

    def _running_job(self, minutes_ago: int):
        from datetime import timedelta

        from django.utils import timezone

        from . import billing

        job = RenderJob.objects.create(track=self.track,
                                       status=RenderJob.STATUS_RUNNING,
                                       stage="separate", progress=25)
        billing.charge(self.user, job)
        RenderJob.objects.filter(pk=job.pk).update(
            updated_at=timezone.now() - timedelta(minutes=minutes_ago))
        return job

    def test_long_silence_closes_the_job_and_returns_the_charge(self):
        from . import billing

        job = self._running_job(minutes_ago=45)
        self.client.force_login(self.user)
        data = self.client.get(f"/api/renders/{job.pk}/status/").json()

        self.assertEqual(data["status"], "error")
        self.assertIn("памяти", data["error"])
        self.assertIn("заново", data["error"], "человеку нужен следующий шаг")
        self.assertEqual(billing.used_this_period(self.user), 0)

    def test_slow_render_is_not_touched(self):
        """Demucs молчит минутами — это не повод объявлять его мёртвым."""
        job = self._running_job(minutes_ago=4)
        self.client.force_login(self.user)
        data = self.client.get(f"/api/renders/{job.pk}/status/").json()
        self.assertEqual(data["status"], RenderJob.STATUS_RUNNING)

    def test_startup_closes_jobs_left_by_a_dead_process(self):
        from django.apps import apps
        from django.test import override_settings

        from . import billing

        job = self._running_job(minutes_ago=1)
        with override_settings(VIBETRACK_INLINE_WORKER=True):
            with mock.patch.object(sys, "argv", ["manage.py", "runserver"]):
                apps.get_app_config("core")._recover_orphaned_jobs()

        job.refresh_from_db()
        self.assertEqual(job.status, RenderJob.STATUS_ERROR)
        self.assertIn("перезапущен", job.error)
        self.assertEqual(billing.used_this_period(self.user), 0)


class CleanupCommandTests(TestCase):
    """Уборка диска: показывает по умолчанию, удаляет только по просьбе."""

    def setUp(self):
        from datetime import timedelta

        from django.utils import timezone

        from .models import Project, Track, User

        self.dir = tempfile.mkdtemp(prefix="vibetrack-clean-")
        user = User.objects.create_user(username="tidy", password="pass12345")
        project = Project.objects.create(owner=user, title="Мои треки")
        track = Track.objects.create(project=project, title="Трек")

        self.fresh = RenderJob.objects.create(track=track)
        self.old = RenderJob.objects.create(track=track)
        RenderJob.objects.filter(pk=self.old.pk).update(
            created_at=timezone.now() - timedelta(days=90))

        for job in (self.fresh, self.old):
            folder = os.path.join(self.dir, "renders", f"job_{job.pk}")
            os.makedirs(folder)
            with open(os.path.join(folder, "master.mp3"), "wb") as fh:
                fh.write(b"0" * 1024)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(self, **kwargs):
        from io import StringIO

        from django.core.management import call_command

        out = StringIO()
        with override_settings(MEDIA_ROOT=self.dir):
            call_command("cleanup", stdout=out, **kwargs)
        return out.getvalue()

    def test_dry_run_touches_nothing(self):
        output = self._run(days=30)
        self.assertIn("только показ", output)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "renders", f"job_{self.old.pk}")),
                        "без --yes ничего удалять нельзя")

    def test_removes_old_renders_and_keeps_fresh_ones(self):
        self._run(days=30, yes=True)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "renders", f"job_{self.old.pk}")))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "renders", f"job_{self.fresh.pk}")),
                        "свежий рендер трогать нельзя")
        self.assertFalse(RenderJob.objects.filter(pk=self.old.pk).exists())
        self.assertTrue(RenderJob.objects.filter(pk=self.fresh.pk).exists())

    def test_report_names_the_model_cache(self):
        """Веса моделей — главный пожиратель системного диска, их видно всегда."""
        output = self._run(days=30)
        self.assertIn("Веса моделей", output)
        self.assertIn("Кэш Hugging Face", output)


class CoverLimitTests(TestCase):
    """Лимит генераций у внешней модели — он же защита от работы в минус."""

    def setUp(self):
        from .models import Project, Track, User

        self.user = User.objects.create_user(username="singer", password="pass12345")
        self.project = Project.objects.create(owner=self.user, title="Мои треки")
        self.track = Track.objects.create(project=self.project, title="Трек")
        AudioFile.objects.create(track=self.track, file="audio/x.wav",
                                 kind=AudioFile.KIND_SOURCE, status="done")

    def _job(self):
        return RenderJob.objects.create(track=self.track, prompt="ню-метал")

    def test_free_plan_cannot_order_cover(self):
        self.client.force_login(self.user)
        response = self.client.post(
            f"/api/tracks/{self.track.pk}/transform/",
            {"options": {"generate_cover": True}}, content_type="application/json")
        self.assertEqual(response.status_code, 402)
        self.assertEqual(response.json()["code"], "cover_quota_exceeded")
        self.assertIn("190", response.json()["detail"],
                      "отказ должен подсказывать, где кавер взять")

    def test_covers_run_out_separately_from_tracks(self):
        from datetime import timedelta

        from django.utils import timezone

        from . import billing
        from .models import Subscription

        Subscription.objects.create(user=self.user, plan="start",
                                    expires_at=timezone.now() + timedelta(days=30))
        for _ in range(billing.PLANS["start"].covers):
            billing.charge_cover(self.user, self._job())
        with self.assertRaises(billing.CoverQuotaExceeded):
            billing.check_cover_quota(self.user)
        # рендеры при этом целы: кавер не съедает трек из тарифа
        self.assertEqual(billing.remaining(self.user), 40)

    def test_failed_cover_returns_the_limit(self):
        from . import billing

        job = self._job()
        billing.charge_cover(self.user, job)
        self.assertEqual(billing.covers_used_this_period(self.user), 1)
        billing.refund_cover(self.user, job, "кавер не сгенерировался")
        self.assertEqual(billing.covers_used_this_period(self.user), 0)

    def test_anonymous_request_never_orders_a_cover(self):
        """Списать кавер не с кого, поэтому опция молча выключается."""
        from .models import User

        User.objects.filter(pk=self.user.pk)  # пользователь не участвует
        with mock.patch("core.views.enqueue", return_value=None) as enqueue:
            response = self.client.post(
                f"/api/tracks/{self.track.pk}/transform/",
                {"options": {"generate_cover": True}}, content_type="application/json")
        self.assertEqual(response.status_code, 202)
        self.assertTrue(enqueue.called)
        job = RenderJob.objects.get(track=self.track)
        self.assertFalse(job.options.get("generate_cover"))

    def test_every_sold_plan_stays_profitable_on_covers(self):
        """Каверы тарифа не должны стоить дороже самого тарифа."""
        from . import billing

        cover_cost_rub = 20            # $0.20 по курсу с запасом
        for plan in billing.PLANS.values():
            if plan.slug in ("free", "unlimited"):
                continue
            self.assertLessEqual(plan.covers * cover_cost_rub, plan.price_rub,
                                 f"тариф {plan.slug} уходит в минус на каверах")
            self.assertLessEqual(plan.covers, plan.tracks,
                                 f"у тарифа {plan.slug} каверов больше, чем рендеров")


class CabinetTests(TestCase):
    """Личный кабинет и вход."""

    def setUp(self):
        from .models import User

        self.user = User.objects.create_user(username="singer", password="pass12345")

    def test_cabinet_requires_login(self):
        response = self.client.get("/cabinet/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response["Location"])

    def test_purchase_list_hides_developer_plan(self):
        self.client.force_login(self.user)
        response = self.client.get("/cabinet/")
        self.assertNotContains(response, 'data-plan="unlimited"',
                               msg_prefix="безлимит разработчика не продаётся")
        self.assertContains(response, 'data-plan="studio"')

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

    def test_render_detail_page_shows_everything(self):
        from .models import Project, RenderJob, Score, Stem, Track

        project = Project.objects.create(owner=self.user, title="Мои треки")
        track = Track.objects.create(project=project, title="Разбор трека")
        job = RenderJob.objects.create(track=track, prompt="ню-метал",
                                       status=RenderJob.STATUS_DONE,
                                       result={"analysis": {"tempo": 132.5, "key_name": "D major"}},
                                       spec={"tuning": "drop_c", "genre": "nu_metal"})
        Stem.objects.create(job=job, name="guitar_rhythm", label="Ритм-гитара",
                            file="stems/g.wav", rms_db=-12.0)
        Score.objects.create(job=job, chord_chart="G | D | A | Em",
                             tabs={"guitar_rhythm": "e|--0--|"}, key="D major")

        self.client.force_login(self.user)
        response = self.client.get(f"/cabinet/track/{job.pk}/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Разбор трека")
        self.assertContains(response, "Ритм-гитара")
        self.assertContains(response, "G | D | A | Em")
        self.assertContains(response, "Таб: guitar_rhythm")
        self.assertContains(response, "D major")

    def test_cover_gets_its_own_block(self):
        """Кавер — главный результат, его нельзя прятать среди дорожек."""
        from .models import Project, RenderJob, Stem, Track

        project = Project.objects.create(owner=self.user, title="Мои треки")
        track = Track.objects.create(project=project, title="Трек с кавером")
        job = RenderJob.objects.create(track=track, status=RenderJob.STATUS_DONE)
        Stem.objects.create(job=job, name="cover", label="Кавер от нейросети",
                            file="renders/cover.mp3", source="generated")
        Stem.objects.create(job=job, name="bass", label="Бас", file="stems/b.wav")

        self.client.force_login(self.user)
        response = self.client.get(f"/cabinet/track/{job.pk}/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Кавер от нейросети")
        self.assertEqual([s.name for s in response.context["stems"]], ["bass"],
                         "кавер не должен дублироваться в списке дорожек")
        self.assertContains(response, "renders/cover.mp3", msg_prefix="плеер кавера")

    def test_render_detail_is_private(self):
        from .models import Project, RenderJob, Track, User

        stranger = User.objects.create_user(username="stranger", password="pass12345")
        project = Project.objects.create(owner=stranger, title="Чужие треки")
        track = Track.objects.create(project=project, title="Чужой трек")
        job = RenderJob.objects.create(track=track)

        self.client.force_login(self.user)
        self.assertEqual(self.client.get(f"/cabinet/track/{job.pk}/").status_code, 404)

    def test_cabinet_links_to_detail_page(self):
        from .models import Project, RenderJob, Track

        project = Project.objects.create(owner=self.user, title="Мои треки")
        track = Track.objects.create(project=project, title="Мой трек")
        job = RenderJob.objects.create(track=track, status=RenderJob.STATUS_DONE)

        self.client.force_login(self.user)
        response = self.client.get("/cabinet/")
        self.assertContains(response, f"/cabinet/track/{job.pk}/")

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


@override_settings(MEDIA_ROOT=MEDIA)
class UnlimitedAccessTests(TestCase):
    """Безлимит владельца сервиса."""

    def setUp(self):
        from .models import User

        self.user = User.objects.create_user(username="dev", password="pass12345")

    def test_staff_gets_unlimited_plan(self):
        from . import billing

        self.assertEqual(billing.plan_for(self.user).slug, "free")
        self.user.is_staff = True
        self.user.save()
        plan = billing.plan_for(self.user)
        self.assertEqual(plan.slug, "unlimited")
        self.assertTrue(plan.wav_stems)
        self.assertEqual(billing.check_quota(self.user).slug, "unlimited")

    def test_unlimited_ignores_spent_tracks(self):
        from . import billing
        from .models import Project, RenderJob, Track

        self.user.is_staff = True
        self.user.save()
        project = Project.objects.create(owner=self.user, title="Мои треки")
        track = Track.objects.create(project=project, title="Трек")
        for _ in range(5):
            billing.charge(self.user, RenderJob.objects.create(track=track))
        billing.check_quota(self.user)     # не должно бросить

    def test_env_list_grants_unlimited(self):
        from . import billing

        with mock.patch.dict(os.environ, {"VIBETRACK_UNLIMITED_USERS": "dev, someone"}):
            self.assertTrue(billing.is_unlimited(self.user))
        with mock.patch.dict(os.environ, {"VIBETRACK_UNLIMITED_USERS": "someone"}):
            self.assertFalse(billing.is_unlimited(self.user))

    def test_grant_plan_command(self):
        from django.core.management import call_command

        from . import billing

        call_command("grant_plan", "dev", "studio", "--days", "10")
        self.assertEqual(billing.plan_for(self.user).slug, "studio")

    def test_grant_plan_without_username_lists_users(self):
        from io import StringIO

        from django.core.management import call_command

        out = StringIO()
        call_command("grant_plan", stdout=out)
        printed = out.getvalue()
        self.assertIn("dev", printed)
        self.assertIn("Проба", printed, "рядом с логином видно текущий тариф")

    def test_grant_plan_unknown_user_shows_list(self):
        from io import StringIO

        from django.core.management import call_command
        from django.core.management.base import CommandError

        out = StringIO()
        with self.assertRaises(CommandError):
            call_command("grant_plan", "нет_такого", stdout=out)
        self.assertIn("dev", out.getvalue(), "в ошибке должен быть список существующих логинов")

    def test_grant_plan_unlimited_flag(self):
        from django.core.management import call_command

        from . import billing

        call_command("grant_plan", "dev", "--unlimited")
        self.user.refresh_from_db()
        self.assertTrue(self.user.is_staff)
        self.assertTrue(billing.is_unlimited(self.user))
