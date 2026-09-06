# core/views.py
"""HTTP-слой: загрузка треков, запуск рендера, партитура, вокальные дубли."""
from __future__ import annotations

import os

from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from engine.arrangement import (BASS_TUNINGS, GENRES, GROOVE_KEYWORDS, INSTRUMENT_CATALOG,
                                REFERENCE_ARTISTS, TUNINGS, VOCAL_STYLE_KEYWORDS,
                                parse_prompt)
from engine.models import status as model_status
from engine.tabs import TUNING_LABELS

from . import billing as billing_service
from .forms import RegisterForm
from .models import (AudioFile, Billing, EffectChain, Payment, Project, RenderJob,
                     Subscription, Track, UsageRecord, User, VocalTake)
from .serializers import (AudioFileSerializer, BillingSerializer, EffectChainSerializer,
                          ProjectSerializer, RenderJobCreateSerializer, RenderJobSerializer,
                          ScoreSerializer, TrackSerializer, TrackUploadSerializer,
                          UserSerializer, VocalTakeSerializer)
from .tasks import enqueue, process_vocal_take, render_track

ALLOWED_AUDIO_EXT = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.webm', '.opus', '.aiff'}


def studio(request):
    """Одностраничная студия."""
    return render(request, 'studio/index.html', {
        'genres': GENRES,
        'instruments': INSTRUMENT_CATALOG,
        'tunings': {k: TUNING_LABELS.get(k, k) for k in TUNINGS},
        'bass_tunings': {k: TUNING_LABELS.get(k, k) for k in BASS_TUNINGS},
        'grooves': list(GROOVE_KEYWORDS),
        'vocal_styles': list(VOCAL_STYLE_KEYWORDS),
        'references': sorted({a for a in REFERENCE_ARTISTS if a.isascii()}),
        'max_upload_mb': settings.VIBETRACK_MAX_UPLOAD_MB,
    })


@login_required
def cabinet(request):
    """Личный кабинет: тариф, остаток, история треков и платежей."""
    jobs = (RenderJob.objects
            .filter(track__project__owner=request.user)
            .select_related('track', 'score')
            .prefetch_related('stems')[:50])
    return render(request, 'studio/cabinet.html', {
        'billing': billing_service.summary(request.user),
        'plans': billing_service.PLANS.values(),
        'jobs': jobs,
        'payments': Payment.objects.filter(user=request.user)[:20],
        'subscription': Subscription.objects.filter(
            user=request.user, expires_at__gt=timezone.now()).first(),
        'usage': UsageRecord.objects.filter(user=request.user)[:20],
    })


def register(request):
    """Регистрация. После создания аккаунта сразу пускаем внутрь."""
    if request.user.is_authenticated:
        return redirect('cabinet')
    form = RegisterForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        return redirect('cabinet')
    return render(request, 'studio/register.html', {'form': form})


class IsOwnerOrOpen(permissions.BasePermission):
    """Владелец объекта; в открытом режиме (VIBETRACK_OPEN_API) — все."""

    def has_object_permission(self, request, view, obj):
        if not request.user or not request.user.is_authenticated:
            return settings.REST_FRAMEWORK['DEFAULT_PERMISSION_CLASSES'][0].endswith('AllowAny')
        owner = getattr(obj, 'owner', None) or getattr(obj, 'user', None)
        if owner is not None:
            return owner == request.user
        project = getattr(obj, 'project', None)
        if project is not None:
            return project.owner == request.user
        track = getattr(obj, 'track', None)
        if track is not None:
            return track.project.owner == request.user
        job = getattr(obj, 'job', None)
        if job is not None:
            return job.track.project.owner == request.user
        return False


def _default_project(user) -> Project:
    owner = user if getattr(user, 'is_authenticated', False) else _service_user()
    project, _ = Project.objects.get_or_create(owner=owner, title='Мои треки',
                                               defaults={'status': 'active'})
    return project


def _service_user() -> User:
    """Пользователь для анонимной работы в открытом режиме."""
    user, _ = User.objects.get_or_create(username='vibetrack', defaults={'display_name': 'VibeTrack'})
    return user


class ProjectViewSet(viewsets.ModelViewSet):
    serializer_class = ProjectSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        qs = Project.objects.all()
        if self.request.user.is_authenticated:
            qs = qs.filter(owner=self.request.user)
        return qs

    def perform_create(self, serializer):
        owner = self.request.user if self.request.user.is_authenticated else _service_user()
        serializer.save(owner=owner)


class TrackViewSet(viewsets.ModelViewSet):
    serializer_class = TrackSerializer
    permission_classes = [permissions.AllowAny]
    parser_classes = [JSONParser, MultiPartParser, FormParser]

    def get_queryset(self):
        qs = Track.objects.select_related('project').prefetch_related('audio_files', 'render_jobs')
        if self.request.user.is_authenticated:
            qs = qs.filter(project__owner=self.request.user)
        project_id = self.request.query_params.get('project')
        return qs.filter(project_id=project_id) if project_id else qs

    @action(detail=False, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def upload(self, request):
        """Загрузка исходного трека: создаёт Track и AudioFile(kind=source)."""
        serializer = TrackUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        upload = serializer.validated_data['file']

        ext = os.path.splitext(upload.name)[1].lower()
        if ext not in ALLOWED_AUDIO_EXT:
            return Response(
                {'detail': f'Формат {ext or "?"} не поддерживается. Разрешены: '
                           f'{", ".join(sorted(ALLOWED_AUDIO_EXT))}'},
                status=status.HTTP_400_BAD_REQUEST)
        limit = settings.VIBETRACK_MAX_UPLOAD_MB * 1024 * 1024
        if upload.size > limit:
            return Response({'detail': f'Файл больше {settings.VIBETRACK_MAX_UPLOAD_MB} МБ.'},
                            status=status.HTTP_400_BAD_REQUEST)

        project = serializer.validated_data.get('project') or _default_project(request.user)
        title = serializer.validated_data.get('title') or os.path.splitext(upload.name)[0]
        track = Track.objects.create(project=project, title=title[:200],
                                     order=project.tracks.count())
        AudioFile.objects.create(track=track, file=upload, kind=AudioFile.KIND_SOURCE,
                                 status='done')
        return Response(TrackSerializer(track, context={'request': request}).data,
                        status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['post'])
    def transform(self, request, pk=None):
        """Запускает превращение трека в ню-метал."""
        track = self.get_object()
        if track.source_file is None:
            return Response({'detail': 'У трека нет исходного файла.'},
                            status=status.HTTP_400_BAD_REQUEST)

        serializer = RenderJobCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        if request.user.is_authenticated:
            try:
                plan = billing_service.check_quota(request.user)
            except billing_service.QuotaExceeded as exc:
                return Response(
                    {'detail': str(exc), 'code': 'quota_exceeded',
                     'billing': billing_service.summary(request.user)},
                    status=status.HTTP_402_PAYMENT_REQUIRED)
            # тариф решает, отдавать WAV или mp3 и звать ли Demucs
            data['options'] = {**billing_service.render_options_for(plan), **data['options']}
        elif getattr(settings, 'VIBETRACK_REQUIRE_LOGIN', False):
            return Response({'detail': 'Войдите, чтобы обрабатывать треки.'},
                            status=status.HTTP_401_UNAUTHORIZED)

        job = RenderJob.objects.create(track=track, prompt=data['prompt'],
                                       overrides=data['overrides'], options=data['options'])
        billing_service.charge(request.user, job, note=data['prompt'][:200])
        async_result = enqueue(render_track, job.pk)
        RenderJob.objects.filter(pk=job.pk).update(
            celery_task_id=getattr(async_result, 'id', '') or '')
        job.refresh_from_db()
        return Response(RenderJobSerializer(job, context={'request': request}).data,
                        status=status.HTTP_202_ACCEPTED)


class RenderJobViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = RenderJobSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        qs = RenderJob.objects.select_related('track', 'score').prefetch_related('stems')
        if self.request.user.is_authenticated:
            qs = qs.filter(track__project__owner=self.request.user)
        track_id = self.request.query_params.get('track')
        return qs.filter(track_id=track_id) if track_id else qs

    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        job = self.get_object()
        return Response({'id': job.pk, 'status': job.status, 'stage': job.stage,
                         'progress': job.progress, 'error': job.error,
                         'warnings': job.warnings})

    @action(detail=True, methods=['get'])
    def score(self, request, pk=None):
        job = self.get_object()
        score = getattr(job, 'score', None)
        if score is None:
            return Response({'detail': 'Партитура ещё не готова.'},
                            status=status.HTTP_404_NOT_FOUND)
        return Response(ScoreSerializer(score).data)

    @action(detail=True, methods=['get'], url_path=r'download/(?P<what>[\w.-]+)')
    def download(self, request, pk=None, what=None):
        """Скачивание мастера, дорожки или текстовой партитуры."""
        job = self.get_object()
        if what == 'master':
            if not job.master:
                raise Http404('Мастер ещё не готов')
            return FileResponse(job.master.open('rb'), as_attachment=True,
                                filename=f'vibetrack_{job.pk}_master{os.path.splitext(job.master.name)[1]}')
        stem = job.stems.filter(name=what).first()
        if stem:
            return FileResponse(stem.file.open('rb'), as_attachment=True,
                                filename=f'{what}{os.path.splitext(stem.file.name)[1]}')
        score = getattr(job, 'score', None)
        if score:
            texts = {'lyrics': score.lyric_sheet, 'chords': score.chord_chart}
            texts.update({f'tab_{k}': v for k, v in (score.tabs or {}).items()})
            if what in texts and texts[what]:
                response = JsonResponse({'name': what, 'text': texts[what]})
                return response
        raise Http404('Файл не найден')


class VocalTakeViewSet(viewsets.ModelViewSet):
    """Запись голоса поверх минусовки и её автоматическая обработка."""

    serializer_class = VocalTakeSerializer
    permission_classes = [permissions.AllowAny]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_queryset(self):
        qs = VocalTake.objects.select_related('track', 'job')
        if self.request.user.is_authenticated:
            qs = qs.filter(track__project__owner=self.request.user)
        track_id = self.request.query_params.get('track')
        return qs.filter(track_id=track_id) if track_id else qs

    def perform_create(self, serializer):
        take = serializer.save(status=RenderJob.STATUS_QUEUED)
        enqueue(process_vocal_take, take.pk)

    @action(detail=True, methods=['post'])
    def reprocess(self, request, pk=None):
        """Перепрогнать дубль с другими настройками (стиль, автотюн, пол)."""
        take = self.get_object()
        for field in ('style', 'gender', 'target_gender', 'autotune', 'gain_db'):
            if field in request.data:
                setattr(take, field, request.data[field])
        take.status = RenderJob.STATUS_QUEUED
        take.save()
        enqueue(process_vocal_take, take.pk)
        return Response(VocalTakeSerializer(take, context={'request': request}).data,
                        status=status.HTTP_202_ACCEPTED)


class AudioFileViewSet(viewsets.ModelViewSet):
    queryset = AudioFile.objects.all()
    serializer_class = AudioFileSerializer
    permission_classes = [permissions.AllowAny]
    parser_classes = [MultiPartParser, FormParser, JSONParser]


class EffectChainViewSet(viewsets.ModelViewSet):
    serializer_class = EffectChainSerializer
    permission_classes = [permissions.AllowAny]

    def get_queryset(self):
        qs = EffectChain.objects.all()
        if self.request.user.is_authenticated:
            qs = qs.filter(track__project__owner=self.request.user)
        return qs


class BillingViewSet(viewsets.ModelViewSet):
    serializer_class = BillingSerializer
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrOpen]

    def get_queryset(self):
        return Billing.objects.filter(user=self.request.user)

    @action(detail=False, methods=['post'])
    def add_credits(self, request):
        amount = int(request.data.get('amount', 0))
        if amount <= 0:
            return Response({'detail': 'Неверная сумма.'}, status=status.HTTP_400_BAD_REQUEST)
        billing, _ = Billing.objects.get_or_create(user=request.user)
        billing.credits += amount
        billing.save()
        return Response(BillingSerializer(billing).data)


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return User.objects.filter(id=self.request.user.id)


@api_view(['GET'])
@permission_classes([permissions.AllowAny])
def capabilities(request):
    """Справочник для UI: инструменты, строи, стили и доступные бэкенды."""
    return Response({
        'genres': {k: {'id': k, **v} for k, v in GENRES.items()},
        'instruments': {k: {**v, 'id': k} for k, v in INSTRUMENT_CATALOG.items()},
        'tunings': {k: TUNING_LABELS.get(k, k) for k in TUNINGS},
        'bass_tunings': {k: TUNING_LABELS.get(k, k) for k in BASS_TUNINGS},
        'grooves': list(GROOVE_KEYWORDS),
        'vocal_styles': list(VOCAL_STYLE_KEYWORDS),
        'references': sorted(REFERENCE_ARTISTS),
        'backends': {name: {'model': info.name, 'available': info.available,
                            'loaded': info.loaded, 'detail': info.detail,
                            'load_seconds': info.load_seconds}
                     for name, info in model_status().items()},
        'limits': {
            'max_upload_mb': settings.VIBETRACK_MAX_UPLOAD_MB,
            'max_duration_sec': settings.VIBETRACK['MAX_DURATION'],
        },
    })


@api_view(['GET'])
@permission_classes([permissions.AllowAny])
def billing_summary(request):
    """Тариф, остаток треков и витрина тарифов."""
    return Response(billing_service.summary(request.user))


@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def checkout(request):
    """Создаёт платёж за тариф.

    Реальный провайдер (ЮKassa и т.п.) подключается здесь: он должен вернуть
    confirmation_url и подтвердить оплату своим вебхуком. Пока платёж
    создаётся в статусе «ожидает» и подтверждается вручную в админке —
    так деньги нельзя выдать себе, просто дёрнув эндпоинт.
    """
    plan_slug = request.data.get('plan', '')
    plan = billing_service.PLANS.get(plan_slug)
    if plan is None or plan.slug == 'free':
        return Response({'detail': 'Неизвестный тариф.'}, status=status.HTTP_400_BAD_REQUEST)

    payment = Payment.objects.create(user=request.user, plan=plan.slug,
                                     amount_rub=plan.price_rub, provider='manual')
    if getattr(settings, 'VIBETRACK_PAYMENTS_TEST_MODE', False):
        activate_payment(payment)
        return Response({'status': 'paid', 'payment': payment.pk,
                         'billing': billing_service.summary(request.user)})
    return Response({'status': 'pending', 'payment': payment.pk,
                     'detail': f'Счёт на {plan.price_rub} ₽ создан. '
                               'Оплата подтверждается после подключения платёжного провайдера.'},
                    status=status.HTTP_202_ACCEPTED)


def activate_payment(payment: Payment) -> Subscription:
    """Отмечает платёж оплаченным и продлевает подписку."""
    from datetime import timedelta

    plan = billing_service.PLANS[payment.plan]
    payment.status = Payment.STATUS_PAID
    payment.paid_at = timezone.now()
    payment.save(update_fields=['status', 'paid_at'])

    current = Subscription.objects.filter(
        user=payment.user, expires_at__gt=timezone.now()).order_by('-expires_at').first()
    # продление добавляется к остатку, а не обнуляет его
    start = current.expires_at if current else timezone.now()
    days = 30 if plan.recurring else 30
    return Subscription.objects.create(
        user=payment.user, plan=plan.slug, started_at=start,
        expires_at=start + timedelta(days=days), payment=payment)


@api_view(['POST'])
@permission_classes([permissions.AllowAny])
def parse_description(request):
    """Разбор текста описания в спецификацию — для предпросмотра в UI."""
    spec = parse_prompt(request.data.get('prompt', ''), None, request.data.get('overrides') or {})
    return Response(spec.to_dict())
