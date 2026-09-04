# core/views.py
"""HTTP-слой: загрузка треков, запуск рендера, партитура, вокальные дубли."""
from __future__ import annotations

import os

from django.conf import settings
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from engine.arrangement import (BASS_TUNINGS, GROOVE_KEYWORDS, INSTRUMENT_CATALOG,
                                REFERENCE_ARTISTS, TUNINGS, VOCAL_STYLE_KEYWORDS,
                                parse_prompt)
from engine.separation import demucs_available
from engine.tabs import TUNING_LABELS
from engine.transcription import whisper_available

from .models import (AudioFile, Billing, EffectChain, Project, RenderJob, Track, User,
                     VocalTake)
from .serializers import (AudioFileSerializer, BillingSerializer, EffectChainSerializer,
                          ProjectSerializer, RenderJobCreateSerializer, RenderJobSerializer,
                          ScoreSerializer, TrackSerializer, TrackUploadSerializer,
                          UserSerializer, VocalTakeSerializer)
from .tasks import process_vocal_take, render_track

ALLOWED_AUDIO_EXT = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.webm', '.opus', '.aiff'}


def studio(request):
    """Одностраничная студия."""
    return render(request, 'studio/index.html', {
        'instruments': INSTRUMENT_CATALOG,
        'tunings': {k: TUNING_LABELS.get(k, k) for k in TUNINGS},
        'bass_tunings': {k: TUNING_LABELS.get(k, k) for k in BASS_TUNINGS},
        'grooves': list(GROOVE_KEYWORDS),
        'vocal_styles': list(VOCAL_STYLE_KEYWORDS),
        'references': sorted({a for a in REFERENCE_ARTISTS if a.isascii()}),
        'max_upload_mb': settings.VIBETRACK_MAX_UPLOAD_MB,
    })


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

        job = RenderJob.objects.create(track=track, prompt=data['prompt'],
                                       overrides=data['overrides'], options=data['options'])
        async_result = render_track.delay(job.pk)
        RenderJob.objects.filter(pk=job.pk).update(celery_task_id=getattr(async_result, 'id', '') or '')
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
        process_vocal_take.delay(take.pk)

    @action(detail=True, methods=['post'])
    def reprocess(self, request, pk=None):
        """Перепрогнать дубль с другими настройками (стиль, автотюн, пол)."""
        take = self.get_object()
        for field in ('style', 'gender', 'target_gender', 'autotune', 'gain_db'):
            if field in request.data:
                setattr(take, field, request.data[field])
        take.status = RenderJob.STATUS_QUEUED
        take.save()
        process_vocal_take.delay(take.pk)
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
        'instruments': {k: {**v, 'id': k} for k, v in INSTRUMENT_CATALOG.items()},
        'tunings': {k: TUNING_LABELS.get(k, k) for k in TUNINGS},
        'bass_tunings': {k: TUNING_LABELS.get(k, k) for k in BASS_TUNINGS},
        'grooves': list(GROOVE_KEYWORDS),
        'vocal_styles': list(VOCAL_STYLE_KEYWORDS),
        'references': sorted(REFERENCE_ARTISTS),
        'backends': {
            'demucs': demucs_available(),
            'whisper': whisper_available(),
        },
        'limits': {
            'max_upload_mb': settings.VIBETRACK_MAX_UPLOAD_MB,
            'max_duration_sec': settings.VIBETRACK['MAX_DURATION'],
        },
    })


@api_view(['POST'])
@permission_classes([permissions.AllowAny])
def parse_description(request):
    """Разбор текста описания в спецификацию — для предпросмотра в UI."""
    spec = parse_prompt(request.data.get('prompt', ''), None, request.data.get('overrides') or {})
    return Response(spec.to_dict())
