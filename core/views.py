# core/views.py
from django.shortcuts import get_object_or_404
from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, FormParser
from django.db import transaction

from .models import Project, Track, AudioFile, EffectChain, Billing, User
from .serializers import (
    ProjectSerializer,
    TrackSerializer,
    AudioFileSerializer,
    EffectChainSerializer,
    BillingSerializer,
    UserSerializer,
)

# Пример: импорт Celery task (реализуй в core/tasks.py)
try:
    from .tasks import process_audio_file
except Exception:
    process_audio_file = None


class IsOwner(permissions.BasePermission):
    """
    Разрешение: доступ только владельцу объекта.
    Поддерживает объекты с полями owner, user или связанные через project.owner.
    """

    def has_object_permission(self, request, view, obj):
        owner = getattr(obj, "owner", None)
        if owner is not None:
            return owner == request.user

        user = getattr(obj, "user", None)
        if user is not None:
            return user == request.user

        project = getattr(obj, "project", None)
        if project is not None:
            return getattr(project, "owner", None) == request.user

        track = getattr(obj, "track", None)
        if track is not None:
            return getattr(track.project, "owner", None) == request.user

        return False


class ProjectViewSet(viewsets.ModelViewSet):
    """
    CRUD для проектов.
    Владелец проекта — текущий пользователь.
    """
    queryset = Project.objects.none()
    serializer_class = ProjectSerializer
    permission_classes = [permissions.IsAuthenticated, IsOwner]

    def get_queryset(self):
        return Project.objects.filter(owner=self.request.user).order_by('-created_at')

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)


class TrackViewSet(viewsets.ModelViewSet):
    """
    Треки внутри проекта. При создании проверяем, что проект принадлежит пользователю.
    """
    queryset = Track.objects.none()
    serializer_class = TrackSerializer
    permission_classes = [permissions.IsAuthenticated, IsOwner]

    def get_queryset(self):
        qs = Track.objects.filter(project__owner=self.request.user).order_by('order')
        project_id = self.request.query_params.get('project')
        if project_id:
            qs = qs.filter(project_id=project_id)
        return qs

    def perform_create(self, serializer):
        project = serializer.validated_data.get('project')
        if project.owner != self.request.user:
            raise permissions.PermissionDenied("Проект не принадлежит вам.")
        serializer.save()


class AudioFileViewSet(viewsets.ModelViewSet):
    queryset = AudioFile.objects.all().order_by('-created_at')
    serializer_class = AudioFileSerializer

    def get_serializer_context(self):
        ctx = super().get_serializer_context()
        ctx.update({"request": self.request})
        return ctx

    @action(detail=True, methods=['post'])
    def process(self, request, pk=None):
        af = self.get_object()
        if af.status in ('processing', 'queued'):
            return Response({'detail': 'already processing'}, status=status.HTTP_400_BAD_REQUEST)
        af.status = 'queued'
        af.save(update_fields=['status'])
        mode = request.data.get('mode', 'denoise')
        process_audio_file.delay(af.id, mode)
        return Response({'status': 'queued'})


class EffectChainViewSet(viewsets.ModelViewSet):
    """
    Сохранённые цепочки эффектов для трека.
    config — JSONField с параметрами эффектов.
    """
    queryset = EffectChain.objects.none()
    serializer_class = EffectChainSerializer
    permission_classes = [permissions.IsAuthenticated, IsOwner]

    def get_queryset(self):
        return EffectChain.objects.filter(track__project__owner=self.request.user).order_by('id')

    def perform_create(self, serializer):
        track = serializer.validated_data.get('track')
        if track.project.owner != self.request.user:
            raise permissions.PermissionDenied("Трек не принадлежит вам.")
        serializer.save()


class BillingViewSet(viewsets.ModelViewSet):
    """
    Просмотр и управление балансом/кредитами пользователя.
    """
    queryset = Billing.objects.none()
    serializer_class = BillingSerializer
    permission_classes = [permissions.IsAuthenticated, IsOwner]

    def get_queryset(self):
        return Billing.objects.filter(user=self.request.user)

    @action(detail=False, methods=['post'])
    def add_credits(self, request):
        """
        Пример ручного пополнения (используется для тестов или админ-операций).
        Тело: {"amount": 100}
        """
        amount = int(request.data.get('amount', 0))
        if amount <= 0:
            return Response({"detail": "Неверная сумма."}, status=status.HTTP_400_BAD_REQUEST)

        billing, _ = Billing.objects.get_or_create(user=request.user)
        billing.credits += amount
        billing.save()
        return Response(BillingSerializer(billing).data)


class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Просмотр пользователей (ограничено правами).
    """
    queryset = User.objects.none()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        # По умолчанию возвращаем только текущего пользователя
        return User.objects.filter(id=self.request.user.id)
