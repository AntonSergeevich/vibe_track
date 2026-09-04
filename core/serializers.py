# core/serializers.py
from django.contrib.auth import get_user_model
from rest_framework import serializers

from engine.arrangement import (BASS_TUNINGS, GROOVE_KEYWORDS, INSTRUMENT_CATALOG,
                                TUNINGS, VOCAL_STYLE_KEYWORDS)

from .models import (AudioFile, Billing, EffectChain, Project, RenderJob, Score, Stem,
                     Track, VocalTake)

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'display_name', 'is_staff')
        read_only_fields = ('id', 'is_staff')


class FileUrlMixin(serializers.Serializer):
    def _url(self, file_field):
        if not file_field:
            return None
        request = self.context.get('request')
        return request.build_absolute_uri(file_field.url) if request else file_field.url


class AudioFileSerializer(FileUrlMixin, serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()

    class Meta:
        model = AudioFile
        fields = ('id', 'track', 'file', 'file_url', 'kind', 'status', 'duration', 'uploaded_at')
        read_only_fields = ('status', 'duration', 'uploaded_at', 'file_url')

    def get_file_url(self, obj):
        return self._url(obj.file)


class StemSerializer(FileUrlMixin, serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()

    class Meta:
        model = Stem
        fields = ('id', 'name', 'label', 'source', 'file_url', 'peak_db', 'rms_db', 'duration')

    def get_file_url(self, obj):
        return self._url(obj.file)


class ScoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Score
        fields = ('lyrics', 'lyric_sheet', 'chords', 'chord_chart', 'tabs', 'key', 'tempo', 'tuning')


class RenderJobSerializer(FileUrlMixin, serializers.ModelSerializer):
    stems = StemSerializer(many=True, read_only=True)
    score = ScoreSerializer(read_only=True)
    master_url = serializers.SerializerMethodField()

    class Meta:
        model = RenderJob
        fields = ('id', 'track', 'prompt', 'overrides', 'options', 'spec', 'status', 'stage',
                  'progress', 'master_url', 'result', 'warnings', 'error', 'stems', 'score',
                  'created_at', 'updated_at')
        read_only_fields = ('spec', 'status', 'stage', 'progress', 'master_url', 'result',
                            'warnings', 'error', 'stems', 'score', 'created_at', 'updated_at')

    def get_master_url(self, obj):
        return self._url(obj.master)


class RenderJobCreateSerializer(serializers.Serializer):
    """Запрос на превращение трека в ню-метал."""

    prompt = serializers.CharField(required=False, allow_blank=True, default='')
    overrides = serializers.DictField(required=False, default=dict)
    options = serializers.DictField(required=False, default=dict)

    def validate_overrides(self, value):
        instruments = value.get('instruments')
        if instruments is not None:
            unknown = set(instruments) - set(INSTRUMENT_CATALOG)
            if unknown:
                raise serializers.ValidationError(f"Неизвестные инструменты: {sorted(unknown)}")
        tuning = value.get('tuning')
        if tuning and tuning not in TUNINGS:
            raise serializers.ValidationError(f"Неизвестный строй: {tuning}")
        bass_tuning = value.get('bass_tuning')
        if bass_tuning and bass_tuning not in BASS_TUNINGS:
            raise serializers.ValidationError(f"Неизвестный строй баса: {bass_tuning}")
        groove = value.get('groove')
        if groove and groove not in GROOVE_KEYWORDS:
            raise serializers.ValidationError(f"Неизвестный грув: {groove}")
        return value


class VocalTakeSerializer(FileUrlMixin, serializers.ModelSerializer):
    processed_url = serializers.SerializerMethodField()
    mixed_url = serializers.SerializerMethodField()
    raw_url = serializers.SerializerMethodField()

    class Meta:
        model = VocalTake
        fields = ('id', 'track', 'job', 'raw_file', 'raw_url', 'processed_url', 'mixed_url',
                  'style', 'gender', 'target_gender', 'autotune', 'gain_db', 'status',
                  'latency_ms', 'notes', 'error', 'created_at')
        read_only_fields = ('status', 'latency_ms', 'notes', 'error', 'created_at',
                            'processed_url', 'mixed_url', 'raw_url')
        extra_kwargs = {'raw_file': {'write_only': True}}

    def validate_style(self, value):
        if value not in VOCAL_STYLE_KEYWORDS:
            raise serializers.ValidationError(f"Стиль должен быть одним из {list(VOCAL_STYLE_KEYWORDS)}")
        return value

    def validate_gender(self, value):
        if value not in ('male', 'female'):
            raise serializers.ValidationError("gender: male или female")
        return value

    def validate_target_gender(self, value):
        if value and value not in ('male', 'female'):
            raise serializers.ValidationError("target_gender: male, female или пусто")
        return value

    def get_processed_url(self, obj):
        return self._url(obj.processed_file)

    def get_mixed_url(self, obj):
        return self._url(obj.mixed_file)

    def get_raw_url(self, obj):
        return self._url(obj.raw_file)


class TrackSerializer(serializers.ModelSerializer):
    audio_files = AudioFileSerializer(many=True, read_only=True)
    last_job = serializers.SerializerMethodField()

    class Meta:
        model = Track
        fields = ('id', 'project', 'title', 'order', 'analysis', 'audio_files', 'last_job')
        read_only_fields = ('id', 'analysis')

    def get_last_job(self, obj):
        job = obj.render_jobs.first()
        return {'id': job.id, 'status': job.status, 'progress': job.progress} if job else None


class TrackUploadSerializer(serializers.Serializer):
    """Загрузка исходника: файл + необязательные проект и название."""

    file = serializers.FileField()
    title = serializers.CharField(required=False, allow_blank=True)
    project = serializers.PrimaryKeyRelatedField(queryset=Project.objects.all(), required=False)


class ProjectSerializer(serializers.ModelSerializer):
    tracks = TrackSerializer(many=True, read_only=True)

    class Meta:
        model = Project
        fields = ('id', 'owner', 'title', 'created_at', 'status', 'tracks')
        read_only_fields = ('id', 'owner', 'created_at')


class EffectChainSerializer(serializers.ModelSerializer):
    class Meta:
        model = EffectChain
        fields = ('id', 'track', 'name', 'config')
        read_only_fields = ('id',)


class BillingSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = Billing
        fields = ('id', 'user', 'credits', 'updated_at')
        read_only_fields = ('id', 'user', 'updated_at')
