# core/serializers.py
from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Project, Track, AudioFile, EffectChain, Billing

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'display_name', 'is_staff')
        read_only_fields = ('id', 'is_staff')

class AudioFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioFile

class TrackSerializer(serializers.ModelSerializer):
    audio_files = AudioFileSerializer(many=True, read_only=True)

    class Meta:
        model = Track
        fields = ('id', 'project', 'title', 'order', 'audio_files')
        read_only_fields = ('id',)

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
