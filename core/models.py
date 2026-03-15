# core/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    display_name = models.CharField(max_length=150, blank=True)

class Project(models.Model):
    owner = models.ForeignKey('User', on_delete=models.CASCADE, related_name='projects')
    title = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=30, default='draft')

class Track(models.Model):
    project = models.ForeignKey('Project', on_delete=models.CASCADE, related_name='tracks')
    title = models.CharField(max_length=200)
    order = models.IntegerField(default=0)

class AudioFile(models.Model):
    track = models.ForeignKey('Track', on_delete=models.CASCADE, related_name='audio_files')
    file = models.FileField(upload_to='audio/')
    duration = models.FloatField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

class EffectChain(models.Model):
    track = models.ForeignKey('Track', on_delete=models.CASCADE, related_name='effect_chains')
    name = models.CharField(max_length=100)
    config = models.JSONField(default=dict)  # store chain params, presets

class Billing(models.Model):
    user = models.ForeignKey('User', on_delete=models.CASCADE, related_name='billings')
    credits = models.IntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)
