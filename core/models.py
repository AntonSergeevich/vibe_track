# core/models.py
"""Доменная модель VibeTrack.

Проект → трек → задачи рендера → дорожки, партитура и вокальные дубли.
"""
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone


class User(AbstractUser):
    display_name = models.CharField(max_length=150, blank=True)

    def __str__(self):
        return self.display_name or self.username


class Project(models.Model):
    owner = models.ForeignKey('User', on_delete=models.CASCADE, related_name='projects')
    title = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=30, default='draft')

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title


class Track(models.Model):
    project = models.ForeignKey('Project', on_delete=models.CASCADE, related_name='tracks')
    title = models.CharField(max_length=200)
    order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    analysis = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['order', 'id']

    def __str__(self):
        return self.title

    @property
    def source_file(self):
        return self.audio_files.filter(kind=AudioFile.KIND_SOURCE).order_by('-uploaded_at').first()


class AudioFile(models.Model):
    KIND_SOURCE = 'source'
    KIND_STEM = 'stem'
    KIND_MASTER = 'master'
    KIND_VOCAL_TAKE = 'vocal_take'
    KIND_CHOICES = [
        (KIND_SOURCE, 'Исходник'),
        (KIND_STEM, 'Дорожка'),
        (KIND_MASTER, 'Мастер'),
        (KIND_VOCAL_TAKE, 'Вокальный дубль'),
    ]
    STATUS_CHOICES = [
        ('new', 'New'),
        ('queued', 'Queued'),
        ('processing', 'Processing'),
        ('done', 'Done'),
        ('error', 'Error'),
    ]

    track = models.ForeignKey('Track', on_delete=models.CASCADE, related_name='audio_files')
    file = models.FileField(upload_to='audio/')
    kind = models.CharField(max_length=20, choices=KIND_CHOICES, default=KIND_SOURCE)
    duration = models.FloatField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new')

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.get_kind_display()}: {self.file.name}"


class RenderJob(models.Model):
    """Одна задача превращения трека в ню-метал."""

    STATUS_QUEUED = 'queued'
    STATUS_RUNNING = 'running'
    STATUS_DONE = 'done'
    STATUS_ERROR = 'error'
    STATUS_CHOICES = [
        (STATUS_QUEUED, 'В очереди'),
        (STATUS_RUNNING, 'Обрабатывается'),
        (STATUS_DONE, 'Готово'),
        (STATUS_ERROR, 'Ошибка'),
    ]

    track = models.ForeignKey('Track', on_delete=models.CASCADE, related_name='render_jobs')
    prompt = models.TextField(blank=True)
    overrides = models.JSONField(default=dict, blank=True)   # инструменты, строй, вокал из UI
    options = models.JSONField(default=dict, blank=True)     # опции конвейера
    spec = models.JSONField(default=dict, blank=True)        # итоговая ArrangementSpec
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_QUEUED)
    stage = models.CharField(max_length=40, blank=True)
    progress = models.IntegerField(default=0)
    celery_task_id = models.CharField(max_length=100, blank=True)
    master = models.FileField(upload_to='renders/', blank=True, null=True)
    result = models.JSONField(default=dict, blank=True)
    warnings = models.JSONField(default=list, blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"RenderJob #{self.pk} ({self.status})"


class Stem(models.Model):
    """Отдельная дорожка результата (гитара, бас, барабаны, вокал…)."""

    job = models.ForeignKey('RenderJob', on_delete=models.CASCADE, related_name='stems')
    name = models.CharField(max_length=60)
    label = models.CharField(max_length=120, blank=True)
    file = models.FileField(upload_to='stems/')
    source = models.CharField(max_length=20, default='generated')  # generated | source
    peak_db = models.FloatField(null=True, blank=True)
    rms_db = models.FloatField(null=True, blank=True)
    duration = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


class Score(models.Model):
    """Партитура: текст, аккорды, табы."""

    job = models.OneToOneField('RenderJob', on_delete=models.CASCADE, related_name='score')
    lyrics = models.JSONField(default=dict, blank=True)
    lyric_sheet = models.TextField(blank=True)
    chords = models.JSONField(default=list, blank=True)
    chord_chart = models.TextField(blank=True)
    tabs = models.JSONField(default=dict, blank=True)
    key = models.CharField(max_length=20, blank=True)
    tempo = models.FloatField(null=True, blank=True)
    tuning = models.CharField(max_length=30, blank=True)

    def __str__(self):
        return f"Score для #{self.job_id}"


class VocalTake(models.Model):
    """Записанный пользователем голос и его автообработка."""

    STATUS_CHOICES = RenderJob.STATUS_CHOICES

    track = models.ForeignKey('Track', on_delete=models.CASCADE, related_name='vocal_takes')
    job = models.ForeignKey('RenderJob', on_delete=models.SET_NULL, null=True, blank=True,
                            related_name='vocal_takes')
    raw_file = models.FileField(upload_to='takes/')
    processed_file = models.FileField(upload_to='takes/processed/', blank=True, null=True)
    mixed_file = models.FileField(upload_to='takes/mixed/', blank=True, null=True)
    style = models.CharField(max_length=20, default='rap')          # rap|scream|clean|whisper
    gender = models.CharField(max_length=10, default='male')
    target_gender = models.CharField(max_length=10, blank=True)     # перекраска голоса
    autotune = models.FloatField(null=True, blank=True)
    gain_db = models.FloatField(default=-3.0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    latency_ms = models.FloatField(null=True, blank=True)
    notes = models.JSONField(default=list, blank=True)
    error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"VocalTake #{self.pk} ({self.style})"


class Subscription(models.Model):
    """Активная подписка пользователя на тариф."""

    user = models.ForeignKey('User', on_delete=models.CASCADE, related_name='subscriptions')
    plan = models.CharField(max_length=20)          # slug из core.billing.PLANS
    started_at = models.DateTimeField(default=timezone.now)
    expires_at = models.DateTimeField()
    auto_renew = models.BooleanField(default=False)
    payment = models.ForeignKey('Payment', on_delete=models.SET_NULL, null=True, blank=True,
                                related_name='subscriptions')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-expires_at']

    def __str__(self):
        return f"{self.user}: {self.plan} до {self.expires_at:%d.%m.%Y}"

    @property
    def is_active(self) -> bool:
        return self.expires_at > timezone.now()


class UsageRecord(models.Model):
    """Списание или возврат одного рендера."""

    KIND_CHARGE = 'charge'
    KIND_REFUND = 'refund'
    KIND_COVER = 'cover'
    KIND_CHOICES = [(KIND_CHARGE, 'Списание'), (KIND_REFUND, 'Возврат'),
                    (KIND_COVER, 'Генерация кавера')]

    user = models.ForeignKey('User', on_delete=models.CASCADE, related_name='usage')
    job = models.ForeignKey('RenderJob', on_delete=models.SET_NULL, null=True, blank=True,
                            related_name='usage')
    kind = models.CharField(max_length=10, choices=KIND_CHOICES, default=KIND_CHARGE)
    plan = models.CharField(max_length=20, blank=True)
    note = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['user', 'created_at'])]

    def __str__(self):
        return f"{self.get_kind_display()} #{self.job_id} ({self.user})"


class Payment(models.Model):
    """Платёж за тариф. Провайдер подключается через webhook."""

    STATUS_PENDING = 'pending'
    STATUS_PAID = 'paid'
    STATUS_FAILED = 'failed'
    STATUS_REFUNDED = 'refunded'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Ожидает оплаты'),
        (STATUS_PAID, 'Оплачен'),
        (STATUS_FAILED, 'Не прошёл'),
        (STATUS_REFUNDED, 'Возвращён'),
    ]

    user = models.ForeignKey('User', on_delete=models.CASCADE, related_name='payments')
    plan = models.CharField(max_length=20)
    amount_rub = models.IntegerField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    provider = models.CharField(max_length=30, blank=True)      # yookassa, robokassa, ...
    external_id = models.CharField(max_length=120, blank=True)  # id платежа у провайдера
    confirmation_url = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    paid_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.amount_rub} ₽ за {self.plan} ({self.get_status_display()})"


class EffectChain(models.Model):
    track = models.ForeignKey('Track', on_delete=models.CASCADE, related_name='effect_chains')
    name = models.CharField(max_length=100)
    config = models.JSONField(default=dict)

    def __str__(self):
        return self.name


class Billing(models.Model):
    user = models.ForeignKey('User', on_delete=models.CASCADE, related_name='billings')
    credits = models.IntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user}: {self.credits} кредитов"
