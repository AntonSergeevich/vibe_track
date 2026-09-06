# core/billing.py
"""Тарифы, лимиты и учёт списаний.

Правила простые и жёсткие:
  * лимит считается за расчётный период подписки, а не за календарный месяц;
  * списание происходит при постановке в очередь, иначе можно наплодить
    задач быстрее, чем они успеют посчитаться;
  * если рендер упал не по вине пользователя — списание возвращается.
    Сгоревший трек за нашу же ошибку — самая обидная причина ухода.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from django.utils import timezone


@dataclass(frozen=True)
class PlanSpec:
    slug: str
    name: str
    price_rub: int
    tracks: int                  # сколько рендеров даёт тариф
    recurring: bool = True       # False — разовая покупка
    wav_stems: bool = False      # дорожки в WAV, а не только mp3
    demucs: bool = True          # студийное разделение исходника
    priority: bool = False       # приоритет в очереди
    storage_days: int = 30
    commercial: bool = False     # коммерческая лицензия на результат


PLANS: dict[str, PlanSpec] = {
    "free": PlanSpec("free", "Проба", 0, 2, wav_stems=False, demucs=False,
                     storage_days=7),
    "single": PlanSpec("single", "Разовый трек", 190, 1, recurring=False,
                       wav_stems=True, storage_days=90, commercial=True),
    "start": PlanSpec("start", "Старт", 390, 40, storage_days=60),
    "studio": PlanSpec("studio", "Студия", 690, 90, wav_stems=True,
                       priority=True, storage_days=90, commercial=True),
    "pro": PlanSpec("pro", "Продакшн", 1690, 200, wav_stems=True, priority=True,
                    storage_days=3650, commercial=True),
}
# Тариф для владельца сервиса и тестировщиков: не продаётся, выдаётся правами
PLANS["unlimited"] = PlanSpec("unlimited", "Безлимит (разработчик)", 0, 1_000_000,
                              recurring=False, wav_stems=True, demucs=True,
                              priority=True, storage_days=3650, commercial=True)

DEFAULT_PLAN = "free"


class QuotaExceeded(Exception):
    """Лимит тарифа исчерпан."""

    def __init__(self, plan: PlanSpec, used: int, resets_at=None):
        self.plan, self.used, self.resets_at = plan, used, resets_at
        if plan.slug == "free":
            message = (f"Бесплатных треков в этом месяце больше нет "
                       f"({used} из {plan.tracks}). Разовый трек — 190 ₽, "
                       f"подписка «Старт» — 390 ₽ за 40 треков.")
        else:
            when = f" Лимит обновится {resets_at:%d.%m.%Y}." if resets_at else ""
            message = (f"На тарифе «{plan.name}» израсходовано {used} треков "
                       f"из {plan.tracks}.{when}")
        super().__init__(message)


def is_unlimited(user) -> bool:
    """Владелец сервиса и тестировщики работают без лимитов.

    Проверяется по правам (staff/superuser) и по списку имён в
    VIBETRACK_UNLIMITED_USERS — чтобы не выдавать себе доступ в админку
    только ради того, чтобы обработать трек.
    """
    import os

    if not getattr(user, "is_authenticated", False):
        return False
    if getattr(user, "is_staff", False) or getattr(user, "is_superuser", False):
        return True
    allowed = {name.strip().lower()
               for name in os.getenv("VIBETRACK_UNLIMITED_USERS", "").split(",")
               if name.strip()}
    return str(getattr(user, "username", "")).lower() in allowed


def plan_for(user) -> PlanSpec:
    """Действующий тариф пользователя."""
    from .models import Subscription

    if not getattr(user, "is_authenticated", False):
        return PLANS[DEFAULT_PLAN]
    if is_unlimited(user):
        return PLANS["unlimited"]
    sub = (Subscription.objects
           .filter(user=user, expires_at__gt=timezone.now())
           .order_by("-expires_at")
           .first())
    if sub is None:
        return PLANS[DEFAULT_PLAN]
    return PLANS.get(sub.plan, PLANS[DEFAULT_PLAN])


def period_start(user) -> "timezone.datetime":
    """Начало текущего расчётного периода.

    У подписчика период отсчитывается от даты покупки, у бесплатного
    пользователя — от начала календарного месяца.
    """
    from .models import Subscription

    now = timezone.now()
    if getattr(user, "is_authenticated", False):
        sub = (Subscription.objects
               .filter(user=user, expires_at__gt=now)
               .order_by("-expires_at")
               .first())
        if sub is not None:
            start = sub.started_at
            while start + timedelta(days=30) <= now:
                start += timedelta(days=30)
            return start
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def period_end(user):
    from .models import Subscription

    now = timezone.now()
    sub = None
    if getattr(user, "is_authenticated", False):
        sub = (Subscription.objects
               .filter(user=user, expires_at__gt=now)
               .order_by("-expires_at")
               .first())
    if sub is not None:
        return period_start(user) + timedelta(days=30)
    start = period_start(user)
    return (start + timedelta(days=32)).replace(day=1)


def used_this_period(user) -> int:
    """Сколько рендеров списано за текущий период (возвраты не считаются)."""
    from .models import UsageRecord

    if not getattr(user, "is_authenticated", False):
        return 0
    charges = UsageRecord.objects.filter(
        user=user, created_at__gte=period_start(user))
    spent = charges.filter(kind=UsageRecord.KIND_CHARGE).count()
    refunded = charges.filter(kind=UsageRecord.KIND_REFUND).count()
    return max(spent - refunded, 0)


def remaining(user) -> int:
    return max(plan_for(user).tracks - used_this_period(user), 0)


def check_quota(user) -> PlanSpec:
    """Бросает QuotaExceeded, если лимит исчерпан. Иначе возвращает тариф."""
    plan = plan_for(user)
    if plan.slug == "unlimited":
        return plan
    used = used_this_period(user)
    if used >= plan.tracks:
        raise QuotaExceeded(plan, used, period_end(user))
    return plan


def charge(user, job, note: str = "") -> "object | None":
    """Списывает один рендер при постановке задачи в очередь."""
    from .models import UsageRecord

    if not getattr(user, "is_authenticated", False):
        return None
    return UsageRecord.objects.create(
        user=user, job=job, kind=UsageRecord.KIND_CHARGE,
        plan=plan_for(user).slug, note=note)


def refund(user, job, reason: str) -> "object | None":
    """Возвращает списание, если рендер упал не по вине пользователя."""
    from .models import UsageRecord

    if not getattr(user, "is_authenticated", False):
        return None
    already = UsageRecord.objects.filter(
        user=user, job=job, kind=UsageRecord.KIND_REFUND).exists()
    if already:
        return None
    return UsageRecord.objects.create(
        user=user, job=job, kind=UsageRecord.KIND_REFUND,
        plan=plan_for(user).slug, note=reason)


def render_options_for(plan: PlanSpec) -> dict:
    """Опции конвейера, вытекающие из тарифа."""
    return {
        "export_format": "wav" if plan.wav_stems else "mp3",
        "separation_backend": "auto" if plan.demucs else "dsp",
    }


def summary(user) -> dict:
    """Данные для личного кабинета и для API."""
    plan = plan_for(user)
    used = used_this_period(user)
    return {
        "plan": {"slug": plan.slug, "name": plan.name, "price_rub": plan.price_rub,
                 "tracks": plan.tracks, "wav_stems": plan.wav_stems,
                 "demucs": plan.demucs, "priority": plan.priority,
                 "storage_days": plan.storage_days, "commercial": plan.commercial},
        "used": used,
        "remaining": max(plan.tracks - used, 0),
        "period_start": period_start(user) if getattr(user, "is_authenticated", False) else None,
        "period_end": period_end(user) if getattr(user, "is_authenticated", False) else None,
        "unlimited": plan.slug == "unlimited",
        "plans": [{"slug": p.slug, "name": p.name, "price_rub": p.price_rub,
                   "tracks": p.tracks, "recurring": p.recurring,
                   "wav_stems": p.wav_stems, "commercial": p.commercial}
                  for p in PLANS.values() if p.slug != "unlimited"],
    }
