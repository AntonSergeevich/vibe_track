"""Выдать пользователю тариф без оплаты.

    python manage.py grant_plan anton studio --days 365
    python manage.py grant_plan anton --unlimited

Нужно для себя, тестировщиков и разбора спорных ситуаций с оплатой.
"""
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from core import billing
from core.models import Payment, Subscription, User


class Command(BaseCommand):
    help = "Выдаёт тариф пользователю без платежа."

    def add_arguments(self, parser):
        parser.add_argument("username", nargs="?",
                            help="логин; без него команда просто покажет список пользователей")
        parser.add_argument("plan", nargs="?", default="studio",
                            help=f"один из: {', '.join(billing.PLANS)}")
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--unlimited", action="store_true",
                            help="Выдать безлимит разработчика (через права staff)")

    def handle(self, *args, **options):
        if not options["username"]:
            self._list_users()
            return
        try:
            user = User.objects.get(username=options["username"])
        except User.DoesNotExist:
            # без подсказки приходится лезть в админку — а логин часто просто забыт
            self._list_users()
            raise CommandError(f"Пользователь «{options['username']}» не найден")

        if options["unlimited"]:
            user.is_staff = True
            user.save(update_fields=["is_staff"])
            self.stdout.write(self.style.SUCCESS(
                f"{user.username}: безлимит включён (права staff). "
                f"Тариф теперь «{billing.plan_for(user).name}»."))
            return

        plan = billing.PLANS.get(options["plan"])
        if plan is None or plan.slug in ("free", "unlimited"):
            raise CommandError(f"Тариф должен быть одним из: "
                               f"{', '.join(p for p in billing.PLANS if p not in ('free', 'unlimited'))}")

        payment = Payment.objects.create(
            user=user, plan=plan.slug, amount_rub=0, provider="manual",
            status=Payment.STATUS_PAID, paid_at=timezone.now())
        Subscription.objects.create(
            user=user, plan=plan.slug, expires_at=timezone.now() + timedelta(days=options["days"]),
            payment=payment)
        self.stdout.write(self.style.SUCCESS(
            f"{user.username}: тариф «{plan.name}» на {options['days']} дней, "
            f"{plan.tracks} треков в месяц."))

    def _list_users(self):
        users = User.objects.order_by("username")
        if not users:
            self.stdout.write("Пользователей пока нет — зарегистрируйтесь на /accounts/register/")
            return
        self.stdout.write("Пользователи:")
        for user in users:
            marks = []
            if user.is_superuser:
                marks.append("суперпользователь")
            elif user.is_staff:
                marks.append("staff — безлимит")
            plan = billing.plan_for(user)
            marks.append(f"тариф «{plan.name}»")
            self.stdout.write(f"  {user.username:20s} {', '.join(marks)}")
