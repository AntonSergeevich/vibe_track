from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import (AudioFile, Billing, EffectChain, Payment, Project, RenderJob, Score,
                     Stem, Subscription, Track, UsageRecord, User, VocalTake)


@admin.register(User)
class VibeUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (('VibeTrack', {'fields': ('display_name',)}),)


class StemInline(admin.TabularInline):
    model = Stem
    extra = 0


@admin.register(RenderJob)
class RenderJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'track', 'status', 'stage', 'progress', 'created_at')
    list_filter = ('status',)
    search_fields = ('prompt', 'track__title')
    inlines = [StemInline]


@admin.register(Track)
class TrackAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'project', 'order')
    search_fields = ('title',)


@admin.register(VocalTake)
class VocalTakeAdmin(admin.ModelAdmin):
    list_display = ('id', 'track', 'style', 'gender', 'status', 'latency_ms', 'created_at')
    list_filter = ('status', 'style')


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'plan', 'amount_rub', 'status', 'created_at', 'paid_at')
    list_filter = ('status', 'plan')
    search_fields = ('user__username', 'external_id')
    actions = ['mark_paid']

    @admin.action(description='Отметить оплаченным и включить подписку')
    def mark_paid(self, request, queryset):
        from .views import activate_payment

        activated = 0
        for payment in queryset.filter(status=Payment.STATUS_PENDING):
            activate_payment(payment)
            activated += 1
        self.message_user(request, f'Подписок активировано: {activated}')


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'plan', 'started_at', 'expires_at', 'auto_renew')
    list_filter = ('plan', 'auto_renew')


@admin.register(UsageRecord)
class UsageRecordAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'kind', 'plan', 'job', 'created_at')
    list_filter = ('kind', 'plan')


admin.site.register([Project, AudioFile, Stem, Score, EffectChain, Billing])
admin.site.site_header = 'VibeTrack — студия ню-метала'
