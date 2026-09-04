from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import (AudioFile, Billing, EffectChain, Project, RenderJob, Score, Stem,
                     Track, User, VocalTake)


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


admin.site.register([Project, AudioFile, Stem, Score, EffectChain, Billing])
admin.site.site_header = 'VibeTrack — студия ню-метала'
