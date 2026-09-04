# core/urls.py
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (AudioFileViewSet, BillingViewSet, EffectChainViewSet, ProjectViewSet,
                    RenderJobViewSet, TrackViewSet, UserViewSet, VocalTakeViewSet,
                    capabilities, parse_description)

router = DefaultRouter()
router.register(r'projects', ProjectViewSet, basename='project')
router.register(r'tracks', TrackViewSet, basename='track')
router.register(r'renders', RenderJobViewSet, basename='render')
router.register(r'vocal-takes', VocalTakeViewSet, basename='vocaltake')
router.register(r'audiofiles', AudioFileViewSet, basename='audiofile')
router.register(r'effect-chains', EffectChainViewSet, basename='effectchain')
router.register(r'billing', BillingViewSet, basename='billing')
router.register(r'users', UserViewSet, basename='user')

urlpatterns = [
    path('capabilities/', capabilities, name='capabilities'),
    path('parse-description/', parse_description, name='parse-description'),
    path('', include(router.urls)),
]
