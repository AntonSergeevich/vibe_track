# core/urls.py
from rest_framework.routers import DefaultRouter
from .views import AudioFileViewSet
from django.urls import path, include

router = DefaultRouter()
router.register(r'audiofiles', AudioFileViewSet, basename='audiofile')

urlpatterns = [
    path('', include(router.urls)),
]
