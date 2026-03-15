"""
URL configuration for vibetrack_site project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# vibetrack_site/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from core.views import ProjectViewSet, TrackViewSet, AudioFileViewSet, EffectChainViewSet

router = DefaultRouter()
router.register(r'projects', ProjectViewSet)
# register others...

urlpatterns = [
    path('api/', include(router.urls)),
]

