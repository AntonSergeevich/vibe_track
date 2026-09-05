# vibetrack_site/urls.py
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

from core.views import cabinet, register, studio

urlpatterns = [
    path('', studio, name='studio'),
    path('cabinet/', cabinet, name='cabinet'),
    path('accounts/register/', register, name='register'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('admin/', admin.site.urls),
    path('api/', include('core.urls')),
    path('api-auth/', include('rest_framework.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
