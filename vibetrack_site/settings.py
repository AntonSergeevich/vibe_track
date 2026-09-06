"""Настройки VibeTrack.

Всё, что зависит от окружения, читается из переменных среды — так один и
тот же код работает и локально на SQLite, и в docker-compose с Postgres.
"""
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> None:
    """Подхватывает переменные из .env рядом с manage.py.

    Без этого ключ API пришлось бы задавать в каждом новом окне терминала и
    отдельно в конфигурации запуска PyCharm — и однажды он бы там не совпал.
    Уже заданные переменные окружения имеют приоритет: на сервере настройки
    приходят из окружения, а не из файла.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        # значение до комментария, без кавычек: VIBETRACK_X=0.75  # подсказка
        value = value.split(" #")[0].strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


load_env_file(BASE_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "django-insecure-r$bnt)nsxmj4p1%$*^dfnw4d$+g0lkx&sy*h17!@36w&=2=wn#",
)
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,0.0.0.0,[::1]")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS")

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'core',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# whitenoise ставится не всегда (например, в лёгком тестовом окружении)
try:  # pragma: no cover
    import whitenoise  # noqa: F401
except ImportError:  # pragma: no cover
    MIDDLEWARE.remove('whitenoise.middleware.WhiteNoiseMiddleware')

ROOT_URLCONF = 'vibetrack_site.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'vibetrack_site.wsgi.application'
ASGI_APPLICATION = 'vibetrack_site.asgi.application'

if os.getenv("DATABASE_URL", "").startswith("postgres"):
    from urllib.parse import urlparse

    url = urlparse(os.environ["DATABASE_URL"])
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': url.path.lstrip('/'),
            'USER': url.username or '',
            'PASSWORD': url.password or '',
            'HOST': url.hostname or 'localhost',
            'PORT': str(url.port or 5432),
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'ru-ru'
TIME_ZONE = os.getenv("DJANGO_TIME_ZONE", "Asia/Krasnoyarsk")
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static'] if (BASE_DIR / 'static').exists() else []
STATIC_ROOT = BASE_DIR / 'staticfiles'
MEDIA_URL = '/media/'
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", BASE_DIR / 'media'))

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
AUTH_USER_MODEL = 'core.User'

# Загружаемые треки бывают большими — не держим их целиком в памяти
DATA_UPLOAD_MAX_MEMORY_SIZE = int(os.getenv("DATA_UPLOAD_MAX_MEMORY_SIZE", 5 * 1024 * 1024))
FILE_UPLOAD_MAX_MEMORY_SIZE = DATA_UPLOAD_MAX_MEMORY_SIZE
VIBETRACK_MAX_UPLOAD_MB = int(os.getenv("VIBETRACK_MAX_UPLOAD_MB", 80))
# Требовать вход для обработки треков (в проде — да, локально удобнее без)
VIBETRACK_REQUIRE_LOGIN = env_bool("VIBETRACK_REQUIRE_LOGIN", False)
# Тестовый режим оплаты: покупка тарифа активируется сразу, без провайдера
VIBETRACK_PAYMENTS_TEST_MODE = env_bool("VIBETRACK_PAYMENTS_TEST_MODE", False)

LOGIN_URL = '/accounts/login/'
LOGIN_REDIRECT_URL = '/cabinet/'
LOGOUT_REDIRECT_URL = '/'

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny' if env_bool("VIBETRACK_OPEN_API", True)
        else 'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 25,
}

# ------------------------------------------------------------------ Celery
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/1')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = int(os.getenv('CELERY_TASK_TIME_LIMIT', 3600))
CELERY_WORKER_MAX_TASKS_PER_CHILD = 8   # рендер держит память — перезапускаем воркер
# Без брокера (локальная разработка, тесты) задачи выполняются синхронно
CELERY_TASK_ALWAYS_EAGER = env_bool('CELERY_TASK_ALWAYS_EAGER', False)
# Без Redis: выполнять задачи в фоновом потоке, чтобы работала полоса прогресса
# Eager-режим выполняет задачу внутри HTTP-запроса, и прогресс стоять будет
# всегда — поэтому вместе с ним по умолчанию включаем фоновый поток
VIBETRACK_INLINE_WORKER = env_bool('VIBETRACK_INLINE_WORKER', CELERY_TASK_ALWAYS_EAGER)
# Папка с живыми сэмплами: барабанный луп и гитара
VIBETRACK_SAMPLES_DIR = os.getenv('VIBETRACK_SAMPLES_DIR', str(BASE_DIR / 'samples'))
CELERY_TASK_EAGER_PROPAGATES = True

# ------------------------------------------------------------- Аудиодвижок
VIBETRACK = {
    'SAMPLE_RATE': int(os.getenv('VIBETRACK_SAMPLE_RATE', 44100)),
    'EXPORT_FORMAT': os.getenv('VIBETRACK_EXPORT_FORMAT', 'wav'),   # wav | mp3 (нужен ffmpeg)
    'MAX_DURATION': float(os.getenv('VIBETRACK_MAX_DURATION', 480)),
    'DEMUCS_MODEL': os.getenv('VIBETRACK_DEMUCS_MODEL', 'htdemucs'),
    'SEPARATION_BACKEND': os.getenv('VIBETRACK_SEPARATION_BACKEND', 'auto'),  # auto|demucs|dsp
    'WHISPER_MODEL': os.getenv('VIBETRACK_WHISPER_MODEL', 'small'),
    'MASTER_LOUDNESS_DB': float(os.getenv('VIBETRACK_MASTER_LOUDNESS_DB', -10.0)),
    # Внешняя нейросеть-генератор: none | suno_proxy | elevenlabs | stability | custom
    'MUSIC_PROVIDER': os.getenv('VIBETRACK_MUSIC_PROVIDER', 'none'),
}

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'simple': {'format': '[{levelname}] {name}: {message}', 'style': '{'}},
    'handlers': {'console': {'class': 'logging.StreamHandler', 'formatter': 'simple'}},
    'root': {'handlers': ['console'], 'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO')},
}
