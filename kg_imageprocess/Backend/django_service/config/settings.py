import os
from pathlib import Path

# BASE_DIR: پوشه اصلی پروژه جانگو (پوشه django_service)
BASE_DIR = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------
# 1. تنظیمات امنیتی پایه (Security Settings)
# ----------------------------------------------------------------------
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'django-insecure-local-dev-key-change-in-production')

DEBUG = True

ALLOWED_HOSTS = ['*']  # در محیط توسعه همه هاس‌ها مجاز هستند


# ----------------------------------------------------------------------
# 2. اپلیکیشن‌های نصب‌شده (Installed Apps)
# ----------------------------------------------------------------------
INSTALLED_APPS = [
    # پیش‌فرض‌های جانگو
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # پکیج‌های شخص ثالث (Third-party)
    'rest_framework',
    'corsheaders',

    # اپلیکیشن‌های داخلی پروژه (Local Apps)
    'apps.users',
    'apps.workouts',
    'apps.payments',
]


# ----------------------------------------------------------------------
# 3. میدلورها (Middleware)
# ----------------------------------------------------------------------
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # میدلور مدیریت CORS (بالای بقیه باشه)
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'


# ----------------------------------------------------------------------
# 4. پایگاه داده (PostgreSQL Database Config)
# ----------------------------------------------------------------------
# django_service/config/settings.py

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'biomechanics_db',
        'USER': 'postgres',
        'PASSWORD': 'postgres',
        'HOST': '127.0.0.1',
        'PORT': '5433',  # پورت جدید کانتینر داکر
    }
}


# ----------------------------------------------------------------------
# 5. مدل کاربر سفارشی (Custom User Model)
# ----------------------------------------------------------------------
# حتماً باید اشاره کنه به مدلی که در apps.users.models ساختیم
AUTH_USER_MODEL = 'users.User'


# ----------------------------------------------------------------------
# 6. اعتبارسنجی رمز عبور (Password Validation)
# ----------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# ----------------------------------------------------------------------
# 7. بومی‌سازی و زمان (Internationalization)
# ----------------------------------------------------------------------
LANGUAGE_CODE = 'fa-ir'
TIME_ZONE = 'Asia/Tehran'
USE_I18N = True
USE_TZ = True


# ----------------------------------------------------------------------
# 8. فایل‌های استاتیک و رسانه (Static & Media Files)
# ----------------------------------------------------------------------
STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# ----------------------------------------------------------------------
# 9. تنظیمات Django REST Framework (DRF)
# ----------------------------------------------------------------------
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',  # در فاز توسعه
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
}


# ----------------------------------------------------------------------
# 10. تنظیمات CORS (ارتباط بدون مشکل با FastAPI و Flutter)
# ----------------------------------------------------------------------
CORS_ALLOW_ALL_ORIGINS = True  # در محیط توسعه
