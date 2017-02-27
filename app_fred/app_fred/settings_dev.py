from app_fred.settings import *

# Database
# https://docs.djangoproject.com/en/1.10/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': DEV_DB_NAME,
        'USER': DEV_DB_USR,
        'PASSWORD': DEV_DB_PASSWORD,
        'HOST': '127.0.0.1',
        'PORT': '3306',
        'TEST': {
            'NAME': 'test_mbuehler',
        },
    }
}
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True


PATH_PPLIB = '/home/marcello/studies/bachelorarbeit/workspace/PPLib'