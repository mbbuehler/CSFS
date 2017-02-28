from app_fred.settings import *


# Database
# https://docs.djangoproject.com/en/1.10/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'mysql.connector.django',#'django.db.backends.mysql',
        'NAME': DEV_DB_NAME,
        'USER': DEV_DB_USER,
        'PASSWORD': DEV_DB_PASSWORD,
        'HOST': 'localhost',
        'PORT': '3306',
        'TEST': {
            'NAME': 'test_mbuehler',
        },
    }
}
print(DATABASES)
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
USE_TZ = False # django.db.utils.DatabaseError: Incorrect datetime value: '2017-02-28 10:50:04.839615+00:00' for column 'applied' at row 1

USE_X_FORWARDED_HOST = True # If using proxy

PATH_PPLIB = '/home/marcello/mbuehler.ch/PPLib'
STATIC_ROOT = '/home/marcello/mbuehler.ch/static_files'