"""
WSGI config for app_fred project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.10/howto/deployment/wsgi/
"""

import os
import sys

path = '/home/marcello/mbuehler.ch/krowdd'

if path not in sys.path:
    sys.path.append(path)
print(sys.path)

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app_fred.settings_prod")

application = get_wsgi_application()
print('ended')