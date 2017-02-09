#!/usr/bin/env python
import os
import sys



from django.db import connections
from django.db.utils import OperationalError


if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app_fred.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError:
        # The above import may fail for some other reason. Ensure that the
        # issue is really that Django is missing to avoid masking other
        # exceptions on Python 2.
        try:
            import django
        except ImportError:
            raise ImportError(
                "Couldn't import Django. Are you sure it's installed and "
                "available on your PYTHONPATH environment variable? Did you "
                "forget to activate a virtual environment?"
            )
        raise
    execute_from_command_line(sys.argv)
    db_conn = connections['default']
    try:
        c = db_conn.cursor()
    except OperationalError:
        connected = False
    else:
        connected = True
    print(connected)
