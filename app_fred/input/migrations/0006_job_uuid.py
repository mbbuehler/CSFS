# -*- coding: utf-8 -*-
# Generated by Django 1.10.3 on 2017-02-23 13:07
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('input', '0005_auto_20170223_1044'),
    ]

    operations = [
        migrations.AddField(
            model_name='job',
            name='uuid',
            field=models.CharField(default='abc', max_length=20),
            preserve_default=False,
        ),
    ]