# -*- coding: utf-8 -*-
# Generated by Django 1.10.3 on 2017-02-23 14:19
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('input', '0006_job_uuid'),
    ]

    operations = [
        migrations.AddField(
            model_name='job',
            name='amt_secret',
            field=models.TextField(default='test'),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='job',
            name='uuid',
            field=models.CharField(max_length=36),
        ),
    ]