from django.db import models
from django.utils import timezone

class Job(models):

    class Status:
        CREATED = 'created'
        STARTED = 'started'
        PROCESSING = 'processing'
        FINISHED = 'finished'
        FAILED = 'failed'

    email = models.EmailField()
    amt_key = models.TextField()
    query_target_mean = models.BooleanField()
    target_mean = models.FloatField()
    target_question = models.TextField()
    status = models.TextField(default=Status.CREATED)
    date_created = models.DateField(default=timezone.now)
    date_started = models.DateField(blank=True)
    date_finished =models.DateField(blank=True)

class Feature(models):
    name = models.CharField()
    job = models.ForeignKey( # a feature belongs to one job, a job can have M features
        Job,
        on_delete=models.CASCADE,
    )
    q_p_0 = models.CharField() # Question for P(X|Y=0)
    q_p_1 = models.CharField()
    q_p = models.CharField()
    p_0 = models.FloatField() # P(X|Y=0)
    p_1 = models.FloatField()
    p = models.FloatField()
    ig = models.FloatField()



