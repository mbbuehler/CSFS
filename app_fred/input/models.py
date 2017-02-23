import csv

from django.db import models
from django.utils import timezone


class Job(models.Model):

    class Status:
        CREATED = 'created'
        STARTED = 'started'
        PROCESSING = 'processing'
        FINISHED = 'finished'
        FAILED = 'failed'

    email = models.EmailField()
    amt_key = models.TextField()
    query_target_mean = models.BooleanField(default=False)
    target_mean = models.FloatField()
    target_mean_question = models.TextField()
    status = models.TextField(default=Status.CREATED)
    date_created = models.DateField(default=timezone.now)
    date_started = models.DateField(blank=True, null=True)
    date_finished = models.DateField(blank=True, null=True)

    def is_valid(self):
        msg = {}
        features = Feature.objects.filter(job__pk=self.pk)
        if len(features) < 1:
            msg['Feature Count'] = "No Features found."
        for f in features:
            is_valid, msg = f.is_valid()
            if not is_valid:
                msg['Feature {} invalid'.format(f.name)] = msg
        is_valid = len(msg) == 0
        return is_valid, msg


class Feature(models.Model):
    name = models.CharField(max_length=100)
    job = models.ForeignKey( # a feature belongs to one job, a job can have M features
        Job,
        on_delete=models.CASCADE,
    )
    q_p_0 = models.CharField(max_length=1000) # Question for P(X|Y=0)
    q_p_1 = models.CharField(max_length=1000)
    q_p = models.CharField(max_length=1000)
    p_0 = models.FloatField(blank=True, default=-1) # P(X|Y=0)
    p_1 = models.FloatField(blank=True, default=-1)
    p = models.FloatField(blank=True, default=-1)
    ig = models.FloatField(blank=True, default=-1)

    def is_valid(self):
        msg = {}
        is_valid = len(msg) == 0
        return is_valid, msg


class JobFactory:
    job_fields = {'email', 'amt_key', 'query_target_mean', 'target_mean', 'target_mean_question'}
    file_destination = 'job_files/input/'

    @classmethod
    def create(cls, data, files):
        """
        Creates a new job from form data
        :param data:
        :return:
        """
        d = {key: data[key] for key in cls.job_fields if key in data}
        job = Job.objects.create(**d)

        # save file and features
        file_path = "{}{}.csv".format(cls.file_destination, job.pk)
        file = files['features_csv']
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        csv_data = csv.DictReader(open(file_path))
        features = [FeatureFactory.create(row, job) for row in csv_data]
        Feature.objects.bulk_create(features)
        return job

    @classmethod
    def update(cls, data):
        job = Job.objects.get(pk=data['job_id'])
        for field in cls.job_fields:
            job.field = data[field]
        job.save()
        return job


class FeatureFactory:
    """
    Creates Features
    """
    @classmethod
    def create(cls, row, job):
        """

        :param row: frow from csv with fields 'Feature', 'Question P(X|Y=0)', 'Question P(X|Y=1)', 'Question P(X)', 'P(X|Y=0)', 'P(X|Y=1)', 'P(X)'
        :return: Feature
        """
        data = {
            'name': row['Feature'],
            'q_p_0': row['Question P(X|Y=0)'],
            'q_p_1': row['Question P(X|Y=1)'],
            'q_p': row['Question P(X)'],
            'p_0': float(row['P(X|Y=0)']) if row['P(X|Y=0)'] != "" else None,
            'p_1': float(row['P(X|Y=1)']) if row['P(X|Y=1)'] != "" else None,
            'p': float(row['P(X)']) if row['P(X)'] != "" else None,
            'job': job
        }
        feature = Feature(**data)
        return feature