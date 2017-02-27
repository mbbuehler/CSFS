import csv
import subprocess
import uuid
import shlex

import numpy as np
import pandas as pd
import re
from django.db import models
from django.db.models import Q
from django.db.transaction import atomic
from django.utils import timezone

from math import log

from app_fred.settings import PATH_KROWDD_FILES, PATH_PPLIB, BASE_DIR

from app_fred.settings import STATIC_URL


class Job(models.Model):

    class Status:
        CREATED = 'created'
        PROCESSING = 'processing'
        FINISHED = 'finished'
        FAILED = 'failed'

    name = models.CharField(max_length=100, default="Estimation Job")
    email = models.EmailField()
    uuid = models.CharField(max_length=36)
    amt_key = models.TextField()
    amt_secret = models.TextField()
    query_target_mean = models.BooleanField(default=False)
    target_mean = models.FloatField()
    target_mean_question = models.TextField()
    status = models.TextField(default=Status.CREATED)
    path_questions = models.CharField(max_length=500, default="")
    path_out_crowd_answers = models.CharField(max_length=100, blank=True, null=True)
    path_out_feature_data = models.CharField(max_length=100, blank=True, null=True)
    date_created = models.DateField(default=timezone.now)
    date_started = models.DateField(blank=True, null=True)
    date_finished = models.DateField(blank=True, null=True)

    number_answers = 1
    sandbox = 1
    price_per_feature = 3

    path_crowd_answers = models.CharField(max_length=100)

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

    def estimate_costs(self):
        no_features = self.feature_set.count()
        no_workers = self.number_answers
        price_per_feature = 0.10
        costs = no_features * price_per_feature * no_workers
        if self.query_target_mean:
            costs += 0.04 * no_workers
        return costs

    @atomic
    def run(self):
        # TODO: adjust variables
        process_id = self.pk
        print('starting task')

        command = '(cd {}; sbt "run-main main.scala.ch.uzh.ifi.pdeboer.pplib.examples.gdpinfluence.krowdd_run {} {} {} {} {} {} {} {}")'.format(PATH_PPLIB, self.name, self.amt_key, self.amt_secret, process_id, self.number_answers, BASE_DIR+'/'+self.path_questions, self.sandbox, self.price_per_feature)
        process = subprocess.Popen(command, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
        print('> AMT task started')
        print(process)
        self.status = self.Status.PROCESSING
        self.date_started = timezone.now()
        self.save()
        return self

    def amt_is_done(self):
        """
        Checks whether the AMT task is completed
        :return:
        """
        features = pd.read_csv(self.path_questions, header=None)[0]
        queried_features = set([f[:-2] if re.search(r'_\d$', f) else f for f in features])
        number_queried_features = len(queried_features)
        number_of_answers_required = number_queried_features * self.number_answers

        number_of_answers_obtained = Queries.objects.filter(process_id=self.pk).filter(~Q(answer='None')).count()
        return number_of_answers_required == number_of_answers_obtained

    def get_features(self, include_target=False):
        if include_target:
            return self.feature_set.all()
        else:
            return self.feature_set.filter(is_target=False)

    def validate_ready_for_finishing(self):
        if not 0 <= self.target_mean <= 1:
            return False
        for feature in self.get_features():
            for mean in ['p', 'p_0', 'p_1']:
                if not 0 <= getattr(feature, mean) <= 1:
                    return False
        return True

    @atomic
    def finish(self):
        """
        Calcs IG for each of its features
        :return: boolean True if computation happened and false if not
        """
        done = False
        messages = dict()
        if not self.amt_is_done():
            messages['Still Processing'] = 'The answers are still being collected on AMT.'
        if self.status != self.Status.PROCESSING:
            messages['Incorrect Status'] = 'Invalid job status detected: {}'.format(self.status)
        # if self.status == self.Status.PROCESSING and self.amt_is_done():
        if self.status == self.Status.FINISHED and self.amt_is_done():
            self.path_crowd_answers = 'static/job_files/output/{}_raw.csv'.format(self.pk)
            answers_raw = Queries.objects.filter(process_id=self.pk).filter(~Q(answer='None'))
            dump(answers_raw, self.path_crowd_answers)
            number_saved = CrowdOutputProcessor(self.path_crowd_answers, self).save_answers()
            CrowdAggregator(self).aggregate_answers()
            print('> number of answers saved', number_saved)

            if self.query_target_mean:
                target_answers = CrowdAnswer.objects.filter(feature__job=self, feature__name='target')
                estimates = [a.answer for a in target_answers]
                self.target_mean = np.median(estimates)
                target_feature = Feature.objects.get(job=self, name='target')
                target_feature.p = self.target_mean
                target_feature.p_0 = 0
                target_feature.p_1 = 1
                target_feature.ig = Calculator().compute_ig(target_feature.p, target_feature.p_0, target_feature.p_1, self.target_mean)
                target_feature.save()

            assert self.validate_ready_for_finishing()

            features = self.get_features()
            # get target mean
            features = [feature.compute_ig(self.target_mean) for feature in features if not feature.is_target]
            print('> IG calculated')

            self.path_out_crowd_answers, no_answers = AnswerFileOutput(self).create()
            print('> {} answers saved'.format(no_answers))

            self.path_out_feature_data, no_features = FeatureFileOutput(self).create()
            print('> {} features saved'.format(no_features))

            self.status = self.Status.FINISHED
            self.date_finished = timezone.now()
            self.save()
            done = True
            messages['Job Finished'] = 'Job successfully finished.'
        return done, messages


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
    is_target = models.BooleanField(default=False)

    def is_valid(self):
        msg = {}
        is_valid = len(msg) == 0
        return is_valid, msg

    def compute_ig(self, p_target):
        ig = Calculator().compute_ig(self.p, self.p_0, self.p_1, p_target)
        self.ig = max(ig, 0) # IG can't be lower than 0
        self.save()
        return self

    def get_fields_as_dict(self, fields):
        return {field: getattr(self, field) for field in fields}

    @property
    def details(self):
        return "{} -- p: {} | p_0: {} | p_1: {}".format(self.name, self.p, self.p_0, self.p_1)


class AnswerFileOutput:
    def __init__(self, job):
        self.job = job

    def _extract_data(self, answer):
        return {
            'feature': answer.feature.name,
            'type': answer.type,
            'estimate': answer.answer,
            'worker_id': answer.worker_id
        }

    def create(self):
        answers = CrowdAnswer.objects.filter(feature__job=self.job)
        data = [self._extract_data(a) for a in answers]
        df = pd.DataFrame(data, columns=['feature', 'type', 'estimate', 'worker_id']).sort_values(['feature', 'type', 'estimate'])
        path_pref = "static/"
        path = "job_files/output/{}_crowd_answers.csv".format(self.job.pk)
        df.to_csv(path_pref+path, index=False)
        return path, len(df)


class FeatureFileOutput:
    def __init__(self, job):
        self.job = job

    def _extract_data(self, feature):
        if feature.name == 'target':
            return {
            'feature': feature.name,
            'P(X)': feature.p,
            'P(Y|X=0)': 0,
            'P(Y|X=1)': 1,
            'IG*': "",
            }
        else:
            return {
                'feature': feature.name,
                'P(X)': feature.p,
                'P(Y|X=0)': feature.p_0,
                'P(Y|X=1)': feature.p_1,
                'IG*': feature.ig,
            }

    def create(self):
        features = self.job.get_features(include_target=True)
        data = [self._extract_data(f) for f in features]
        df = pd.DataFrame(data, columns=['feature', 'P(X)', 'P(Y|X=0)', 'P(Y|X=1)', 'IG*']).sort_values(['IG*', 'feature'])
        path_pref = "static/"
        path = "job_files/output/{}_feature_data.csv".format(self.job.pk)
        df.to_csv(path_pref+path, index=False)
        return path, len(df)


class CrowdOutputProcessor:

    def __init__(self, path_crowd_answers, job):
        self.path_crowd_answers = path_crowd_answers
        self.job = job

    def _preprocess_field(self, field):
        field = field.replace('\n', '') # remove new lines
        return field

    def _save_single(self, question, answer, worker_id):
        feature = None
        crowd_answer = None
        if Feature.objects.filter(q_p_0=question).exists():
            answer_type = CrowdAnswer.Type.P_0
            feature = Feature.objects.get(q_p_0=question, job=self.job)
        elif Feature.objects.filter(q_p_1=question).exists():
            answer_type = CrowdAnswer.Type.P_1
            feature = Feature.objects.get(q_p_1=question, job=self.job)
        elif Feature.objects.filter(q_p=question).exists():
            answer_type = CrowdAnswer.Type.P
            feature = Feature.objects.get(q_p=question, job=self.job)
        if feature is not None:
            crowd_answer = CrowdAnswer.objects.create(feature=feature, type=answer_type, worker_id=worker_id, answer=answer)
        else:
            print('> Feature is None')
        return crowd_answer

    def _get_nth(self, n, field_answer):
        if n == 1:
            return self._get_first(field_answer)
        elif n == 2:
            return self._get_second(field_answer)
        else:
            return self._get_third(field_answer)

    def _get_first(self, field_answer):
        match = re.search(r'(<i>.*\?.*</i>.*?::\d+ \(\d+%\))', field_answer.rstrip()) # from <i> to first answer (::)
        extract = match.group(0)
        match_question = re.search(r'<i>(.*\?.*)</i>', extract)
        match_answer = re.search(r'::\d+ \((\d+)%\)', extract)
        question = match_question.group(1).strip()
        answer = float(match_answer.group(1))/100
        return question, answer

    def _get_second(self, field_answer):
        match = re.search(r'::\d+ \(\d+%\)(.*?::\d+ \(\d+%\))', field_answer.rstrip()) # from <i> to first answer (::)
        extract = match.group(1)
        # print(extract)
        match_question = re.search(r'(.*?\?)', extract)
        match_answer = re.search(r'::\d+ \((\d+)%\)', extract)
        question = match_question.group(1).strip()
        answer = float(match_answer.group(1))/100
        return question, answer

    def _get_third(self, field_answer):
        match = re.search(r'::\d+ \(\d+%\).*?::\d+ \(\d+%\)(.*?::\d+ \(\d+%\))', field_answer.rstrip()) # from <i> to first answer (::)
        extract = match.group(1)
        match_question = re.search(r'(.*?\?)', extract)
        match_answer = re.search(r'::\d+ \((\d+)%\)', extract)
        question = match_question.group(1).strip()
        answer = float(match_answer.group(1))/100
        return question, answer

    def save_row(self, row):
        worker_id = row.answeruser

        field_answer = self._preprocess_field(row.answer)
        number_of_estimates = field_answer.count('::')
        for i in range(1, number_of_estimates+1):
            q, a = self._get_nth(i, field_answer)
            answer = self._save_single(q, a, worker_id)
        row['saved'] = number_of_estimates
        return row

    @atomic
    def save_answers(self):
        df_raw = pd.read_csv(self.path_crowd_answers)
        df_raw = df_raw.apply(self.save_row, axis='columns')
        number_saved = sum(df_raw['saved'])
        return number_saved



class CrowdAggregator:
    def __init__(self, job):
        self.job = job

    @atomic
    def aggregate_answers(self):
        features = self.job.get_features(include_target=False)
        conditions = {'P(X)': 'p', 'P(Y|X=0)': 'p_0', 'P(Y|X=1)': 'p_1'}
        for f in features:
            for c in conditions:
                if getattr(f, conditions[c]) == -1:
                    answers = CrowdAnswer.objects.filter(feature=f, type=c)
                    assert answers.count() > 0
                    estimates = [a.answer for a in answers]
                    setattr(f, conditions[c], np.median(estimates))

            f.save()


class CrowdAnswer(models.Model):
    class Type:
        P = 'P(X)'
        P_0 = 'P(Y|X=0)'
        P_1 = 'P(Y|X=1)'
        P_TARGET = 'P(Y)'

    worker_id = models.CharField(max_length=20)
    answer = models.FloatField()
    feature = models.ForeignKey(
        Feature,
        on_delete=models.CASCADE
    )
    type = models.CharField(max_length=8)


class JobFactory:
    job_fields = {'name', 'email', 'amt_key', 'amt_secret', 'query_target_mean', 'target_mean', 'target_mean_question'}

    @classmethod
    @atomic
    def create(cls, data, files):
        """
        Creates a new job from form data
        :param data:
        :return:
        """
        d = {key: data[key] for key in cls.job_fields if key in data}
        d['query_target_mean'] = d['query_target_mean'] == 'on'
        d['uuid'] = uuid.uuid1()
        print(d)
        job = Job.objects.create(**d)

        # save file and features
        file_path = "{}{}.csv".format(PATH_KROWDD_FILES+'input/', job.pk)
        file = files['features_csv']
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        csv_data = csv.DictReader(open(file_path))
        features = [FeatureFactory.create(row, job) for row in csv_data]
        print(job.query_target_mean)
        if job.query_target_mean:
            job.target_mean = -1
            features.append(Feature(q_p=job.target_mean_question, is_target=True, job=job,  name='target'))

        Feature.objects.bulk_create(features)
        questions_path = cls.create_questions_csv(features, job)
        job.path_questions = questions_path
        job.save()
        return job

    @staticmethod
    def create_questions_csv(features, job):
        conditions = ['p_0', 'p_1', 'p']
        data = []
        for f in features:
            if f.name == 'target':
                data.append([f.name, f.q_p])
            else:
                for c in conditions:
                    if not 0 <= getattr(f, c) <= 1: # if we dont already have a value for it
                        identifier = "{}".format(f.name) if c == 'p' else "{}{}".format(f.name, c[-2:])
                        question = getattr(f, "q_{}".format(c))
                        data.append([identifier, question])

        questions = pd.DataFrame(data, columns=None)
        path = "{}input/{}_questions.csv".format(PATH_KROWDD_FILES, job.pk)
        questions.to_csv(path, index=False, header=False)
        return path

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
    @atomic
    def create(cls, row, job, is_target=False):
        """

        :param row: frow from csv with fields 'Feature', 'Question P(X|Y=0)', 'Question P(X|Y=1)', 'Question P(X)', 'P(X|Y=0)', 'P(X|Y=1)', 'P(X)'
        :return: Feature
        """
        data = {
            'name': row['Feature'],
            'q_p_0': row['Question P(Y|X=0)'],
            'q_p_1': row['Question P(Y|X=1)'],
            'q_p': row['Question P(X)'],
            'p_0': float(row['P(Y|X=0)']) if row['P(Y|X=0)'] != "" else -1,
            'p_1': float(row['P(Y|X=1)']) if row['P(Y|X=1)'] != "" else -1,
            'p': float(row['P(X)']) if row['P(X)'] != "" else -1,
            'job': job,
            'is_target': is_target,
        }
        feature = Feature(**data)
        return feature


class Calculator:

    def compute_ig(self, p, p_0, p_1, p_target):
        """

        :param instance: series
        :param h_x: H(x) entropy of target variable
        :return:
        """
        h_y = self.compute_H([p_target, 1-p_target])
        cond_h = self.compute_H_Y_X(p, p_0, p_1)
        return h_y - cond_h

    def compute_H_Y_X(self, p, p_0, p_1):
        """
        Conditonal entropy H(Y|X)
        :param p: P(X=1)
        :param p_0: P(X=1|Y=0)
        :param p_1: P(X=1|Y=1)
        :return: float
        """
        return (1-p) * self.compute_H([p_0, 1-p_0]) \
               + p * self.compute_H([p_1, 1-p_1])

    def compute_H(self, probabilities):
        """

        :param probabilities: list with probabilities, summing up to 1
        :return:
        """
        assert sum(probabilities) == 1
        return sum([-p_0 * log(p_0, 2) for p_0 in probabilities if p_0 != 0])


class Queries(models.Model):
    """
    DB Log from PPLib
    """
    process_id = models.IntegerField(blank=True, null=True)
    question = models.TextField(blank=True, null=True)
    fullquery = models.TextField(db_column='fullQuery', blank=True, null=True)  # Field name made lowercase.
    answer = models.TextField(blank=True, null=True)
    fullanswer = models.TextField(db_column='fullAnswer', blank=True, null=True)  # Field name made lowercase.
    paymentcents = models.IntegerField(db_column='paymentCents', blank=True, null=True)  # Field name made lowercase.
    fullproperties = models.TextField(db_column='fullProperties', blank=True, null=True)  # Field name made lowercase.
    questioncreationdate = models.DateTimeField(db_column='questionCreationDate', blank=True, null=True)  # Field name made lowercase.
    questionanswerdate = models.DateTimeField(db_column='questionAnswerDate', blank=True, null=True)  # Field name made lowercase.
    createdate = models.DateTimeField(db_column='createDate', blank=True, null=True)  # Field name made lowercase.
    answeruser = models.CharField(db_column='answerUser', max_length=255, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'queries'


import csv

def dump(qs, outfile_path):
    """
    http://palewi.re/posts/2009/03/03/django-recipe-dump-your-queryset-out-as-a-csv-file/
    Takes in a Django queryset and spits out a CSV file.

    Usage::

        >> from utils import dump2csv
        >> from dummy_app.models import *
        >> qs = DummyModel.objects.all()
        >> dump2csv.dump(qs, './data/dump.csv')

    Based on a snippet by zbyte64::

        http://www.djangosnippets.org/snippets/790/

    """
    model = qs.model
    writer = csv.writer(open(outfile_path, 'w'))

    headers = []
    for field in model._meta.fields:
        headers.append(field.name)
    writer.writerow(headers)

    for obj in qs:
        row = []
        for field in headers:
            val = getattr(obj, field)
            if callable(val):
                val = val()
            # if type(val) == unicode:
            # val = val.encode("utf-8")
            row.append(val)
        writer.writerow(row)