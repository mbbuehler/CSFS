import csv
import uuid

import pandas as pd
import re
from django.db import models
from django.db.models import Q
from django.utils import timezone

from app_fred.settings import STATIC_URL
from math import log


class Job(models.Model):

    class Status:
        CREATED = 'created'
        PROCESSING = 'processing'
        FINISHED = 'finished'
        FAILED = 'failed'

    name = models.CharField(max_length=100, default="Job")
    email = models.EmailField()
    uuid = models.CharField(max_length=36)
    amt_key = models.TextField()
    amt_secret = models.TextField()
    query_target_mean = models.BooleanField(default=False)
    target_mean = models.FloatField()
    target_mean_question = models.TextField()
    status = models.TextField(default=Status.CREATED)
    date_created = models.DateField(default=timezone.now)
    date_started = models.DateField(blank=True, null=True)
    date_finished = models.DateField(blank=True, null=True)

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
        no_workers = 9
        price_per_feature = 0.10
        costs = no_features * price_per_feature * no_workers
        if self.query_target_mean:
            costs += 0.04 * no_workers
        return costs

    def run(self):
        # TODO: run AMT task
        self.status = self.Status.PROCESSING
        self.date_started = timezone.now()
        self.save()
        return self

    def amt_is_done(self):
        """
        Checks whether the AMT task is completed
        :return:
        """
        completed = True
        return completed

    def get_features(self):
        return self.feature_set.all()

    def validate_ready_for_finishing(self):
        if not 0 <= self.target_mean <= 1:
            return False
        for feature in self.get_features():
            for mean in ['p', 'p_0', 'p_1']:
                if not 0 <= getattr(feature, mean) <= 1:
                    return False
        return True

    def finish(self):
        """
        Calcs IG for each of its features
        :return: boolean True if computation happened and false if not
        """
        assert self.validate_ready_for_finishing()
        self.path_crowd_answers = 'static/job_files/output/56_raw.xlsx'

        if self.status == self.Status.PROCESSING and self.amt_is_done():
            # CrowdOutputProcessor(self.path_crowd_answers).save_answers()



            features = self.get_features()
            # read and clean crowd answers. save as csv
            # add crows answers to corresponding feature
            # call compute_ig() on each feature


            features = [feature.compute_ig(self.target_mean) for feature in features]

            # self.status == self.Status.FINISHED
            self.date_finished = timezone.now()
            self.save()
            return True
        return False

    # def get_feature_ranking(self):
    #     """
    #     Returns data for feature ranking
    #     :return:
    #     """
    #     fields = {'name', 'ig'}
    #     features = self._get_features()
    #     data = {
    #
    #     }


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

    def compute_ig(self, p_target):
        ig = Calculator().compute_ig(self.p, self.p_0, self.p_1, p_target)
        self.ig = max(ig, 0) # IG can't be lower than 0
        self.save()
        return self

    def get_fields_as_dict(self, fields):
        return {field: getattr(self, field) for field in fields}


class CrowdOutputProcessor:

    def __init__(self, path_crowd_answers):
        self.path_crowd_answers = path_crowd_answers

    def _preprocess_field(self, field):
        field = field.replace('\n', '') # remove new lines
        return field

    def _save_single(self, question, answer, worker_id):
        feature = None
        crowd_answer = None
        if Feature.objects.filter(q_p_0=question).exists():
            answer_type = CrowdAnswer.Type.P_0
            feature = Feature.objects.get(q_p_0=question)
        elif Feature.objects.filter(q_p_1=question).exists():
            answer_type = CrowdAnswer.Type.P_1
            feature = Feature.objects.get(q_p_1=question)
        elif Feature.objects.filter(q_p=question).exists():
            answer_type = CrowdAnswer.Type.P
            feature = Feature.objects.get(q_p=question)
        else:
            answer_type = CrowdAnswer.Type.P_TARGET
        if feature is not None:
            crowd_answer = CrowdAnswer.objects.create(feature=feature, type=answer_type, worker_id=worker_id, answer=answer)
        else:
            print('Feature is None')
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
        worker_id = row.answerUser

        field_answer = self._preprocess_field(row.answer)
        number_of_estimates = field_answer.count('::')
        for i in range(1, number_of_estimates+1):
            q, a = self._get_nth(i, field_answer)
            answer = self._save_single(q, a, worker_id)


    def save_answers(self):
        df_raw = pd.read_excel(self.path_crowd_answers)
        df_raw.loc[:1].apply(self.save_row, axis='columns')


class CrowdAnswer(models.Model):
    class Type:
        P = 'P(X)'
        P_0 = 'P(X|Y=0)'
        P_1 = 'P(X|Y=1)'
        P_TARGET = 'P(Y)'

    worker_id = models.CharField(max_length=20)
    answer = models.FloatField()
    feature = models.ForeignKey(
        Feature,
        on_delete=models.CASCADE
    )
    type = models.CharField(max_length=8)


class JobFactory:
    job_fields = {'name', 'email', 'amt_key', 'query_target_mean', 'target_mean', 'target_mean_question'}
    file_destination = 'static/job_files/input/'

    @classmethod
    def create(cls, data, files):
        """
        Creates a new job from form data
        :param data:
        :return:
        """
        d = {key: data[key] for key in cls.job_fields if key in data}
        d['uuid'] = uuid.uuid1()
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


class Calculator:

    def compute_ig(self, p, p_0, p_1, p_target):
        """

        :param instance: series
        :param h_x: H(x) entropy of target variable
        :return:
        """
        h_x = self.compute_H([p_target, 1-p_target])
        cond_h = self.compute_H_X_Y(p, p_0, p_1)
        return h_x - cond_h

    def compute_H_X_Y(self, p, p_0, p_1):
        """
        Conditonal entropy H(X|Y)
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