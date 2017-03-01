from django.core.files.uploadedfile import SimpleUploadedFile
from django.http import QueryDict
from django.test import TestCase
from django.utils.datastructures import MultiValueDict

from input.models import CrowdOutputProcessor


# class CrowdOutputProcessorTest(TestCase):
#     def setUp(self):
#         self.path = 'static/job_files/output/56_raw.xlsx'
#
#     def test_save_answers(self):
#         CrowdOutputProcessor(self.path).save_answers()
from input.forms import NewJobForm


class NewJobFormTest(TestCase):

    def test_validate_valid_csv(self):
        uploaded_file = open('input/test_data/features_student_small.csv', 'rb')
        data = {'amt_secret': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY', 'name': 'Estimate Student Performance', 'amt_key': 'AKIAIOSFODNN7EXAMPLE', 'email': 'marcel.buehler@uzh.ch', 'job_id': '', 'query_target_mean': 'on', 'csrfmiddlewaretoken': 'ruca4jTObc7QPKbxjoBKfKATGuyWERv7BbrMHUM6AEXRDM37lqX5A7s9MNsVtlRW', 'target_mean': '0.5', 'target_mean_question': 'What is it?'}
        files = {'features_csv': SimpleUploadedFile(uploaded_file.name, uploaded_file.read(), content_type='text/csv')}
        form = NewJobForm(data, files)
        # print('valid form')
        # print(form.errors)
        self.assertTrue(form.is_valid(), 'Should be a valid form')

    def test_validate_missing_feature_name(self):
        uploaded_file = open('input/test_data/features_student_missing_feature_name.csv', 'rb')
        data = {'amt_secret': 'bPxRfiCYEXAMPLEKEY/i5WmI/VBW7ZGfW7E6', 'name': 'Estimate Student Performance', 'amt_key': 'AKIAIOSFODNN7EXAMPLE', 'email': 'marcel.buehler@uzh.ch', 'job_id': '', 'query_target_mean': 'on', 'csrfmiddlewaretoken': 'ruca4jTObc7QPKbxjoBKfKATGuyWERv7BbrMHUM6AEXRDM37lqX5A7s9MNsVtlRW', 'target_mean': '0.5', 'target_mean_question': 'What is it?'}
        files = {'features_csv': SimpleUploadedFile(uploaded_file.name, uploaded_file.read(), content_type='text/csv')}
        form = NewJobForm(data, files)
        self.assertFalse(form.is_valid(), 'Should have invalid fields and incorrect feature entries')

    def test_validate_invalid_mean(self):
        uploaded_file = open('input/test_data/features_student_incorrect_mean.csv', 'rb')
        data = {'amt_secret': 'bPxRfiCYEXAMPLEKEY/i5WmI/VBW7ZGfW7E6', 'name': 'Estimate Student Performance', 'amt_key': 'AKIAIOSFODNN7EXAMPLE', 'email': 'marcel.buehler@uzh.ch', 'job_id': '', 'query_target_mean': 'on', 'csrfmiddlewaretoken': 'ruca4jTObc7QPKbxjoBKfKATGuyWERv7BbrMHUM6AEXRDM37lqX5A7s9MNsVtlRW', 'target_mean': '0.5', 'target_mean_question': 'What is it?'}
        files = {'features_csv': SimpleUploadedFile(uploaded_file.name, uploaded_file.read(), content_type='text/csv')}
        form = NewJobForm(data, files)
        print(form.errors)
        self.assertFalse(form.is_valid(), 'Should have invalid fields and incorrect feature entries')
