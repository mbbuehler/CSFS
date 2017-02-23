from django.test import TestCase

from input.models import CrowdOutputProcessor


class CrowdOutputProcessorTest(TestCase):
    def setUp(self):
        self.path = 'static/job_files/output/56_raw.xlsx'

    def test_save_answers(self):
        CrowdOutputProcessor(self.path).save_answers()

