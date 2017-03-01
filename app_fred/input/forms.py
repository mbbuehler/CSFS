from io import StringIO

import pandas as pd
from django import forms
import csv
from django.core.exceptions import ValidationError


class KrowDDCSVField(forms.FileField):
    """
    FileField that checks that the file is a valid CSV and if specified
    in `expected_fieldnames` checks that the fields match exactly.

    The widget's `accept` parameter is set to accept csv, text and excel files.

    TODO: Validate the entirety of the CSV file, not just the headers.
          But this should be enough for most use cases, as checking the
          whole file could be computationally expensive for huge files.

    source: https://djangosnippets.org/snippets/10596/

    extended by marcello

    Example usage:
        people = CSVField(expected_fieldnames=['First Name', 'Last Name'])
    """

    def __init__(self, *args, **kwargs):
        self.expected_fieldnames = kwargs.pop('expected_fieldnames', None)
        super(KrowDDCSVField, self).__init__(*args, **kwargs)
        self.error_messages['required'] = 'You must select a file'
        self.widget.attrs.update(
            {'accept': '.csv,'
                       'text/plain,'
                       'application/vnd.ms-excel,'
                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'})

    def clean(self, data, initial=None):

        value = super(KrowDDCSVField, self).clean(data)
        content = value.read().decode('utf-8')

        try:
            rows = content.splitlines()
            fieldnames = rows[0].split(',')
        except csv.Error:
            raise ValidationError('You must upload a valid CSV file')

        # Check the fieldnames are as specified, if requested
        if self.expected_fieldnames and fieldnames != self.expected_fieldnames:
            raise ValidationError(
                u'The CSV fields are expected to be "{0}"'.format(
                    u','.join(self.expected_fieldnames)))


        # each sibling pair needs a value in one of the two fields
        siblings = {'Question P(X)': 'P(X)', 'Question P(Y|X=1)': 'P(Y|X=1)', 'Question P(Y|X=0)': 'P(Y|X=0)'}
        reader = csv.DictReader(rows)
        for row in reader:
            # check if feature has a name
            if 'Feature' not in row or row['Feature'] == '':
                raise ValidationError(u'There is a feature name missing')
            # we need a question or a given mean
            for key in siblings:
                if not row[key] and not row[siblings[key]]:
                    raise ValidationError(u'Feature {} should have a valid entry in either {} or {}'.format(row['Feature'], key, siblings[key]))
            # all the means have to be floats and 0 <= mean <= 1
            for key in siblings.values():
                if row[key] != "":
                    try:
                        mean = float(row[key])
                        assert 0 <= mean <= 1
                    except:
                        raise ValidationError(u'Provided invalid value for feature {} and field {}'.format(row['Feature'], row['key']))

            # make sure all questions contain a question mark at the end
            for question in siblings:
                if row[question] != "":
                    if row[question][-1] != '?':
                        raise ValidationError(u'Please end all you questions with a question mark (please check feature {} and {}'.format(row['Feature'], row[question]))
        return value


class QuestionField(forms.CharField):

    #def __init__(self, *args, **kwargs):
     #   super(QuestionField, self).__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        value = super(QuestionField, self).clean(data)
        if data != "" and data[-1] != '?':
            raise ValidationError(u'Please end your question with a question mark (?)')
        return value


class NewJobForm(forms.Form):
    # TODO: implement validator for target mean (either mean or question has to be available, question with question mark)+ Check amt key for validity
    name = forms.CharField(label='Job title')
    email = forms.EmailField(label='Email')
    amt_key = forms.CharField(label='AMT access key ID', max_length=21, min_length=10) #TODO check min and max amt secret and key length
    amt_secret = forms.CharField(label='AMT secret access key', max_length=41, min_length=30, widget=forms.PasswordInput)
    features_csv = KrowDDCSVField(label='CSV file', required=True, expected_fieldnames=['Feature', 'Question P(Y|X=0)', 'Question P(Y|X=1)', 'Question P(X)', 'P(Y|X=0)', 'P(Y|X=1)', 'P(X)'])
    query_target_mean = forms.BooleanField(label='Query the target mean', initial=True, required=False)
    target_mean = forms.FloatField(label='Target mean', initial=0.5, required=False)
    target_mean_question = QuestionField(label='Question for target mean P(Y)', required=False)
    job_id = forms.IntegerField(widget=forms.HiddenInput(), required=False, label="")

    def clean(self):
        cleaned_data = super(NewJobForm, self).clean()
        print('here')
        print(self.data)
        query_target_mean = 'query_target_mean' in cleaned_data and self.data['query_target_mean'] == 'on'
        print('query?',query_target_mean)
        if query_target_mean:
            if cleaned_data['target_mean_question'] == '':
                raise ValidationError(u'Please provide a question for the target mean.')
        else:
            try:
                target_mean = float(cleaned_data['target_mean'])
                assert 0 <= target_mean <= 1
            except:
                raise ValidationError(u'Please provide a valid target mean')

        return cleaned_data


class SelectJobForm(forms.Form):
    uuid = forms.CharField(max_length=36)
    email = forms.EmailField(label='E-mail')




