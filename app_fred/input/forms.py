
from django import forms
import csv
from django.core.exceptions import ValidationError

class CSVField(forms.FileField):
    """
    FileField that checks that the file is a valid CSV and if specified
    in `expected_fieldnames` checks that the fields match exactly.

    The widget's `accept` parameter is set to accept csv, text and excel files.

    TODO: Validate the entirety of the CSV file, not just the headers.
          But this should be enough for most use cases, as checking the
          whole file could be computationally expensive for huge files.

    source: https://djangosnippets.org/snippets/10596/

    Example usage:
        people = CSVField(expected_fieldnames=['First Name', 'Last Name'])
    """

    def __init__(self, *args, **kwargs):
        self.expected_fieldnames = kwargs.pop('expected_fieldnames', None)
        super(CSVField, self).__init__(*args, **kwargs)
        self.error_messages['required'] = 'You must select a file'
        self.widget.attrs.update(
            {'accept': '.csv,'
                       'text/plain,'
                       'application/vnd.ms-excel,'
                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'})

    def clean(self, data, initial=None):
        value = super(CSVField, self).clean(data)
        reader = csv.reader(data)
        # Check it's a valid CSV file
        try:
            fieldnames = reader.next()
        except csv.Error:
            raise ValidationError('You must upload a valid CSV file')

        # Check the fieldnames are as specified, if requested
        if self.expected_fieldnames and fieldnames != self.expected_fieldnames:
            raise ValidationError(
                u'The CSV fields are expected to be "{0}"'.format(
                    u','.join(self.expected_fieldnames)))

        return value

class NewJobForm(forms.Form):
    email = forms.EmailField(label='E-mail')
    amt_key = forms.CharField(label='AMT key', max_length=21, min_length=20)
    features_csv = CSVField(label='CSV file', expected_fieldnames=['Feature',
                                                                   'Question P(X|Y=0)',
                                                                   'Question P(X|Y=1)',
                                                                   'Question P(X)',
                                                                   'P(X|Y=0)',
                                                                   'P(X|Y=1)',
                                                                   'P(X)'
                                                                   ]
                            )
    target_mean_checkbox = forms.BooleanField(label='Query the target mean', initial=True, required=False)
    target_mean = forms.FloatField(label='Target mean', initial=0.5, required=False)
    target_mean_question = forms.CharField(label='Question for target mean (P(Y))', required=False)
    job_id = forms.IntegerField(widget=forms.HiddenInput(), required=False)



