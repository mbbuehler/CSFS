
from django import forms


class NewJobForm(forms.Form):
    email = forms.EmailField(label='E-mail')
    amt_key = forms.CharField(label='AMT key', max_length=21, min_length=20)
    features_csv = forms.FileField(label='CSV file')
    target_mean_checkbox = forms.BooleanField(label='Query the target mean', initial=True, required=False)
    target_mean = forms.FloatField(label='Target mean', initial=0.5, required=False)
