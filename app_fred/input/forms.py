
from django import forms


class NewJobForm(forms.Form):
    email = forms.EmailField(label='E-mail')
    amt_key = forms.FloatField(label='AMT key')
    features_csv = forms.FileField(label='CSV file')
    target_mean_checkbox = forms.BooleanField(label='Query the target mean', initial=True)
    target_mean = forms.FloatField(label='Target mean')
