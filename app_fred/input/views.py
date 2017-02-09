import json

from django.http import HttpResponse
from django.shortcuts import render
from django.urls import reverse

from input.forms import NewJobForm


def index(request):
    context = {
        'form': NewJobForm(),
        'links': {
            'onFormSubmit': reverse('new-job-on-submit')
        }

               }
    return render(request, 'input/index.html', context)

def on_form_submit(request):
    context = {
        'success': 'success',
        'user_token': 'ABCD',
        'costs': 33.50
    }
    json_data = json.dumps(context)
    return HttpResponse(json_data, content_type='application/json')
