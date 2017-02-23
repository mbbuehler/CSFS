import json

from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from input.forms import NewJobForm

from input.models import JobFactory, Job


@require_http_methods(['GET', 'POST'])
def index(request):
    newjobform = NewJobForm()
    msg = ""
    if request.method == 'POST':
        # csrfmiddlewaretoken=CUa2JJgJv22UnYj0nynyX10lfFNW8gOSujjv3mzAESMuGEPZl2Tkx1bUZDNpLCPT&email=marcel.buehler%40uzh.ch&amt_key=abcde&features_csv=features_student.csv&target_mean=0.5&target_mean_question=&job_id=
        data = request.POST
        files = request.FILES
        job = JobFactory.create(data, files) if data['job_id'] == '' else JobFactory.update(data)
        is_valid, msg = job.is_valid()
        if is_valid:
            print('is valid')
            return HttpResponseRedirect(reverse('job_overview', kwargs=dict(job_id=job.pk)))
        else:
            print('is invalid')
            newjobform = NewJobForm(initial=data)
    context = {'form': newjobform, 'messages': msg}

    return render(request, 'input/index.html', context)

def job_overview(request, job_id):
    context = {}
    return render(request, 'input/job_overview.html', context)
