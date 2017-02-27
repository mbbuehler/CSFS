import json

from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from input.forms import NewJobForm

from input.models import JobFactory, Job

from input.forms import SelectJobForm


@require_http_methods(['GET'])
def index(request):
    return render(request, 'input/index.html', {})



@require_http_methods(['GET', 'POST'])
def job_new(request, job_id=-1):
    newjobform = NewJobForm()
    msg = ""
    if Job.objects.filter(pk=job_id).count() == 1:
        job = Job.objects.get(pk=job_id)
        form_fields = {'name', 'email', 'amt_key', 'query_target_mean', 'target_mean', 'target_mean_question'}
        form_data = {field: getattr(job, field) for field in form_fields}
        newjobform = NewJobForm(form_data)

    if request.method == 'POST':
        data = request.POST
        files = request.FILES
        newjobform = NewJobForm(data, files)
        if True: #newjobform.is_valid(): # does not work with csv reader TODO
            # csrfmiddlewaretoken=CUa2JJgJv22UnYj0nynyX10lfFNW8gOSujjv3mzAESMuGEPZl2Tkx1bUZDNpLCPT&email=marcel.buehler%40uzh.ch&amt_key=abcde&features_csv=features_student.csv&target_mean=0.5&target_mean_question=&job_id=
            # job = JobFactory.create(data, files) if data['job_id'] == '' else JobFactory.update(data)
            job = JobFactory.create(data, files)
            is_valid, msg = job.is_valid()
            if is_valid:
                return HttpResponseRedirect(reverse('job_status', kwargs=dict(job_id=job.pk)))
            newjobform = NewJobForm(initial=data)

    context = {'form': newjobform, 'messages': msg}

    return render(request, 'input/job_new.html', context)


@require_http_methods(['GET'])
def job_status_select(request):
    messages = {}
    form = SelectJobForm()
    if 'uuid' in request.GET:
        qs_job_id = Job.objects.\
            filter(uuid=request.GET['uuid']).\
            filter(email=request.GET['email']).\
            values('id')
        form = SelectJobForm(request.GET)
        if qs_job_id.count() == 0:
            messages['No job found'] = "Did you provide the correct credentials?"
        elif qs_job_id.count() > 1:
            messages['Several jobs found'] = "Sorry, this should not happen. Please contact the admin."
        else:
            job_id = qs_job_id.first()
            job_id = job_id['id']
            return HttpResponseRedirect(reverse('job_status', kwargs={'job_id': job_id}))

    context = {
        'form': form,
        'messages': messages
    }
    return render(request, 'input/job_select.html', context)


@require_http_methods(['GET'])
def job_status(request, job_id):
    context = {
        'job': Job.objects.get(pk=job_id),
        'links': {
            'edit': reverse('job_edit', kwargs={'job_id': job_id}),
            'start': reverse('job_start', kwargs={'job_id': job_id}),
            'result': reverse('job_result', kwargs={'job_id': job_id}),
        }
    }
    return render(request, 'input/job_status.html', context)


def job_start(request, job_id):
    job = Job.objects.get(pk=job_id)
    job = job.run()
    # start job here
    context = {
        'success': 'success',
        'message': 'Job successfully started. ',
        'job_status': job.status,
    }
    return JsonResponse(context)


def job_result(request, job_id=-1):
    job = Job.objects.get(pk=job_id)
    success, messages = job.finish()

    context = {
        'success': 'success' if success else 'failed',
        'job': job,
        'links': {
            'download_answers': job.path_out_crowd_answers,
            'download_features': job.path_out_feature_data,
        },
        'messages': messages
    }
    return render(request, 'input/job_result.html', context)
