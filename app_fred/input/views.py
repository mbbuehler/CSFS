from django.shortcuts import render

from input.forms import NewJobForm


def index(request):
    context = {'form': NewJobForm()}
    return render(request, 'input/index.html', context)