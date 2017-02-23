from django.conf.urls import include, url
from django.contrib import admin
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^jobs/new', views.index, name='job_new'),
    url(r'^jobs/(?P<job_id>[0-9]+)/edit$', views.index, name='job_edit'),
    url(r'jobs/(?P<job_id>[0-9]+)/status$', views.job_overview, name='job_overview'),
    url(r'jobs/(?P<job_id>[0-9]+)/result', views.job_result, name='job_result'),
    url(r'jobs/(?P<job_id>[0-9]+)/start$', views.job_start, name='job_start'),
    url(r'jobs/overview$', views.job_overview, name='job_overview_all'),
    url(r'jobs/result', views.job_result, name='job_result_all'),
]
