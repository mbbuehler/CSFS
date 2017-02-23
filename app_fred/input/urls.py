from django.conf.urls import include, url
from django.contrib import admin
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^jobs/(?P<job_id>[0-9]+)/edit$', views.index, name='job_edit'),
    url(r'jobs/(?P<job_id>[0-9]+)/overview$', views.job_overview, name='job_overview'),
    url(r'jobs/(?P<job_id>[0-9]+)/status$', views.job_status, name='job_status'),
    url(r'jobs/(?P<job_id>[0-9]+)/start$', views.job_start, name='job_start'),
]
