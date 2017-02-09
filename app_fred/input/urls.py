from django.conf.urls import include, url
from django.contrib import admin
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'jobs/new/on-submit', views.on_form_submit, name='new-job-on-submit'),
]
