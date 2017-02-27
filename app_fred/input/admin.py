from django.contrib import admin

from input.models import Job, Feature


class JobAdmin(admin.ModelAdmin):
    list_display = ('id', 'uuid', 'email', 'status')
    ordering = ('id', 'uuid', 'email', 'status')
    search_fields = ('uuid', 'email')

class FeatureAdmin(admin.ModelAdmin):
    list_display = ('name', 'q_p_0', 'q_p_1', 'q_p', 'p_0', 'p_1', 'p', 'ig')
    ordering = ('name', 'q_p_0', 'q_p_1', 'q_p', 'p_0', 'p_1', 'p', 'ig')
    search_fields = ('name',)



admin.site.register(Job, JobAdmin)
admin.site.register(Feature, FeatureAdmin)