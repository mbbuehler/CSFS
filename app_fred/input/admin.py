from django.contrib import admin

from input.models import Job, Feature, CrowdAnswer


class JobAdmin(admin.ModelAdmin):
    list_display = ('id', 'uuid', 'email', 'status')
    ordering = ('id', 'uuid', 'email', 'status')
    search_fields = ('uuid', 'email')

class FeatureAdmin(admin.ModelAdmin):
    list_display = ('name', 'q_p_0', 'q_p_1', 'q_p', 'p_0', 'p_1', 'p', 'ig')
    ordering = ('name', 'q_p_0', 'q_p_1', 'q_p', 'p_0', 'p_1', 'p', 'ig')
    search_fields = ('name',)

class CrowdAnswerAdmin(admin.ModelAdmin):
    def feature_name(self, obj):
        return obj.feature.name
    list_display = ('answer', 'type', 'worker_id', 'feature_name')
    ordering = ('answer', 'type', 'worker_id')
    search_fields = ('type',)



admin.site.register(Job, JobAdmin)
admin.site.register(Feature, FeatureAdmin)
admin.site.register(CrowdAnswer, CrowdAnswerAdmin)