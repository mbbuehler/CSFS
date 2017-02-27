from django.contrib import admin

from input.models import Job


class JobAdmin(admin.ModelAdmin):
    list_display = ('id', 'uuid', 'email', 'status')
    ordering = ('id', 'uuid', 'email', 'status')
    search_fields = ('uuid', 'email')


admin.site.register(Job, JobAdmin)