from django.contrib import admin
from .models import Exercise, WorkoutPlan, WorkoutPlanItem, CompletedSessionSummary


class WorkoutPlanItemInline(admin.TabularInline):
    """
    نمایش و مدیریت حرکات به صورت درخط (Inline) درون صفحه برنامه تمرینی
    """
    model = WorkoutPlanItem
    extra = 1
    autocomplete_fields = ['exercise']
    fields = ['order', 'exercise', 'target_sets', 'target_reps_scheme', 'prescribed_rpe', 'tempo', 'rest_period_seconds']


@admin.register(Exercise)
class ExerciseAdmin(admin.ModelAdmin):
    """
    مدیریت بانک مرجع حرکات ورزشی و شاخص‌های بیومکانیکی
    """
    list_display = ('name', 'kinematic_chain', 'plane_of_motion', 'difficulty', 'created_at')
    list_filter = ('kinematic_chain', 'plane_of_motion', 'difficulty')
    search_fields = ('name', 'description')
    prepopulated_fields = {'slug': ('name',)}
    ordering = ('name',)


@admin.register(WorkoutPlan)
class WorkoutPlanAdmin(admin.ModelAdmin):
    """
    مدیریت برنامه‌های تمرینی تجویز شده به ورزشکاران
    """
    list_display = ('title', 'user', 'is_active', 'start_date', 'end_date', 'created_at')
    list_filter = ('is_active', 'created_at')
    search_fields = ('title', 'user__email', 'user__username')
    autocomplete_fields = ['user']
    inlines = [WorkoutPlanItemInline]
    ordering = ('-created_at',)


@admin.register(WorkoutPlanItem)
class WorkoutPlanItemAdmin(admin.ModelAdmin):
    """
    مدیریت مجزای آیتم‌های تمرینی در صورت نیاز
    """
    list_display = ('plan', 'order', 'exercise', 'target_sets', 'prescribed_rpe', 'tempo')
    list_filter = ('exercise',)
    search_fields = ('plan__title', 'exercise__name')
    autocomplete_fields = ['plan', 'exercise']


@admin.register(CompletedSessionSummary)
class CompletedSessionSummaryAdmin(admin.ModelAdmin):
    """
    مشاهده خلاصه‌های تحلیلی ارسال‌شده از FastAPI و هوش مصنوعی
    """
    list_display = (
        'exercise', 
        'user', 
        'overall_form_score', 
        'total_completed_reps', 
        'valid_reps_count', 
        'invalid_reps_count', 
        'created_at'
    )
    list_filter = ('exercise', 'created_at')
    search_fields = ('user__email', 'exercise__name', 'neo4j_session_uuid')
    readonly_fields = ('id', 'created_at', 'neo4j_session_uuid')
    autocomplete_fields = ['user', 'exercise']
    ordering = ('-created_at',)