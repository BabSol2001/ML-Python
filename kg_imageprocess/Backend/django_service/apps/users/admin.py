from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, UserProfile, BiomechanicalSession, FrameAnalysis


class UserProfileInline(admin.StackedInline):
    """نمایش و ویرایش مستقیم پروفایل در صفحه مدیریت کاربر"""
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'پروفایل بیومکانیکی و آنتروپومتری'
    fk_name = 'user'
    
    fieldsets = (
        ('۱. مشخصات فردی و آنتروپومتری', {
            'fields': (
                ('date_of_birth', 'gender'),
                ('height_cm', 'weight_kg'),
                ('torso_length_cm', 'femur_length_cm', 'tibia_length_cm'),
            )
        }),
        ('۲. وضعیت پزشکی و آناتومیک', {
            'fields': (
                'injury_history',
                'medical_conditions',
                'postural_deviations',
                'baseline_flexibility_limits',
                'physiological_status',
            )
        }),
        ('۳. سبک زندگی و هدف ورزشی', {
            'fields': (
                ('occupational_lifestyle', 'fitness_level'),
                ('target_sport', 'primary_goal'),
                'therapy_details',
            )
        }),
        ('۴. سبک تمرینی و تجهیزات', {
            'fields': (
                'preferred_training_style',
                'used_equipment_gear',
            )
        }),
    )


@admin.register(User)
class CustomUserAdmin(BaseUserAdmin):
    """سفارشی‌سازی نمایش مدل کاربر"""
    inlines = (UserProfileInline,)
    list_display = ('email', 'username', 'first_name', 'last_name', 'is_staff', 'is_active')
    list_filter = ('is_staff', 'is_superuser', 'is_active')
    search_fields = ('email', 'username', 'first_name', 'last_name')
    ordering = ('email',)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    """مدیریت مستقل پروفایل‌های کاربر"""
    list_display = ('user', 'gender', 'height_cm', 'weight_kg', 'fitness_level', 'primary_goal', 'created_at')
    list_filter = ('gender', 'fitness_level', 'primary_goal', 'physiological_status', 'occupational_lifestyle')
    search_fields = ('user__email', 'user__first_name', 'user__last_name', 'target_sport')
    readonly_fields = ('created_at', 'updated_at')


class FrameAnalysisInline(admin.TabularInline):
    """نمایش فریم‌های آنالیز شده در صفحه همان جلسه بیومکانیکی"""
    model = FrameAnalysis
    extra = 0
    readonly_fields = ('frame_number', 'keypoints_data', 'joint_angles', 'created_at')
    can_delete = True


@admin.register(BiomechanicalSession)
class BiomechanicalSessionAdmin(admin.ModelAdmin):
    """مدیریت نشست‌های تحلیل بیومکانیکی"""
    list_display = ('id', 'user', 'exercise_type', 'set_system', 'status', 'created_at')
    list_filter = ('status', 'set_system', 'exercise_type', 'created_at')
    search_fields = ('user__email', 'exercise_type', 'notes')
    inlines = [FrameAnalysisInline]
    readonly_fields = ('created_at',)


@admin.register(FrameAnalysis)
class FrameAnalysisAdmin(admin.ModelAdmin):
    """مدیریت فریم‌های تحلیل‌شده"""
    list_display = ('id', 'session', 'frame_number', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('session__id', 'session__user__email')
    readonly_fields = ('created_at',)