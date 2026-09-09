import uuid
from django.db import models
from django.conf import settings


class Exercise(models.Model):
    """
    بانک مرجع حرکات ورزشی به همراه متغیرهای آناتومیک و بیومکانیکی
    """
    KINEMATIC_CHAIN_CHOICES = [
        ('open', 'Open Kinematic Chain (OKC)'),
        ('closed', 'Closed Kinematic Chain (CKC)'),
    ]

    PLANE_OF_MOTION_CHOICES = [
        ('sagittal', 'Sagittal Plane'),
        ('frontal', 'Frontal Plane'),
        ('transverse', 'Transverse Plane'),
        ('multi_planar', 'Multi-Planar'),
    ]

    DIFFICULTY_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
        ('elite', 'Elite'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=150, unique=True, verbose_name="نام حرکت")
    slug = models.SlugField(max_length=150, unique=True)
    description = models.TextField(blank=True, null=True, verbose_name="توضیحات و نحوه اجرا")
    
    # ویژگی‌های بیومکانیکی حرکت
    primary_muscles = models.JSONField(default=list, help_text="لیست عضلات اصلی هدف (مثلاً ['quadriceps', 'gluteus_maximus'])")
    secondary_muscles = models.JSONField(default=list, blank=True, help_text="لیست عضلات کمکی")
    kinematic_chain = models.CharField(max_length=20, choices=KINEMATIC_CHAIN_CHOICES, default='closed')
    plane_of_motion = models.CharField(max_length=20, choices=PLANE_OF_MOTION_CHOICES, default='sagittal')
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES, default='intermediate')
    
    # مفاصل کلیدی که FastAPI باید زوایای آن‌ها را در این حرکت بررسی کند
    target_joints_to_analyze = models.JSONField(
        default=list, 
        help_text="مفاصل بحرانی برای آنالیز هوش مصنوعی (مثلاً ['left_knee', 'right_knee', 'hip', 'lower_spine'])"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']
        verbose_name = "حرکت ورزشی"
        verbose_name_plural = "بانک حرکات ورزشی"

    def __str__(self):
        return self.name


class WorkoutPlan(models.Model):
    """
    برنامه تمرینی اختصاص داده شده به ورزشکار
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='workout_plans',
        verbose_name="ورزشکار"
    )
    title = models.CharField(max_length=200, verbose_name="عنوان برنامه")
    description = models.TextField(blank=True, null=True)
    is_active = models.BooleanField(default=True, verbose_name="برنامه فعال")
    start_date = models.DateField(auto_now_add=True)
    end_date = models.DateField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "برنامه تمرینی"
        verbose_name_plural = "برنامه‌های تمرینی"

    def __str__(self):
        return f"{self.title} - {self.user.email}"


class WorkoutPlanItem(models.Model):
    """
    جزئیات هر حرکت در یک برنامه تمرینی (تعداد ست، الگوی تکرار، Tempo و وزنه تجویزی)
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    plan = models.ForeignKey(WorkoutPlan, on_delete=models.CASCADE, related_name='items')
    exercise = models.ForeignKey(Exercise, on_delete=models.PROTECT, related_name='plan_items')
    
    order = models.PositiveIntegerField(default=1, help_text="ترتیب اجرای حرکت در جلسه")
    target_sets = models.PositiveIntegerField(default=3)
    target_reps_scheme = models.JSONField(default=list, help_text="الگوی تکرار تجویز شده مثلا [10, 8, 6]")
    prescribed_rpe = models.FloatField(blank=True, null=True, help_text="درجه فشار تجویز شده (RPE 1-10)")
    tempo = models.CharField(max_length=20, default="3-1-1-0", help_text="اکسنتوریک-مکث-کنسنتریک-مکث")
    rest_period_seconds = models.PositiveIntegerField(default=90)
    notes = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['order']
        verbose_name = "حرکت در برنامه"
        verbose_name_plural = "حرکات برنامه‌های تمرینی"

    def __str__(self):
        return f"{self.exercise.name} ({self.target_sets} sets) in {self.plan.title}"


class CompletedSessionSummary(models.Model):
    """
    خلاصه تحلیلی نهایی جلسه تمرینی که پس از پردازش ویدیو در FastAPI/Neo4j به Django برمی‌گردد
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='session_summaries')
    exercise = models.ForeignKey(Exercise, on_delete=models.PROTECT, related_name='session_summaries')
    
    # شناسه جلسه در Neo4j برای Traceability کامل
    neo4j_session_uuid = models.UUIDField(unique=True, help_text="آی‌دی تطبیق داده شده با Root Node در Neo4j")
    
    # خلاصه عملکرد بیومکانیکی
    total_completed_reps = models.PositiveIntegerField(default=0)
    valid_reps_count = models.PositiveIntegerField(default=0)
    invalid_reps_count = models.PositiveIntegerField(default=0)
    
    overall_form_score = models.FloatField(help_text="امتیاز فرم بیومکانیکی از ۱۰۰")
    detected_compensations = models.JSONField(
        default=list, 
        help_text="جبران‌های حرکتی و خطاهای شناساگر (مثلاً ['knee_valgus', 'excessive_forward_lean'])"
    )
    joint_range_of_motion_summary = models.JSONField(
        default=dict, 
        help_text="دامنه حرکتی ثبت شده برای مفاصل کلیدی (مثلاً {'knee_flexion_max': 118.5})"
    )
    
    perceived_rpe = models.FloatField(blank=True, null=True, help_text="فشار گزارش شده توسط ورزشکار")
    ai_feedback_summary = models.TextField(blank=True, null=True, help_text="توصیه و تحلیل نهایی متنی سیستم هوش مصنوعی")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "خلاصه جلسه تمرینی"
        verbose_name_plural = "خلاصه‌های جلسات تمرینی"

    def __str__(self):
        return f"Session {self.exercise.name} - Score: {self.overall_form_score} ({self.user.email})"