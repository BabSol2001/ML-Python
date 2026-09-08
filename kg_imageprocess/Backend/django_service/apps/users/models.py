from django.contrib.auth.models import AbstractUser
from django.db import models
from datetime import date
import uuid


class User(AbstractUser):
    """مدل سفارشی کاربر"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email


class UserProfile(models.Model):
    """پروفایل بیومکانیکی، فیزیولوژیک، اهداف، آنتروپومتری و ترجیحات سیستم‌های تمرینی کاربر"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    
    # --- ۱. دموگرافیک و آنتروپومتریک ---
    date_of_birth = models.DateField(null=True, blank=True, verbose_name="تاریخ تولد")
    gender = models.CharField(
        max_length=15,
        choices=[
            ('male', 'مرد'),
            ('female', 'زن'),
            ('other', 'سایر'),
        ],
        null=True,
        blank=True,
        verbose_name="جنسیت بیولوژیک"
    )
    height_cm = models.FloatField(null=True, blank=True, verbose_name="قد (سانتی‌متر)")
    weight_kg = models.FloatField(null=True, blank=True, verbose_name="وزن (کیلوگرم)")
    
    # آنتروپومتری پیشرفته جهت دقیق‌تر کردن محاسبات بیومکانیکی (زاویه مفاصل و مرکز ثقل)
    torso_length_cm = models.FloatField(null=True, blank=True, verbose_name="طول تنه (سانتی‌متر)")
    femur_length_cm = models.FloatField(null=True, blank=True, verbose_name="طول استخوان ران (سانتی‌متر)")
    tibia_length_cm = models.FloatField(null=True, blank=True, verbose_name="طول ساق پا (سانتی‌متر)")
    
    # --- ۲. سوابق پزشکی، آسیب‌دیدگی و وضعیت‌های آناتومیک ---
    injury_history = models.JSONField(default=list, blank=True, verbose_name="سابقه آسیب‌دیدگی‌های اسکلتی-عضلانی")
    medical_conditions = models.JSONField(default=list, blank=True, verbose_name="بیماری‌های زمینه‌ای و مزمن")
    postural_deviations = models.JSONField(default=list, blank=True, verbose_name="ناهنجاری‌های قامتی (مانند زانوی ضربدری، پشت گرد، گودی کمر)")
    baseline_flexibility_limits = models.JSONField(default=dict, blank=True, verbose_name="محدودیت‌های انعطاف‌پذیری و دامنه حرکتی مفاصل")
    
    physiological_status = models.CharField(
        max_length=50,
        choices=[
            ('none', 'عادی / بدون شرایط خاص'),
            ('pregnant', 'باردار'),
            ('menstruating', 'دوره ماهانه / قاعدگی'),
            ('postpartum', 'پس از زایمان'),
        ],
        default='none',
        verbose_name="وضعیت فیزیولوژیک جاری"
    )
    
    # --- ۳. سبک زندگی و شغل ---
    occupational_lifestyle = models.CharField(
        max_length=50,
        choices=[
            ('sedentary_desk', 'نشسته / پشت میزی'),
            ('active_standing', 'ایستاده / فعال'),
            ('heavy_labor', 'کار فیزیکی سنگین'),
        ],
        default='sedentary_desk',
        verbose_name="الگوی حرکتی و وضعیت شغلی روزمره"
    )

    # --- ۴. اهداف ورزشی، فیزیوتراپی و تجهیزات ---
    target_sport = models.CharField(
        max_length=100,
        default='general_fitness',
        verbose_name="رشته ورزشی تخصصی"
    )
    
    primary_goal = models.CharField(
        max_length=50,
        choices=[
            ('general_fitness', 'تناسب اندام عمومی'),
            ('fat_loss', 'کاهش وزن و چربی‌سوزی'),
            ('hypertrophy_general', 'عضله‌سازی عمومی'),
            ('hypertrophy_glutes', 'هایپرتروفی و فرم‌دهی تخصصی سرینی (کپل)'),
            ('explosive_power', 'توسعه توان و قدرت انفجاری'),
            ('max_strength', 'افزایش حداکثر قدرت (Powerlifting)'),
            ('physiotherapy_rehab', 'فیزیوتراپی و توانبخشی پس از آسیب/جراحی'),
            ('corrective_exercise', 'حرکات اصلاحی (اصلاح پاسچر و ناهنجاری قامتی)'),
        ],
        default='general_fitness',
        verbose_name="هدف اصلی از تمرین"
    )
    
    therapy_details = models.JSONField(
        default=dict, 
        blank=True, 
        verbose_name="دستورالعمل و محدودیت‌های فیزیوتراپی/پزشکی"
    )

    # --- ۵. ترجیحات سبک تمرینی و تجهیزات حمایتی ---
    preferred_training_style = models.CharField(
        max_length=50,
        choices=[
            ('straight_sets', 'ست‌های معمولی و استاندارد'),
            ('supersets', 'سوپرست (دو حرکت متوالی)'),
            ('giant_sets', 'جاینت ست / سوپرست ۴ تایی'),
            ('pyramid_sets', 'سیستم هرمی (کاهش تکرار / افزایش وزنه)'),
            ('reverse_pyramid', 'هرمی معکوس'),
            ('high_rep_endurance', 'تکرارهای بالا و استقامتی (۱۵+ تکرار)'),
            ('drop_sets', 'دراپ ست (کاهش وزنه بدون استراحت)'),
        ],
        default='straight_sets',
        verbose_name="ترجیح سبک و سیستم تمرینی"
    )
    
    used_equipment_gear = models.JSONField(
        default=list, 
        blank=True, 
        verbose_name="تجهیزات تمرینی مورداستفاده (مانند کمربند، کفش وزنه برداری، زانوبند)"
    )

    fitness_level = models.CharField(
        max_length=50,
        choices=[
            ('beginner', 'مبتدی / بیمار تحت درمان'),
            ('intermediate', 'متوسط'),
            ('advanced', 'پیشرفته'),
        ],
        default='beginner',
        verbose_name="سطح آمادگی جسمانی"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def age(self) -> int | None:
        """محاسبه دقیق و پویای سن بر اساس تاریخ تولد"""
        if self.date_of_birth:
            today = date.today()
            return today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
        return None

    def __str__(self):
        return f"Profile of {self.user.email}"


class BiomechanicalSession(models.Model):
    """نشست تحلیل حرکتی/بیومکانیکی کاربر با جزئیات پروتکل و متغیرهای تمرینی"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    exercise_type = models.CharField(max_length=100, verbose_name="نوع حرکت ورزشی")
    
    # --- متغیرهای ساختاری ست و تکرار در جلسه ---
    set_system = models.CharField(
        max_length=50,
        choices=[
            ('straight', 'ست معمولی'),
            ('superset', 'سوپرست'),
            ('giant_set', 'جاینت ست (۴ تایی)'),
            ('pyramid', 'هرمی (مثلاً ۱۲-۱۰-۸)'),
            ('drop_set', 'دراپ ست'),
        ],
        default='straight',
        verbose_name="سیستم ساختار ست"
    )
    
    target_reps_scheme = models.JSONField(
        default=list, 
        blank=True, 
        verbose_name="الگوی تکرار هدف در ست‌ها (مثلاً [12, 10, 8] یا [15, 15, 15])"
    )
    
    rest_interval_seconds = models.PositiveIntegerField(
        default=60, 
        verbose_name="زمان استراحت محاسبه‌شده بین ست‌ها (ثانیه)"
    )
    
    tempo = models.CharField(
        max_length=20, 
        default='2-0-2-0', 
        blank=True,
        verbose_name="ریتم اجرای حرکت (Tempo: Eccentric-Pause-Concentric-Pause)"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'در انتظار پردازش'),
            ('processing', 'در حال پردازش خط لوله AI'),
            ('completed', 'تکمیل‌شده'),
            ('failed', 'خطا در پردازش'),
        ],
        default='pending'
    )
    notes = models.TextField(blank=True, null=True, verbose_name="توضیحات و ملاحظات")

    def __str__(self):
        return f"{self.exercise_type} ({self.set_system}) - {self.user.email}"


class FrameAnalysis(models.Model):
    """ذخیره خروجی تحلیل فریم به فریم هوش مصنوعی (FastAPI)"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(BiomechanicalSession, on_delete=models.CASCADE, related_name='frames')
    frame_number = models.PositiveIntegerField(verbose_name="شماره فریم")
    keypoints_data = models.JSONField(verbose_name="مختصات کی‌پوینت‌ها")
    joint_angles = models.JSONField(blank=True, null=True, verbose_name="زوایای مفاصل")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['frame_number']

    def __str__(self):
        return f"Session {self.session.id} - Frame {self.frame_number}"