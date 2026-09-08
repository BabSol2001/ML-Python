from django.contrib.auth.models import AbstractUser
from django.db import models
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
    """پروفایل بیومکانیکی و ورزشی کاربر جهت استفاده در آنتولوژی"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    age = models.PositiveIntegerField(null=True, blank=True, verbose_name="سن")
    height_cm = models.FloatField(null=True, blank=True, verbose_name="قد (سانتی‌متر)")
    weight_kg = models.FloatField(null=True, blank=True, verbose_name="وزن (کیلوگرم)")
    
    injury_history = models.JSONField(default=list, blank=True, verbose_name="سابقه آسیب‌دیدگی‌ها")
    
    fitness_level = models.CharField(
        max_length=50,
        choices=[
            ('beginner', 'مبتدی'),
            ('intermediate', 'متوسط'),
            ('advanced', 'پیشرفته'),
        ],
        default='beginner'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profile of {self.user.email}"


class BiomechanicalSession(models.Model):
    """نشست تحلیل حرکتی/بیومکانیکی کاربر"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    exercise_type = models.CharField(max_length=100, verbose_name="نوع حرکت ورزشی")
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
        return f"{self.exercise_type} - {self.user.email}"


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