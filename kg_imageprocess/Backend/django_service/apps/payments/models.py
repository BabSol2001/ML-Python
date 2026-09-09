import uuid
from django.db import models
from django.conf import settings


class SubscriptionPlan(models.Model):
    """
    تعریف پلن‌های اشتراک سیستم (مثلاً برنز، نقره‌ای، طلایی)
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=100, verbose_name="عنوان پلن")
    slug = models.SlugField(max_length=100, unique=True)
    price_irr = models.DecimalField(max_digits=12, decimal_places=0, verbose_name="قیمت (ریال)")
    duration_days = models.PositiveIntegerField(default=30, verbose_name="مدت اعتبار (روز)")
    max_video_analyses = models.PositiveIntegerField(
        default=10, 
        help_text="حداکثر تعداد آنالیز ویدیو مجاز در این دوره"
    )
    is_active = models.BooleanField(default=True, verbose_name="فعال")
    features = models.JSONField(default=list, blank=True, help_text="لیست ویژگی‌های پلن به صورت JSON")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['price_irr']
        verbose_name = "پلن اشتراک"
        verbose_name_plural = "پلن‌های اشتراک"

    def __str__(self):
        return f"{self.title} - {self.price_irr:,} IRR"


class UserSubscription(models.Model):
    """
    وضعیت اشتراک فعال کاربر در سیستم
    """
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('expired', 'Expired'),
        ('canceled', 'Canceled'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.CASCADE, 
        related_name='subscriptions',
        verbose_name="کاربر"
    )
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.PROTECT, related_name='user_subscriptions')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    
    start_date = models.DateTimeField(auto_now_add=True)
    end_date = models.DateTimeField(verbose_name="تاریخ انقضا")
    used_analyses_count = models.PositiveIntegerField(default=0, verbose_name="تعداد آنالیزهای استفاده شده")

    class Meta:
        ordering = ['-start_date']
        verbose_name = "اشتراک کاربر"
        verbose_name_plural = "اشتراک‌های کاربران"

    def __str__(self):
        return f"{self.user.email} - {self.plan.title} ({self.status})"


class Transaction(models.Model):
    """
    سوابق تراکنش‌ها و پرداخت‌های مالی کاربر
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('successful', 'Successful'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='transactions')
    plan = models.ForeignKey(SubscriptionPlan, on_delete=models.PROTECT, related_name='transactions')
    
    amount = models.DecimalField(max_digits=12, decimal_places=0, verbose_name="مبلغ")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    payment_gateway = models.CharField(max_length=50, default='zarinpal', verbose_name="درگاه پرداخت")
    authority = models.CharField(max_length=255, blank=True, null=True, verbose_name="کد Authority درگاه")
    ref_id = models.CharField(max_length=255, blank=True, null=True, verbose_name="کد پیگیری تراکنش")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "تراکنش مالی"
        verbose_name_plural = "تراکنش‌های مالی"

    def __str__(self):
        return f"Tx: {self.id} - {self.user.email} - {self.status}"