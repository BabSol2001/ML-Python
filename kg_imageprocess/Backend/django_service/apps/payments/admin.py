from django.contrib import admin
from .models import SubscriptionPlan, UserSubscription, Transaction


@admin.register(SubscriptionPlan)
class SubscriptionPlanAdmin(admin.ModelAdmin):
    """
    مدیریت پلن‌های اشتراک مالی سیستم
    """
    list_display = ('title', 'price_irr', 'duration_days', 'max_video_analyses', 'is_active', 'created_at')
    list_filter = ('is_active', 'duration_days')
    search_fields = ('title', 'description')
    prepopulated_fields = {'slug': ('title',)}
    ordering = ('price_irr',)


@admin.register(UserSubscription)
class UserSubscriptionAdmin(admin.ModelAdmin):
    """
    مدیریت و مشاهده اشتراک‌های فعال و منقضی‌شده کاربران
    """
    list_display = ('user', 'plan', 'status', 'start_date', 'end_date', 'used_analyses_count')
    list_filter = ('status', 'plan', 'start_date')
    search_fields = ('user__email', 'user__username', 'plan__title')
    autocomplete_fields = ['user', 'plan']
    readonly_fields = ('start_date',)
    ordering = ('-start_date',)


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    """
    مشاهده و پیگیری سوابق تراکنش‌های مالی و درگاه‌های پرداخت
    """
    list_display = ('id', 'user', 'plan', 'amount', 'status', 'payment_gateway', 'ref_id', 'created_at')
    list_filter = ('status', 'payment_gateway', 'created_at')
    search_fields = ('user__email', 'ref_id', 'authority', 'id')
    readonly_fields = ('id', 'created_at', 'updated_at')
    autocomplete_fields = ['user', 'plan']
    ordering = ('-created_at',)