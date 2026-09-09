from rest_framework import serializers
from .models import SubscriptionPlan, UserSubscription, Transaction


class SubscriptionPlanSerializer(serializers.ModelSerializer):
    """
    سریالایزر پلن‌های اشتراک
    """
    class Meta:
        model = SubscriptionPlan
        fields = [
            'id',
            'title',
            'slug',
            'price_irr',
            'duration_days',
            'max_video_analyses',
            'is_active',
            'features',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class UserSubscriptionSerializer(serializers.ModelSerializer):
    """
    سریالایزر مشاهده و مدیریت اشتراک‌های فعال کاربران
    """
    plan_details = SubscriptionPlanSerializer(source='plan', read_only=True)

    class Meta:
        model = UserSubscription
        fields = [
            'id',
            'user',
            'plan',
            'plan_details',
            'status',
            'start_date',
            'end_date',
            'used_analyses_count',
        ]
        read_only_fields = ['id', 'start_date']


class TransactionSerializer(serializers.ModelSerializer):
    """
    سریالایزر تراکنش‌های مالی و درگاه پرداخت
    """
    plan_title = serializers.ReadOnlyField(source='plan.title')

    class Meta:
        model = Transaction
        fields = [
            'id',
            'user',
            'plan',
            'plan_title',
            'amount',
            'status',
            'payment_gateway',
            'authority',
            'ref_id',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']