from typing import Any
from django.db.models import QuerySet
from rest_framework import viewsets
from rest_framework.request import Request

from .models import SubscriptionPlan, UserSubscription, Transaction
from .serializers import (
    SubscriptionPlanSerializer,
    UserSubscriptionSerializer,
    TransactionSerializer,
)


class SubscriptionPlanViewSet(viewsets.ModelViewSet):
    """
    مدیریت پلن‌های اشتراک سیستم (عمومی برای مشاهده، ادمین برای ویرایش)
    """
    queryset = SubscriptionPlan.objects.all()
    serializer_class = SubscriptionPlanSerializer
    lookup_field = 'id'


class UserSubscriptionViewSet(viewsets.ModelViewSet):
    """
    مشاهده و مدیریت اشتراک‌های فعال کاربران
    """
    queryset = UserSubscription.objects.all()
    serializer_class = UserSubscriptionSerializer

    def get_queryset(self) -> Any:
        """فیلتر اشتراک‌ها بر اساس کاربر: ?user_id=..."""
        queryset = UserSubscription.objects.all()
        req: Request = self.request  # type: ignore[assignment]
        user_id = req.query_params.get('user_id')
        if user_id:
            queryset = queryset.filter(user_id=user_id)
        return queryset


class TransactionViewSet(viewsets.ModelViewSet):
    """
    ثبت و استعلام وضعیت تراکنش‌های مالی و درگاه پرداخت
    """
    queryset = Transaction.objects.all()
    serializer_class = TransactionSerializer

    def get_queryset(self) -> Any:
        """فیلتر تراکنش‌ها بر اساس کاربر: ?user_id=..."""
        queryset = Transaction.objects.all()
        req: Request = self.request  # type: ignore[assignment]
        user_id = req.query_params.get('user_id')
        if user_id:
            queryset = queryset.filter(user_id=user_id)
        return queryset