from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    SubscriptionPlanViewSet,
    UserSubscriptionViewSet,
    TransactionViewSet,
)

# تعریف روتر استاندارد REST API برای ماژول پرداخت
router = DefaultRouter()
router.register(r'plans', SubscriptionPlanViewSet, basename='subscription-plan')
router.register(r'subscriptions', UserSubscriptionViewSet, basename='user-subscription')
router.register(r'transactions', TransactionViewSet, basename='transaction')

urlpatterns = [
    # شامل شدن تمام مسیرهای CRUD خودکار ماژول payments
    path('', include(router.urls)),
]